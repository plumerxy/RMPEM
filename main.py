import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import datetime
import argparse

import sys

import yaml
import json
import hashlib

from tqdm import tqdm
from copy import deepcopy

# Import user-defined models and interpretability methods
from models import *
from interpretability_methods import *
from interpretability_methods import ZORRO
import interpretability_methods
# Import user-defined functions
from utilities.ground_truth_loaders import load_dataset_ground_truth
from utilities.load_data import load_model_data
from utilities.util import graph_to_tensor
from utilities.output_results import output_to_images
from utilities.metrics import auc_scores, compute_metric, get_accuracy, get_accuracy_max
from utilities.GNN4BAmotif import model_selector, train_model_graph, data2nxgraph

# Define timer list to report running statistics
timing_dict = {"forward": [], "backward": []}
run_statistics_string = "Run statistics: \n"


def loop_dataset(g_list, classifier, sample_idxes, config, dataset_features, optimizer=None):
	'''
	:param g_list: list of graphs to trainover
	:param classifier: the initialised classifier
	:param sample_idxes: indexes to mark the training and test graphs
	:param config: Run configurations as stated in config.yml
	:param dataset_features: Dataset features obtained from load_data.py
	:param optimizer: optimizer to use
	:return: average loss and other model performance metrics
	'''
	n_samples = 0
	all_targets = []
	all_scores = []
	total_loss = []

	# Determine batch size and initialise progress bar (pbar)
	bsize = max(config["general"]["batch_size"], 1)
	total_iters = (len(sample_idxes) + (bsize - 1) *
				   (optimizer is None)) // bsize
	pbar = tqdm(range(total_iters), unit='batch')

	# Create temporary timer dict to store timing data for this loop
	temp_timing_dict = {"forward": [], "backward": []}

	# For each batch
	for pos in pbar:
		selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

		batch_graph = [g_list[idx] for idx in selected_idx]
		targets = [g_list[idx].label for idx in selected_idx]
		all_targets += targets

		# Convert graph to tensor
		node_feat, n2n, subg = graph_to_tensor(
			batch_graph, dataset_features["feat_dim"],
			dataset_features["edge_feat_dim"], cmd_args.cuda)
		# Get graph labels of all graphs in batch
		labels = torch.LongTensor(len(batch_graph))

		for i in range(len(batch_graph)):
			labels[i] = batch_graph[i].label

		if cmd_args.cuda == 1:
			labels = labels.cuda()

		# Perform training
		start_forward = time.perf_counter()
		output = classifier(node_feat, n2n, subg, batch_graph)
		temp_timing_dict["forward"].append(time.perf_counter() - start_forward)
		logits = F.log_softmax(output, dim=1)
		prob = F.softmax(logits, dim=1)

		# Calculate accuracy and loss
		loss = classifier.loss(logits, labels)
		pred = logits.data.max(1, keepdim=True)[1]
		acc = pred.eq(labels.data.view_as(pred)).cpu().sum().item() / \
			  float(labels.size()[0])
		all_scores.append(prob.cpu().detach())  # for classification

		# Back propagate loss
		if optimizer is not None:
			optimizer.zero_grad()
			start_backward = time.perf_counter()
			loss.backward()
			temp_timing_dict["backward"].append(
				time.perf_counter() - start_backward)
			optimizer.step()

		loss = loss.data.cpu().detach().numpy()
		pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
		total_loss.append(np.array([loss, acc]) * len(selected_idx))

		n_samples += len(selected_idx)

	if optimizer is None:
		assert n_samples == len(sample_idxes)

	# Calculate average loss and report performance metrics
	total_loss = np.array(total_loss)
	avg_loss = np.sum(total_loss, 0) / n_samples
	roc_auc, prc_auc = auc_scores(all_targets, all_scores)
	avg_loss = np.concatenate((avg_loss, [roc_auc], [prc_auc]))

	# Append loop average to global timer tracking list.
	# Only for training phase
	if optimizer is not None:
		timing_dict["forward"].append(
			sum(temp_timing_dict["forward"]) /
			len(temp_timing_dict["forward"]))
		timing_dict["backward"].append(
			sum(temp_timing_dict["backward"]) /
			len(temp_timing_dict["backward"]))

	return avg_loss


'''
	Main program execution
'''
if __name__ == '__main__':
	# Get run arguments
	cmd_opt = argparse.ArgumentParser(
		description='Argparser for graph classification')
	cmd_opt.add_argument('-cuda', default='0', help='0-CPU, 1-GPU')
	cmd_opt.add_argument('-gm', default='GNN', help='GNN model to use')
	cmd_opt.add_argument('-data', default='ba2', help='Dataset to use')
	cmd_opt.add_argument('-retrain', default='0', help='Whether to re-train the classifier or use saved trained model')
	cmd_args, _ = cmd_opt.parse_known_args()

	# Get run configurations
	config = yaml.safe_load(open("config.yml"))
	config["run"]["model"] = cmd_args.gm
	config["run"]["dataset"] = cmd_args.data
	config["retrain"] = cmd_args.retrain

	# Set random seed
	random.seed(config["run"]["seed"])
	np.random.seed(config["run"]["seed"])
	torch.manual_seed(config["run"]["seed"])

	arg = {'lr': 0.001, "epochs": 1000, "clip_max": 2.0, "batch_size": 64, "early_stopping": 500, "seed": 42, "eval_enabled": True}

	# [0] 加载数据集和ground truth
	test_graph_list, dataset_features, test_data_list = data2nxgraph(config["run"]["dataset"])

	# [1] 获取新模型
	if config["retrain"] == '0':
		# 已有模型的情况下，直接读取模型
		model = model_selector(config["run"]["model"], config["run"]["dataset"], pretrained=True)
	else:  # 否则训练新模型
		model = train_model_graph(arg)

	# [2] 进行解释
	best_saliency_outputs_dict = {}
	saliency_map_generation_time_dict = {
		method: [] for method in config["interpretability_methods"].keys()}
	qualitative_metrics_dict_by_method = {
		method: {"fidelity": [], "contrastivity": [], "sparsity": [], "accuracy": [], "accuracy_max": [],
				 "fidelity-": []}
		for method in config["interpretability_methods"].keys()}
	print("Applying interpretability methods")
	kwargs = {"model": "GNNExplainer", "data": test_data_list}
	# For each enabled interpretability method
	for method in config["interpretability_methods"].keys():  # 对于其中一种解释方法
		if config["interpretability_methods"][method]["enabled"] is True:
			print("Running method: %s" % str(method))

			score_output, saliency_output, generate_score_execution_time = eval(method)(model, config,
																						dataset_features,
																						test_graph_list, -1,
																						cmd_args.cuda, **kwargs)
			saliency_map_generation_time_dict[method].append(generate_score_execution_time)
			best_saliency_outputs_dict.update(saliency_output)
			# Calculate qualitative metrics
			fidelity, contrastivity, sparsity, fidelityminus = compute_metric(
				model, score_output, dataset_features, config, cmd_args.cuda, **kwargs)
			if config["run"]["dataset"] == 'ba2':
				explanation_labels, indices = load_dataset_ground_truth(config["run"]["dataset"])
				accuracy = get_accuracy(config, explanation_labels, score_output)
				accuracy_max = get_accuracy_max(config, explanation_labels, score_output)
			else:
				accuracy = 0
				accuracy_max = 0

			qualitative_metrics_dict_by_method[method]["fidelity"].append(fidelity)
			qualitative_metrics_dict_by_method[method]["contrastivity"].append(contrastivity)
			qualitative_metrics_dict_by_method[method]["sparsity"].append(sparsity)
			qualitative_metrics_dict_by_method[method]["accuracy"].append(accuracy)
			qualitative_metrics_dict_by_method[method]["accuracy_max"].append(accuracy_max)
			qualitative_metrics_dict_by_method[method]["fidelity-"].append(fidelityminus)


	# Report qualitative metrics and configuration used
	run_statistics_string += ("== Interpretability methods settings and results ==\n")
	for method, qualitative_metrics_dict in \
			qualitative_metrics_dict_by_method.items():
		if config["interpretability_methods"][method]["enabled"] is True:
			# Report configuration settings used
			run_statistics_string += \
				"Qualitative metrics and settings for method %s:\n " % \
				method
			for option, value in config["interpretability_methods"][method].items():
				run_statistics_string += "%s: %s\n" % (str(option), str(value))

			# Report qualitative metrics
			if 'accuracy' in qualitative_metrics_dict:
				run_statistics_string += \
					"Accuracy (avg): %s " % \
					str(round(sum(qualitative_metrics_dict["accuracy"]) / len(qualitative_metrics_dict["accuracy"]), 5))
			if 'accuracy_max' in qualitative_metrics_dict:
				run_statistics_string += \
					"Accuracy_max (avg): %s " % \
					str(round(sum(qualitative_metrics_dict["accuracy_max"]) / len(qualitative_metrics_dict["accuracy_max"]), 5))
			run_statistics_string += \
				"Fidelity+ (avg): %s " % \
				str(round(sum(qualitative_metrics_dict["fidelity"]) / len(qualitative_metrics_dict["fidelity"]), 5))
			run_statistics_string += \
				"Fidelity- (avg): %s " % \
				str(round(sum(qualitative_metrics_dict["fidelity-"]) / len(qualitative_metrics_dict["fidelity-"]), 5))
			run_statistics_string += \
				"Contrastivity (avg): %s " % \
				str(round(
					sum(qualitative_metrics_dict["contrastivity"]) / len(qualitative_metrics_dict["contrastivity"]), 5))
			run_statistics_string += \
				"Sparsity (avg): %s\n" % \
				str(round(sum(qualitative_metrics_dict["sparsity"]) / len(qualitative_metrics_dict["sparsity"]), 5))
			run_statistics_string += \
				"Time taken to generate saliency scores: %s\n" % \
				str(round(sum(saliency_map_generation_time_dict[method]) /
						  len(saliency_map_generation_time_dict[method]) * 1000, 5))

			run_statistics_string += "\n"

	run_statistics_string += "\n\n"

	# [5] Create heatmap from the model with the best ROC_AUC output ==================================================
	custom_model_visualisation_options = None
	custom_dataset_visualisation_options = None

	# Sanity check
	if config["run"]["model"] in \
			config["custom_visualisation_options"]["GNN_models"].keys():
		custom_model_visualisation_options = \
			config["custom_visualisation_options"]["GNN_models"][config["run"]["model"]]

	if config["run"]["dataset"] in \
			config["custom_visualisation_options"]["dataset"].keys():
		custom_dataset_visualisation_options = \
			config["custom_visualisation_options"]["dataset"][config["run"]["dataset"]]

	# Generate saliency visualistion images
	output_count = output_to_images(best_saliency_outputs_dict,
									dataset_features,
									custom_model_visualisation_options,
									custom_dataset_visualisation_options,
									output_directory="results/image")
	print("Generated %s saliency map images." % output_count)

	# [6] Print and log run statistics ========================================
	if len(timing_dict["forward"]) > 0:
		run_statistics_string += \
			"Average forward propagation time taken(ms): %s\n" % \
			str(sum(timing_dict["forward"]) / len(timing_dict["forward"]) * 1000)
	if len(timing_dict["backward"]) > 0:
		run_statistics_string += \
			"Average backward propagation time taken(ms): %s\n" % \
			str(sum(timing_dict["backward"]) / len(timing_dict["backward"]) * 1000)

	print(run_statistics_string)

	# Save dataset features and run statistics to log
	current_datetime = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
	log_file_name = "%s_%s_datetime_%s.txt" % \
					(dataset_features["name"],
					 config["run"]["model"],
					 str(current_datetime))

	# Save log to text file
	with open("results/logs/%s" % log_file_name, "w") as f:
		if "dataset_info" in dataset_features.keys():
			dataset_info = dataset_features["dataset_info"] + "\n"
		else:
			dataset_info = ""
		f.write(dataset_info + run_statistics_string)
