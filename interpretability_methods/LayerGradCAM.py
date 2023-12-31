import numpy as np
import torch
import json
import random

from time import perf_counter
from os import path
from copy import deepcopy
from captum.attr import LayerGradCam
from utilities.util import graph_to_tensor, standardize_scores

def LayerGradCAM(classifier_model, config, dataset_features, GNNgraph_list, current_fold=None, cuda=0, **kwargs):
	'''
		Attribute to input layer using soft assign
		:param classifier_model: trained classifier model GNN
		:param config: parsed configuration file of config.yml 配置参数
		:param dataset_features: a dictionary of dataset features obtained from load_data.py 数据集的参数
		:param GNNgraph_list: a list of GNNgraphs obtained from the dataset 图数据集 输入的是测试集
		:param current_fold: has no use in this method
		:param cuda: whether to use GPU to perform conversion to tensor
	'''
	# Initialise settings
	config = config
	interpretability_config = config["interpretability_methods"]["LayerGradCAM"]
	dataset_features = dataset_features
	assign_type = interpretability_config["assign_attribution"]

	# Perform grad cam on the classifier model and on a specific layer
	layer_idx = interpretability_config["layer"]
	if len(kwargs) > 0:
		gc = LayerGradCam(classifier_model, classifier_model.conv1)  # ptc-fr conv1, else conv3
	else:
		if layer_idx == 0:
			gc = LayerGradCam(classifier_model, classifier_model.graph_convolution)
		else:
			gc = LayerGradCam(classifier_model, classifier_model.conv_modules[layer_idx-1])

	output_for_metrics_calculation = []
	output_for_generating_saliency_map = {}

	# Obtain attribution score for use in qualitative metrics
	tmp_timing_list = []

	i = 0
	for GNNgraph in GNNgraph_list:
		output = {'graph': GNNgraph}
		for _, label in dataset_features["label_dict"].items():
			# Relabel all just in case, may only relabel those that need relabelling
			# if performance is poor
			original_label = GNNgraph.label
			GNNgraph.label = label

			if len(kwargs) > 0:
				data = kwargs["data"][i]
				start_generation = perf_counter()
				node_feat = data.x
				edge_index = data.edge_index
				attribution = gc.attribute(node_feat,
										   additional_forward_args=edge_index,
										   target=label, relu_attributions=True)

			# Attribute to the input layer using the assign method specified
			reverse_assign_tensor_list = []
			for i in range(1, layer_idx + 1):
				assign_tensor = classifier_model.cur_assign_tensor_list[i - 1]
				max_index = torch.argmax(assign_tensor, dim=1, keepdim=True)
				if assign_type == "hard":
					reverse_assign_tensor = torch.transpose(
						torch.zeros(assign_tensor.size()).scatter_(1, max_index, value=1), 0, 1)
				else:
					reverse_assign_tensor = torch.transpose(assign_tensor, 0, 1)

				reverse_assign_tensor_list.append(reverse_assign_tensor)

			attribution = torch.transpose(attribution, 0, 1)

			for reverse_tensor in reversed(reverse_assign_tensor_list):
				attribution = attribution @ reverse_tensor

			attribution = torch.transpose(attribution, 0, 1)
			tmp_timing_list.append(perf_counter() - start_generation)

			attribution_score = torch.sum(attribution, dim=1).tolist()
			attribution_score = standardize_scores(attribution_score)

			GNNgraph.label = original_label

			#  ---------------- 只展示5个最重要的节点---------------------
			# index = np.argsort(np.array(attribution_score))[:-5]
			# attribution_score_nd = np.array(attribution_score)
			# attribution_score_nd[index] = 0
			# attribution_score = attribution_score_nd.tolist()

			output[label] = attribution_score
		output_for_metrics_calculation.append(output)
		i += 1
	execution_time = sum(tmp_timing_list)/(len(tmp_timing_list))

	# Obtain attribution score for use in generating saliency map for comparison with zero tensors
	if interpretability_config["sample_ids"] is not None:
		if ',' in str(interpretability_config["sample_ids"]):
			sample_graph_id_list = list(map(int, interpretability_config["sample_ids"].split(',')))
		else:
			sample_graph_id_list = [int(interpretability_config["sample_ids"])]

		output_for_generating_saliency_map.update({"layergradcam_%s_%s" % (str(assign_type), str(label)): []
												   for _, label in dataset_features["label_dict"].items()})

		for index in range(len(output_for_metrics_calculation)):
			tmp_output = output_for_metrics_calculation[index]
			tmp_label = tmp_output['graph'].label
			if tmp_output['graph'].graph_id in sample_graph_id_list:
				element_name = "layergradcam_%s_%s" % (str(assign_type), str(tmp_label))
				output_for_generating_saliency_map[element_name].append(
					(tmp_output['graph'], tmp_output[tmp_label]))

	elif interpretability_config["number_of_samples"] > 0:
		# Randomly sample from existing list:
		graph_idxes = list(range(len(output_for_metrics_calculation)))
		random.shuffle(graph_idxes)
		output_for_generating_saliency_map.update({"layergradcam_%s_%s" % (str(assign_type), str(label)): []
												   for _, label in dataset_features["label_dict"].items()})
		output_for_generating_comparing_saliency_map = {}
		output_for_generating_comparing_saliency_map.update({"layergradcam_class_non%s" % str(label): []
															 for _, label in dataset_features[
																 "label_dict"].items()})

		# Begin appending found samples
		for index in graph_idxes:
			tmp_label = output_for_metrics_calculation[index]['graph'].label
			element_name = "layergradcam_%s_%s" % (str(assign_type), str(tmp_label))
			if len(output_for_generating_saliency_map[element_name]) < interpretability_config["number_of_samples"]:
				output_for_generating_saliency_map[element_name].append(
					(output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))
				output_for_generating_comparing_saliency_map["layergradcam_class_non%s" % str(tmp_label)].append(
					(output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][int(not tmp_label)]))
		output_for_generating_saliency_map.update(output_for_generating_comparing_saliency_map)
	return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time
