import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm

from utilities.util import graph_to_tensor, standardize_scores
from time import perf_counter
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models.GNN_paper import GraphGCNPerturb


class CFexplainer:
    def __init__(self, model, edges, feats, n_hid, dropout, label, y_pred_orig, num_classes, beta):
        self.model = model
        self.model.eval()
        self.edges = edges
        self.feats = feats
        self.n_hid = n_hid
        self.dropout = dropout
        self.label = label
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes

        # Instantiate CF model class, load weights from original model
        self.cf_model = GraphGCNPerturb(feats.shape[1], num_classes, edges, beta)
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

    def explain(self, cf_optimizer, lr, n_momentum, num_epochs, arg):
        self.x = self.feats
        self.A_x = self.edges

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch, arg)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
        print("{} CF examples".format(num_cf_examples))
        print(" ")
        return best_cf_example[-1]

    def train(self, epoch, arg='max'):
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)
        # output_actual, self.P = self.cf_model.forward_prediction(self.x, self.A_x)

        if arg == 'max':
            y_pred_new = torch.argmax(output)
        else:
            y_pred_new = torch.argmin(output)
        # y_pred_new_actual = torch.argmax(output_actual)

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output, self.y_pred_orig,
                                                                            y_pred_new)
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.item()),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(output.data),
              'Output nondiff: {}\n'.format(output.data),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new,
                                                                         y_pred_new))
        print(" ")
        cf_stats = []
        # if y_pred_new != self.y_pred_orig:
        cf_stats = [cf_adj.detach().numpy()]

        return cf_stats, loss_total.item()


def CFExplainer(classifier_model, config, dataset_features, GNNgraph_list, current_flold=None, cuda=0, **kwargs):
    """
    	:param classifier_model: trained classifier model  重点是这个model 从中获取梯度
    	:param config: parsed configuration file of config.yml
    	:param dataset_features: a dictionary of dataset features obtained from load_data.py
    	:param GNNgraph_list: a list of GNNgraphs obtained from the dataset  图数据 因为是图分类问题 所以有很多张图
    	:param current_fold: has no use in this method
    	:param cuda: whether to use GPU to perform conversion to tensor
    """
    # GNNgraph_list = GNNgraph_list[:10]
    # Initialise settings
    config = config
    interpretability_config = config["interpretability_methods"]["CFExplainer"]
    dataset_features = dataset_features

    output_for_metrics_calculation = []
    output_for_generating_saliency_map = {}

    # Obtain attribution score for use in qualitative metrics
    tmp_timing_list = []

    i = 0
    for GNNgraph in GNNgraph_list:
        print("explian for graph %d" % GNNgraph.graph_id)
        output = {'graph': GNNgraph}
        if len(kwargs) > 0:
            data = kwargs["data"][i]
            start_generation = perf_counter()
            node_feat = data.x
            edge_index = data.edge_index

            y_pred_orig = torch.argmax(classifier_model(node_feat, edge_index))
            explainer = CFexplainer(classifier_model, edge_index, node_feat, 20, 0.0, data.y, y_pred_orig=y_pred_orig,
                                    num_classes=dataset_features["num_class"], beta=0.5)
            cf_example = explainer.explain(cf_optimizer="SGD", lr=0.5, n_momentum=0, num_epochs=1000, arg='max')
            if len(cf_example) == 0:
                continue
            cf_example = cf_example[0]
            cf_example = 1-cf_example
            attribution = [0. for j in range(data.num_nodes)]
            # for j in range(len(cf_example)):  # 把每条边的重要性分到两个端点上
            #     attribution[edge_index[0][j]] += cf_example[j] / 2
            #     attribution[edge_index[1][j]] += cf_example[j] / 2
            # 另一种计算重要性的方法，取这个点连接的边中最大的重要性
            for j in range(len(cf_example)):
                attribution[edge_index[0][j]] = max(attribution[edge_index[0][j]], cf_example[j])
                attribution[edge_index[1][j]] = max(attribution[edge_index[1][j]], cf_example[j])
            # 另一个标签的预测
            y_pred_orig_non = 1 - torch.argmax(classifier_model(node_feat, edge_index))
            explainer = CFexplainer(classifier_model, edge_index, node_feat, 20, 0.0, data.y, y_pred_orig=y_pred_orig_non,
                                    num_classes=dataset_features["num_class"], beta=0.5)
            cf_example_non = explainer.explain(cf_optimizer="SGD", lr=0.5, n_momentum=0, num_epochs=1000, arg='min')
            if len(cf_example_non) == 0:
                cf_example_non = cf_example
            else:
                cf_example_non = cf_example_non[0]
                cf_example_non = 1 - cf_example_non
            attribution_non = [0. for j in range(data.num_nodes)]
            # for j in range(len(cf_example_non)):  # 把每条边的重要性分到两个端点上
            #     attribution_non[edge_index[0][j]] += cf_example_non[j] / 2
            #     attribution_non[edge_index[1][j]] += cf_example_non[j] / 2
            for j in range(len(cf_example_non)):
                attribution_non[edge_index[0][j]] = max(attribution_non[edge_index[0][j]], cf_example_non[j])
                attribution_non[edge_index[1][j]] = max(attribution_non[edge_index[1][j]], cf_example_non[j])

        tmp_timing_list.append(perf_counter() - start_generation)
        for _, label in dataset_features["label_dict"].items():
            # attribution_score = attribution[label].tolist()
            # attribution_score = standardize_scores(attribution_score)
            mm = MinMaxScaler(feature_range=(0, 1))
            # mm = StandardScaler()
            if label == int(data.y.detach().numpy()):
                attribution_score = mm.fit_transform(np.array(attribution).reshape(-1, 1)).reshape(-1).tolist()
            else:
                attribution_score = mm.fit_transform(np.array(attribution_non).reshape(-1, 1)).reshape(-1).tolist()
            #  ---------------- 只展示5个最重要的节点---------------------
            # index = np.argsort(np.array(attribution_score))[:-5]
            # attribution_score_nd = np.array(attribution_score)
            # attribution_score_nd[index] = 0
            # attribution_score = attribution_score_nd.tolist()

            output[label] = attribution_score
        output_for_metrics_calculation.append(output)
        i += 1
    execution_time = 1

    # Obtain attribution score for use in generating saliency map for comparison with zero tensors
    if interpretability_config["sample_ids"] is not None:
        if ',' in str(interpretability_config["sample_ids"]):
            sample_graph_id_list = list(map(int, interpretability_config["sample_ids"].split(',')))
        else:
            sample_graph_id_list = [int(interpretability_config["sample_ids"])]

        output_for_generating_saliency_map.update({"cf_%s_%s" % (str(assign_type), str(label)): []
                                                   for _, label in dataset_features["label_dict"].items()})

        for index in range(len(output_for_metrics_calculation)):
            tmp_output = output_for_metrics_calculation[index]
            tmp_label = tmp_output['graph'].label
            if tmp_output['graph'].graph_id in sample_graph_id_list:
                element_name = "cf_%s_%s" % (str(assign_type), str(tmp_label))
                output_for_generating_saliency_map[element_name].append(
                    (tmp_output['graph'], tmp_output[tmp_label]))

    elif interpretability_config["number_of_samples"] > 0:
        # Randomly sample from existing list:
        graph_idxes = list(range(len(output_for_metrics_calculation)))
        random.shuffle(graph_idxes)
        output_for_generating_saliency_map.update({"cf_class_%s" % str(label): []
                                                   for _, label in
                                                   dataset_features["label_dict"].items()})
        output_for_generating_comparing_saliency_map = {}
        output_for_generating_comparing_saliency_map.update({"cf_class_non%s" % str(label): []
                                                             for _, label in dataset_features[
                                                                 "label_dict"].items()})
        # Begin appending found samples
        for index in graph_idxes:
            tmp_label = output_for_metrics_calculation[index]['graph'].label
            if len(output_for_generating_saliency_map["cf_class_%s" % str(tmp_label)]) < \
                    interpretability_config["number_of_samples"]: 
                output_for_generating_saliency_map["cf_class_%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))
                output_for_generating_comparing_saliency_map["cf_class_non%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'],
                     output_for_metrics_calculation[index][int(not tmp_label)]))
        output_for_generating_saliency_map.update(output_for_generating_comparing_saliency_map)
    return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time


if __name__ == "__main__":
    r = rise(3, 0.8, 5)
    print("not end")
