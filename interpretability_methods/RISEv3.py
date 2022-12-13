import numpy as np
import torch
import torch.nn.functional as F
from utilities.util import graph_to_tensor, standardize_scores
from time import perf_counter
import random
from sklearn.preprocessing import MinMaxScaler


class rise():
    def __init__(self, classifier_model):
        """
        classifier_model: 待解释的GNN模型
        """
        self.classifier_model = classifier_model

    def generate_masks(self, N, p, num_of_nodes):
        """
        生成掩码
        p: 掩码为1的概率
        N: 生成掩码的数量
        num_of_nodes: 图中节点的数量
        """
        node_masks = (np.random.rand(N, num_of_nodes) < p).astype(float)  # 概率1-p置为0
        return node_masks

    def attribute(self, node_feat, n2n, subg, graph, N=2000, p=0.5):
        """
        node_feat: 特征矩阵
        n2n: 邻接矩阵
        graph: GNNgraph
        target: 待解释的label
        p: 掩码为1的概率
        N: 生成掩码的数量
        """
        node_masks = self.generate_masks(N, p, len(node_feat))
        feat_masked = node_feat.unsqueeze(0).repeat(N, 1, 1)
        feat_masked = torch.mul(torch.tensor(node_masks).unsqueeze(2), feat_masked)
        weights = []
        for feat_m in feat_masked:
            output = self.classifier_model(feat_m.to(torch.float32), n2n, subg, graph)
            prob = F.softmax(output, dim=1)
            weights.append(prob[0])
        weights = torch.stack(weights).detach().numpy()
        attribution = weights.T.dot(node_masks) / N / p
        return attribution


def RISEv3(classifier_model, config, dataset_features, GNNgraph_list, current_flold=None, cuda=0):
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
    interpretability_config = config["interpretability_methods"]["RISE"]
    dataset_features = dataset_features

    # Perform RISE on the classifier model
    ri = rise(classifier_model)

    output_for_metrics_calculation = []
    output_for_generating_saliency_map = {}

    # Obtain attribution score for use in qualitative metrics
    tmp_timing_list = []

    for GNNgraph in GNNgraph_list:  # 对于test set的每一张图
        print("explian for graph %d" %GNNgraph.graph_id)
        output = {'graph': GNNgraph}
        node_feat, n2n, subg = graph_to_tensor(
            [GNNgraph], dataset_features["feat_dim"],  # 这里就是把图要放进代码的tensor都准备好
            dataset_features["edge_feat_dim"], cuda)
        start_generation = perf_counter()  # 用于测量时间
        attribution = ri.attribute(node_feat, n2n, subg, [GNNgraph])

        tmp_timing_list.append(perf_counter() - start_generation)  # 测量解释时间
        for _, label in dataset_features["label_dict"].items():
            # attribution_score = attribution[label].tolist()  # 各节点的重要性得分
            # attribution_score = standardize_scores(attribution_score)
            mm = MinMaxScaler(feature_range=(0, 1))
            attribution_score = mm.fit_transform(attribution[label].reshape(-1, 1)).reshape(-1).tolist()
            output[label] = attribution_score
        output_for_metrics_calculation.append(output)

    execution_time = sum(tmp_timing_list) / (len(tmp_timing_list))

    # Obtain attribution score for use in generating saliency map for comparison with zero tensors
    if interpretability_config["sample_ids"] is not None:  # 已经指定了一个样本的id的话 为它balabala
        if ',' in str(interpretability_config["sample_ids"]):
            sample_graph_id_list = list(map(int, interpretability_config["sample_ids"].split(',')))
        else:
            sample_graph_id_list = [int(interpretability_config["sample_ids"])]

        output_for_generating_saliency_map.update({"rise_%s_%s" % (str(assign_type), str(label)): []
                                                   for _, label in dataset_features["label_dict"].items()})

        for index in range(len(output_for_metrics_calculation)):
            tmp_output = output_for_metrics_calculation[index]
            tmp_label = tmp_output['graph'].label
            if tmp_output['graph'].graph_id in sample_graph_id_list:
                element_name = "rise_%s_%s" % (str(assign_type), str(tmp_label))
                output_for_generating_saliency_map[element_name].append(
                    (tmp_output['graph'], tmp_output[tmp_label]))

    elif interpretability_config["number_of_samples"] > 0:  # 如果指定了要随机采样几个样本的个数
        # Randomly sample from existing list:
        graph_idxes = list(range(len(output_for_metrics_calculation)))
        random.shuffle(graph_idxes)
        output_for_generating_saliency_map.update({"rise_class_%s" % str(label): []
                                                   for _, label in
                                                   dataset_features["label_dict"].items()})  # 对每个标签建立一个字典项
        output_for_generating_comparing_saliency_map = {}
        output_for_generating_comparing_saliency_map.update({"rise_class_non%s" % str(label): []
                                                             for _, label in dataset_features[
                                                                 "label_dict"].items()})  # 为了更好的分析解释结果，除了原标签外，再把另一个类的结果也输出可视化看一下。
        # Begin appending found samples
        for index in graph_idxes:
            tmp_label = output_for_metrics_calculation[index]['graph'].label
            if len(output_for_generating_saliency_map["rise_class_%s" % str(tmp_label)]) < \
                    interpretability_config["number_of_samples"]:  # 对每个标签都采样3个样本, 这里只保存它真实标签的解释结果
                output_for_generating_saliency_map["rise_class_%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))
                output_for_generating_comparing_saliency_map["rise_class_non%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'],
                     output_for_metrics_calculation[index][int(not tmp_label)]))
        output_for_generating_saliency_map.update(output_for_generating_comparing_saliency_map)
    return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time

if __name__ == "__main__":
    r = rise(3, 0.8, 5)
    print("not end")
