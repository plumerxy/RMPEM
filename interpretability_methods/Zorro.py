import numpy as np
import torch
import torch.nn.functional as F

from interpretability_methods.ZorroOri.explainer import Zorro
from utilities.util import graph_to_tensor, standardize_scores
from time import perf_counter
import random
from sklearn.preprocessing import MinMaxScaler


def ZORRO(classifier_model, config, dataset_features, GNNgraph_list, current_flold=None, cuda=0, **kwargs):
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
    interpretability_config = config["interpretability_methods"]["ZORRO"]
    dataset_features = dataset_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Perform Zorro on the classifier model
    zorro = Zorro(classifier_model, device, greedy=True,
                  record_process_time=True, samples=100)

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
            attribution_max = zorro.explain_node(1, node_feat, edge_index, tau=0.1, recursion_depth=1,
                                             save_initial_improve=True, type="max")
            attribution_max = attribution_max[0][0][0]
            attribution_min = zorro.explain_node(1, node_feat, edge_index, tau=0.1, recursion_depth=1,
                                               save_initial_improve=True, type="min")
            attribution_min = attribution_min[0][0][0]
            with torch.no_grad():
                log_logits = classifier_model(node_feat, edge_index)
                predicted_label = log_logits.argmax(dim=-1).squeeze()
            if predicted_label == 0:
                attribution = np.vstack((attribution_max, attribution_min))
            else:
                attribution = np.vstack((attribution_min, attribution_max))

        tmp_timing_list.append(perf_counter() - start_generation)  # 测量解释时间
        for _, label in dataset_features["label_dict"].items():
            # attribution_score = attribution[label].tolist()  # 各节点的重要性得分
            # attribution_score = standardize_scores(attribution_score)
            mm = MinMaxScaler(feature_range=(0, 1))
            attribution_score = mm.fit_transform(attribution[label].reshape(-1, 1)).reshape(-1).tolist()
            # #  ---------------- 只展示5个最重要的节点---------------------
            # index = np.argsort(np.array(attribution_score))[:-5]
            # attribution_score_nd = np.array(attribution_score)
            # attribution_score_nd[index] = 0
            # attribution_score = attribution_score_nd.tolist()

            output[label] = attribution_score
        output_for_metrics_calculation.append(output)
        i += 1
    execution_time = sum(tmp_timing_list) / (len(tmp_timing_list))

    # Obtain attribution score for use in generating saliency map for comparison with zero tensors
    if interpretability_config["sample_ids"] is not None:
        if ',' in str(interpretability_config["sample_ids"]):
            sample_graph_id_list = list(map(int, interpretability_config["sample_ids"].split(',')))
        else:
            sample_graph_id_list = [int(interpretability_config["sample_ids"])]

        output_for_generating_saliency_map.update({"zorro_%s_%s" % (str(assign_type), str(label)): []
                                                   for _, label in dataset_features["label_dict"].items()})

        for index in range(len(output_for_metrics_calculation)):
            tmp_output = output_for_metrics_calculation[index]
            tmp_label = tmp_output['graph'].label
            if tmp_output['graph'].graph_id in sample_graph_id_list:
                element_name = "zorro_%s_%s" % (str(assign_type), str(tmp_label))
                output_for_generating_saliency_map[element_name].append(
                    (tmp_output['graph'], tmp_output[tmp_label]))

    elif interpretability_config["number_of_samples"] > 0:
        # Randomly sample from existing list:
        graph_idxes = list(range(len(output_for_metrics_calculation)))
        random.shuffle(graph_idxes)
        output_for_generating_saliency_map.update({"zorro_class_%s" % str(label): []
                                                   for _, label in
                                                   dataset_features["label_dict"].items()})
        output_for_generating_comparing_saliency_map = {}
        output_for_generating_comparing_saliency_map.update({"zorro_class_non%s" % str(label): []
                                                             for _, label in dataset_features[
                                                                 "label_dict"].items()})
        # Begin appending found samples
        for index in graph_idxes:
            tmp_label = output_for_metrics_calculation[index]['graph'].label
            if len(output_for_generating_saliency_map["zorro_class_%s" % str(tmp_label)]) < \
                    interpretability_config["number_of_samples"]:
                output_for_generating_saliency_map["zorro_class_%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))
                output_for_generating_comparing_saliency_map["zorro_class_non%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'],
                     output_for_metrics_calculation[index][int(not tmp_label)]))
        output_for_generating_saliency_map.update(output_for_generating_comparing_saliency_map)
    return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time
