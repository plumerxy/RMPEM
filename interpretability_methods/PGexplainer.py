from interpretability_methods.PGExplainer_RE.PGExplainer import PGExplainer
import numpy as np
import torch
import torch.nn.functional as F
from utilities.util import graph_to_tensor, standardize_scores
from time import perf_counter
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilities.util import graph_to_tensor, standardize_scores

def PGexplainer(classifier_model, config, dataset_features, GNNgraph_list, current_flold=None, cuda=0, **kwargs):
    # GNNgraph_list = GNNgraph_list[:10]
    # Initialise settings
    config = config
    interpretability_config = config["interpretability_methods"]["PGexplainer"]
    dataset_features = dataset_features
    task = 'graph'


    features_list = []
    edges_list = []
    if len(kwargs) > 0:
        data_list = kwargs["data"]
    for data in data_list:
        features_list.append(data.x)
        edges_list.append(data.edge_index)

    explainer = PGExplainer(classifier_model, edges_list, features_list, task, epochs=20, lr=0.005, reg_coefs=[0.03, 0.01], temp=[5.0, 1.0], sample_bias=0.0)
    explainer.prepare(range(len(features_list)), explain_class='max')

    explainer_non = PGExplainer(classifier_model, edges_list, features_list, task, epochs=20, lr=0.005, reg_coefs=[0.03, 0.01], temp=[5.0, 1.0], sample_bias=0.0)
    explainer_non.prepare(range(len(features_list)), explain_class='min')

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
            graph, expl = explainer.explain(i)
            expl = expl.detach().numpy()
            attribution = [0. for j in range(data.num_nodes)]
            # for j in range(len(expl)):
            #     attribution[graph[0][j]] += expl[j] / 2
            #     attribution[graph[1][j]] += expl[j] / 2
            for j in range(len(expl)):
                attribution[graph[0][j]] = max(attribution[graph[0][j]], expl[j])
                attribution[graph[1][j]] = max(attribution[graph[1][j]], expl[j])


            graph_non, expl_non = explainer_non.explain(i)
            expl_non = expl_non.detach().numpy()
            attribution_non = [0. for j in range(data.num_nodes)]
            # for j in range(len(expl_non)):
            #     attribution_non[graph_non[0][j]] += expl_non[j] / 2
            #     attribution_non[graph_non[1][j]] += expl_non[j] / 2
            # 另一种计算重要性的方法，取这个点连接的边中最大的重要性
            for j in range(len(expl_non)):
                attribution_non[graph_non[0][j]] = max(attribution_non[graph_non[0][j]], expl_non[j])
                attribution_non[graph_non[1][j]] = max(attribution_non[graph_non[1][j]], expl_non[j])

        tmp_timing_list.append(perf_counter() - start_generation)
        for _, label in dataset_features["label_dict"].items():
            mm = MinMaxScaler(feature_range=(0, 1))
            # stdsc = StandardScaler()
            if label == int(data.y.detach().numpy()):
                attribution_score = mm.fit_transform(np.array(attribution).reshape(-1, 1)).reshape(-1).tolist()
            else:
                attribution_score = mm.fit_transform(np.array(attribution_non).reshape(-1, 1)).reshape(-1).tolist()

            # if label == int(data.y.detach().numpy()):
            #     attribution_score = standardize_scores(attribution)
            # else:
            #     attribution_score = standardize_scores(attribution_non)

            #  ---------------- 只展示5个最重要的节点---------------------
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

        output_for_generating_saliency_map.update({"PGExplainer_%s_%s" % (str(assign_type), str(label)): []
                                                   for _, label in dataset_features["label_dict"].items()})

        for index in range(len(output_for_metrics_calculation)):
            tmp_output = output_for_metrics_calculation[index]
            tmp_label = tmp_output['graph'].label
            if tmp_output['graph'].graph_id in sample_graph_id_list:
                element_name = "rise_%s_%s" % (str(assign_type), str(tmp_label))
                output_for_generating_saliency_map[element_name].append(
                    (tmp_output['graph'], tmp_output[tmp_label]))

    elif interpretability_config["number_of_samples"] > 0:
        # Randomly sample from existing list:
        graph_idxes = list(range(len(output_for_metrics_calculation)))
        random.shuffle(graph_idxes)
        output_for_generating_saliency_map.update({"PGExplainer_class_%s" % str(label): []
                                                   for _, label in
                                                   dataset_features["label_dict"].items()})
        output_for_generating_comparing_saliency_map = {}
        output_for_generating_comparing_saliency_map.update({"PGExplainer_class_non%s" % str(label): []
                                                             for _, label in dataset_features[
                                                                 "label_dict"].items()})
        # Begin appending found samples
        for index in graph_idxes:
            tmp_label = output_for_metrics_calculation[index]['graph'].label
            if len(output_for_generating_saliency_map["PGExplainer_class_%s" % str(tmp_label)]) < \
                    interpretability_config["number_of_samples"]:
                output_for_generating_saliency_map["PGExplainer_class_%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))
                output_for_generating_comparing_saliency_map["PGExplainer_class_non%s" % str(tmp_label)].append(
                    (output_for_metrics_calculation[index]['graph'],
                     output_for_metrics_calculation[index][int(not tmp_label)]))
        output_for_generating_saliency_map.update(output_for_generating_comparing_saliency_map)
    return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time