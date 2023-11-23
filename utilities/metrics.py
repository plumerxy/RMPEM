import numpy
import torch

import torch.nn.functional as F
from copy import deepcopy
from sklearn import metrics
import numpy as np

from utilities.util import graph_to_tensor, hamming


def auc_scores(all_targets, all_scores):
    all_scores = torch.cat(all_scores).cpu().numpy()
    number_of_classes = int(all_scores.shape[1])

    # For binary classification:
    roc_auc = 0.0
    prc_auc = 0.0
    if number_of_classes == 2:
        # Take only second column (i.e. scores for positive label)
        all_scores = all_scores[:, 1]
        roc_auc = metrics.roc_auc_score(
            all_targets, all_scores, average='macro')
        prc_auc = metrics.average_precision_score(
            all_targets, all_scores, average='macro', pos_label=1)
    # For multi-class classification:
    if number_of_classes > 2:
        # Hand & Till (2001) implementation (ovo)
        roc_auc = metrics.roc_auc_score(
            all_targets, all_scores, multi_class='ovo', average='macro')

    # TODO: build PRC-AUC calculations for multi-class datasets

    return roc_auc, prc_auc


# Fidelity ====================================================================
def get_accuracy(trained_classifier_model, GNNgraph_list, dataset_features, cuda):
    trained_classifier_model.eval()
    true_equal_pred_pairs = []

    # Instead of sending the whole list as batch,
    # do it one by one in case classifier do not support batch-processing
    # TODO: Enable batch processing support
    for GNNgraph in GNNgraph_list:  # 对测试集的每一张图
        node_feat, n2n, subg = graph_to_tensor(
            [GNNgraph], dataset_features["feat_dim"],
            dataset_features["edge_feat_dim"], cuda)

        output = trained_classifier_model(node_feat, n2n, subg, [GNNgraph])
        logits = F.log_softmax(output, dim=1)
        pred = logits.data.max(1, keepdim=True)[1]

        if GNNgraph.label == int(pred[0]):
            true_equal_pred_pairs.append(1)
        else:
            true_equal_pred_pairs.append(0)

    return sum(true_equal_pred_pairs) / len(true_equal_pred_pairs)


def get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda, **kwargs):
    trained_classifier_model.eval()
    score_list = []
    target_list = []

    if dataset_features["num_class"] > 2:
        print("Unable to calculate fidelity for multiclass datset")
        return 0

    # Instead of sending the whole list as batch,
    # do it one by one in case classifier do not support batch-processing
    # TODO: Enable batch processing support
    i = 0
    for GNNgraph in GNNgraph_list:
        if len(kwargs) > 0:  # 使用torch geo数据的模型
            if "occluded" in kwargs.keys() and kwargs["occluded"]:
                data = kwargs["newdata"][i]
            else:
                data = kwargs["data"][i]  # 获取第i个图 Data对象的形式
            node_feat = data.x
            edge_index = data.edge_index
            output = trained_classifier_model(node_feat, edge_index)
        else:
            node_feat, n2n, subg = graph_to_tensor(
                [GNNgraph], dataset_features["feat_dim"],
                dataset_features["edge_feat_dim"], cuda)
            output = trained_classifier_model(node_feat, n2n, subg, [GNNgraph])

        logits = F.log_softmax(output, dim=1)
        prob = F.softmax(logits, dim=1)

        score_list.append(prob.cpu().detach())
        target_list.append(GNNgraph.label)

        i += 1

    score_list = torch.cat(score_list).cpu().numpy()
    score_list = score_list[:, 1]

    roc_auc = metrics.roc_auc_score(
        target_list, score_list, average='macro')

    return roc_auc


def is_salient(score, importance_range):
    start, end = importance_range
    if start <= score <= end:
        return True
    else:
        return False


def occlude_graphs_new(metric_attribution_scores, dataset_features, importance_prop, **kwargs):
    # Transform the graphs, occlude nodes with significant attribution scores
    occluded_GNNgraph_list = []
    occluded_GNNdata_list = []
    index = 0
    for group in metric_attribution_scores:
        GNNgraph = deepcopy(group['graph'])
        attribution_score = group[GNNgraph.label]

        if len(kwargs) > 0:
            GNNdata = deepcopy(kwargs['data'][index])
            import_nodes_index = np.argsort(np.array(attribution_score))[::-1][
                                 :int(len(attribution_score) * importance_prop)]  # 按照比例认为是重要的节点们
            for i in range(len(import_nodes_index)):
                if attribution_score[import_nodes_index[i]] != 0:  # 如果这个节点的重要性为0，不算
                    GNNdata.x[import_nodes_index[i]] = 0
            occluded_GNNdata_list.append(GNNdata)

        else:
            # Go through every node in graph to check if node is salient
            import_nodes_index = np.argsort(np.array(attribution_score))[::-1][
                                 :int(len(attribution_score) * importance_prop)]  # 按照比例认为是重要的节点们
            for i in range(len(import_nodes_index)):
                if attribution_score[import_nodes_index[i]] != 0:  # 如果这个节点的重要性为0，不算
                    # Occlude node by assigning it an "UNKNOWN" label
                    if dataset_features['have_node_labels'] is True:
                        GNNgraph.node_labels[import_nodes_index[i]] = None
                    if dataset_features['have_node_attributions'] is True:
                        GNNgraph.node_features[import_nodes_index[i]].fill(0)
            occluded_GNNgraph_list.append(GNNgraph)
        index += 1

    if len(kwargs) > 0:
        return occluded_GNNdata_list
    else:
        return occluded_GNNgraph_list


def occlude_graphs_new_n(metric_attribution_scores, dataset_features, importance_prop, **kwargs):
    """
    遮不重要的，留下最重要的
    """
    # Transform the graphs, occlude nodes with significant attribution scores
    occluded_GNNgraph_list = []
    occluded_GNNdata_list = []
    index = 0
    for group in metric_attribution_scores:
        GNNgraph = deepcopy(group['graph'])
        attribution_score = group[GNNgraph.label]

        if len(kwargs) > 0:
            GNNdata = deepcopy(kwargs['data'][index])
            import_nodes_index = np.argsort(np.array(attribution_score))[:-int(len(attribution_score) * importance_prop)]  # 按照比例认为是重要的节点们
            for i in range(len(import_nodes_index)):
                if attribution_score[import_nodes_index[i]] != 0:  # 如果这个节点的重要性为0，不算
                    GNNdata.x[import_nodes_index[i]] = 0
            occluded_GNNdata_list.append(GNNdata)

        else:
            # Go through every node in graph to check if node is salient
            import_nodes_index = np.argsort(np.array(attribution_score))[::-1][
                                 :int(len(attribution_score) * importance_prop)]  # 按照比例认为是重要的节点们
            for i in range(len(import_nodes_index)):
                if attribution_score[import_nodes_index[i]] != 0:  # 如果这个节点的重要性为0，不算
                    # Occlude node by assigning it an "UNKNOWN" label
                    if dataset_features['have_node_labels'] is True:
                        GNNgraph.node_labels[import_nodes_index[i]] = None
                    if dataset_features['have_node_attributions'] is True:
                        GNNgraph.node_features[import_nodes_index[i]].fill(0)
            occluded_GNNgraph_list.append(GNNgraph)
        index += 1

    if len(kwargs) > 0:
        return occluded_GNNdata_list
    else:
        return occluded_GNNgraph_list


def occlude_graphs(metric_attribution_scores, dataset_features, importance_range, **kwargs):
    # Transform the graphs, occlude nodes with significant attribution scores
    occluded_GNNgraph_list = []
    occluded_GNNdata_list = []
    index = 0
    for group in metric_attribution_scores:
        GNNgraph = deepcopy(group['graph'])
        attribution_score = group[GNNgraph.label]

        if len(kwargs) > 0:
            GNNdata = deepcopy(kwargs['data'][index])
            for i in range(len(attribution_score)):
                # Only occlude nodes that provide significant positive contribution
                if is_salient((attribution_score[i]), importance_range):
                    GNNdata.x[i] = 0
            occluded_GNNdata_list.append(GNNdata)

        else:
            # Go through every node in graph to check if node is salient
            for i in range(len(attribution_score)):
                # Only occlude nodes that provide significant positive contribution
                if is_salient((attribution_score[i]), importance_range):
                    # Occlude node by assigning it an "UNKNOWN" label
                    if dataset_features['have_node_labels'] is True:
                        GNNgraph.node_labels[i] = None
                    if dataset_features['have_node_attributions'] is True:
                        GNNgraph.node_features[i].fill(0)
            occluded_GNNgraph_list.append(GNNgraph)
        index += 1

    if len(kwargs) > 0:
        return occluded_GNNdata_list
    else:
        return occluded_GNNgraph_list


def get_fidelity(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda, **kwargs):
    importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
    importance_range = [float(bound) for bound in importance_range]

    GNNgraph_list = [group["graph"] for group in metric_attribution_scores]

    roc_auc_prior_occlusion = get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda, **kwargs)
    occluded_GNNgraph_list = occlude_graphs(metric_attribution_scores, dataset_features, importance_range, **kwargs)
    if len(kwargs) > 0:
        kwargs['occluded'] = True
        kwargs['newdata'] = occluded_GNNgraph_list
        occluded_GNNgraph_list = [group["graph"] for group in metric_attribution_scores]
    roc_auc_after_occlusion = get_roc_auc(trained_classifier_model, occluded_GNNgraph_list, dataset_features, cuda,
                                          **kwargs)

    fidelity_score = roc_auc_prior_occlusion - roc_auc_after_occlusion
    return fidelity_score


# fidelity+====================================================================
def get_fidelity_new(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda, **kwargs):
    """
    更新fidelity，不根据重要性范围进行计算，而是选取前百分之p最重要的节点
    """
    importance_prop = config["metrics"]["fidelity"]["importance_prop"]

    GNNgraph_list = [group["graph"] for group in metric_attribution_scores]

    roc_auc_prior_occlusion = get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda,
                                          **kwargs)  # 这个地方，把图输进去算auc，与扰动方法无关，不需要更改
    occluded_GNNgraph_list = occlude_graphs_new(metric_attribution_scores, dataset_features, importance_prop, **kwargs)
    if len(kwargs) > 0:
        kwargs['occluded'] = True
        kwargs['newdata'] = occluded_GNNgraph_list
        occluded_GNNgraph_list = [group["graph"] for group in metric_attribution_scores]
    roc_auc_after_occlusion = get_roc_auc(trained_classifier_model, occluded_GNNgraph_list, dataset_features, cuda,
                                          **kwargs)
    fidelity_score = roc_auc_prior_occlusion - roc_auc_after_occlusion
    return fidelity_score


# fidelity-====================================================================
def get_fidelityminus_new(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda, **kwargs):
    """
    更新fidelity，不根据重要性范围进行计算，而是选取前百分之p最重要的节点
    fidelity-是只保留最重要的
    """
    importance_prop = config["metrics"]["fidelity"]["importance_prop"]

    GNNgraph_list = [group["graph"] for group in metric_attribution_scores]

    roc_auc_prior_occlusion = get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda,
                                          **kwargs)  # 这个地方，把图输进去算auc，与扰动方法无关，不需要更改
    occluded_GNNgraph_list = occlude_graphs_new_n(metric_attribution_scores, dataset_features, importance_prop, **kwargs)
    if len(kwargs) > 0:
        kwargs['occluded'] = True
        kwargs['newdata'] = occluded_GNNgraph_list
        occluded_GNNgraph_list = [group["graph"] for group in metric_attribution_scores]
    roc_auc_after_occlusion = get_roc_auc(trained_classifier_model, occluded_GNNgraph_list, dataset_features, cuda,
                                          **kwargs)
    fidelity_score = roc_auc_prior_occlusion - roc_auc_after_occlusion
    return fidelity_score


# Contrastivity ====================================================================
def binarize_score_list(attribution_scores_list, importance_range):
    binarized_scores_list = []
    for scores in attribution_scores_list:
        binary_score = ''
        for score in scores:
            if is_salient(abs(float(score)), importance_range):
                binary_score += '1'
            else:
                binary_score += '0'
        binarized_scores_list.append(binary_score)
    return binarized_scores_list


def binarize_score_list_new(attribution_scores_list, importance_prop):
    binarized_scores_list = []
    for scores in attribution_scores_list:
        import_nodes_index = np.argsort(np.array(scores))[::-1][
                             :int(len(scores) * importance_prop)]
        binary_score_list = ['0' for i in range(len(scores))]
        for idx in import_nodes_index:
            binary_score_list[idx] = '1'
        binary_score = ''.join(binary_score_list)
        binarized_scores_list.append(binary_score)
    return binarized_scores_list


def get_contrastivity(metric_attribution_scores, dataset_features, config):
    importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
    importance_range = [float(bound) for bound in importance_range]

    # Binarize score list according to their saliency
    class_0_binarized_scores_list = binarize_score_list(
        [group[0] for group in metric_attribution_scores], importance_range)
    class_1_binarized_scores_list = binarize_score_list(
        [group[1] for group in metric_attribution_scores], importance_range)

    result_list = []
    # Calculate hamming distance
    for class_0, class_1 in zip(class_0_binarized_scores_list, class_1_binarized_scores_list):
        assert len(class_0) == len(class_1)
        d = hamming(class_0, class_1)
        result_list.append(d / len(class_0))

    return sum(result_list) / len(result_list)


def get_contrastivity_new(metric_attribution_scores, dataset_features, config):
    importance_prop = config["metrics"]["contrastivity"]["importance_prop"]

    # Binarize score list according to their saliency
    class_0_binarized_scores_list = binarize_score_list_new(
        [group[0] for group in metric_attribution_scores], importance_prop)
    class_1_binarized_scores_list = binarize_score_list_new(
        [group[1] for group in metric_attribution_scores], importance_prop)

    result_list = []
    # Calculate hamming distance
    for class_0, class_1 in zip(class_0_binarized_scores_list, class_1_binarized_scores_list):
        assert len(class_0) == len(class_1)
        d = hamming(class_0, class_1)
        result_list.append(d / len(class_0))

    return sum(result_list) / len(result_list)


# Sparsity ====================================================================
def count_salient_nodes(attribution_scores_list, important_range):
    salient_node_count_list = []
    for scores in attribution_scores_list:
        count = 0
        for score in scores:
            if is_salient(abs(float(score)), important_range):
                count += 1
        salient_node_count_list.append(count)
    return salient_node_count_list


def count_salient_nodes_new(attribution_scores_list, important_prop):
    salient_node_count_list = []
    for scores in attribution_scores_list:
        important_range = [important_prop * np.max(np.array(scores)), 1]
        count = 0
        for score in scores:
            if is_salient(abs(float(score)), important_range):
                count += 1
        salient_node_count_list.append(count)
    return salient_node_count_list


def get_sparsity(metric_attribution_scores, config):
    importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
    importance_range = [float(bound) for bound in importance_range]

    class_0_significant_nodes_count = count_salient_nodes(
        [group[0] for group in metric_attribution_scores], importance_range)
    class_1_significant_nodes_count = count_salient_nodes(
        [group[1] for group in metric_attribution_scores], importance_range)
    graphs_number_of_nodes = [group['graph'].number_of_nodes for group in metric_attribution_scores]

    # measure the average sparsity score across all samples
    result_list = []
    for i in range(len(graphs_number_of_nodes)):
        d = class_0_significant_nodes_count[i] + \
            class_1_significant_nodes_count[i]
        d /= (graphs_number_of_nodes[i] * 2)
        result_list.append(1 - d)
    return sum(result_list) / len(result_list)


def get_sparsity_new(metric_attribution_scores, config):
    # test 只要正类预测结果的sparsity
    importance_prop = config["metrics"]["sparsity"]["importance_prop"]
    nodes_cout = count_salient_nodes_new([group[group['graph'].label] for group in metric_attribution_scores],
                                         importance_prop)
    graphs_number_of_nodes = [group['graph'].number_of_nodes for group in metric_attribution_scores]
    result_list = []
    for i in range(len(graphs_number_of_nodes)):
        d = nodes_cout[i]
        d /= (graphs_number_of_nodes[i])
        result_list.append(1 - d)

    # importance_prop = config["metrics"]["sparsity"]["importance_prop"]
    # class_0_significant_nodes_count = count_salient_nodes_new(
    #     [group[0] for group in metric_attribution_scores], importance_prop)
    # class_1_significant_nodes_count = count_salient_nodes_new(
    #     [group[1] for group in metric_attribution_scores], importance_prop)
    # graphs_number_of_nodes = [group['graph'].number_of_nodes for group in metric_attribution_scores]
    #
    # # measure the average sparsity score across all samples
    # result_list = []
    # for i in range(len(graphs_number_of_nodes)):
    #     d = class_0_significant_nodes_count[i] + \
    #         class_1_significant_nodes_count[i]
    #     d /= (graphs_number_of_nodes[i] * 2)
    #     result_list.append(1 - d)
    return sum(result_list) / len(result_list)


def compute_metric(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda, **kwargs):
    if config["metrics"]["fidelity"]["enabled"] is True:
        fidelity_metric = get_fidelity_new(trained_classifier_model, metric_attribution_scores, dataset_features,
                                           config, cuda, **kwargs)
        fidelityminus_metirc = get_fidelityminus_new(trained_classifier_model, metric_attribution_scores, dataset_features,
                                           config, cuda, **kwargs)
    else:
        fidelity_metric = 0

    if config["metrics"]["contrastivity"]["enabled"] is True:
        contrastivity_metric = get_contrastivity_new(metric_attribution_scores, dataset_features, config)
    else:
        contrastivity_metric = 0

    if config["metrics"]["sparsity"]["enabled"] is True:
        sparsity_metric = get_sparsity_new(metric_attribution_scores, config)
    else:
        sparsity_metric = 0

    return fidelity_metric, contrastivity_metric, sparsity_metric, fidelityminus_metirc


def get_accuracy(config, ground_truth, metric_attribution_scores):
    importance_prop = config["metrics"]["fidelity"]["importance_prop"]
    accuracy = 0
    for score_dict in metric_attribution_scores:
        # 解释的top-k重要性二值化
        explanation = score_dict[score_dict['graph'].label]  # 该图对应的节点重要性解释
        node_explanation = [0 for i in range(len(explanation))]  # 二值化的节点重要性，前top-k个重要节点为1
        import_nodes_index = np.argsort(np.array(explanation))[::-1][
                             :int(len(explanation) * importance_prop)]  # 按照比例认为是重要的节点们
        for i in range(len(import_nodes_index)):
            if explanation[import_nodes_index[i]] != 0:  # 如果这个节点的重要性为0，不算
                node_explanation[import_nodes_index[i]] = 1

        # 获取节点的ground truth
        graph_id = score_dict['graph'].graph_id
        edges = ground_truth[0][graph_id]
        edge_truth = ground_truth[1][graph_id]
        node_truth = [0 for i in range(len(explanation))]
        for i in range(len(edge_truth)):
            if edge_truth[i] == 1:
                nodei, nodej = edges[0][i], edges[1][i]
                node_truth[nodei] = 1
                node_truth[nodej] = 1

        # 计算explanation在ground truth中的占比
        count = 0.
        for i in range(len(node_truth)):
            if node_truth[i] == 1 and node_explanation[i] == 1:
                count += 1
        accuracy += count / len(np.where(np.array(node_truth) == 1)[0])
    return accuracy / len(metric_attribution_scores)


def get_accuracy_max(config, ground_truth, metric_attribution_scores):
    """
        尝试使用 max*threshold的判定重要方式
    """
    importance_prop = config["metrics"]["sparsity"]["importance_prop"]
    accuracy = 0
    for score_dict in metric_attribution_scores:
        # 解释的top-k重要性二值化
        explanation = score_dict[score_dict['graph'].label]  # 该图对应的节点重要性解释
        node_explanation = [0 for i in range(len(explanation))]  # 二值化的节点重要性，前top-k个重要节点为1
        important_range = [importance_prop * np.max(np.array(explanation)), 1]
        for i in range(len(node_explanation)):
            if is_salient(abs(float(explanation[i])), important_range):
                node_explanation[i] = 1

        # 获取节点的ground truth
        graph_id = score_dict['graph'].graph_id
        edges = ground_truth[0][graph_id]
        edge_truth = ground_truth[1][graph_id]
        node_truth = [0 for i in range(len(explanation))]
        for i in range(len(edge_truth)):
            if edge_truth[i] == 1:
                nodei, nodej = edges[0][i], edges[1][i]
                node_truth[nodei] = 1
                node_truth[nodej] = 1

        # 计算explanation在ground truth中的占比
        count = 0.
        for i in range(len(node_truth)):
            if node_truth[i] == 1 and node_explanation[i] == 1:
                count += 1
        accuracy += count / len(np.where(np.array(node_truth) == 1)[0])
    return accuracy / len(metric_attribution_scores)
