import numpy as np
import networkx as nx
import pickle

from utilities.load_data import load_model_data
from utilities.util import graph_to_tensor

"""
    用于数据格式转换的
"""
if __name__ == "__main__":
    feats = []
    adjs = []
    labels = []

    train_graphs, test_graphs, dataset_features = load_model_data("TOX21_AR", 2, False, True)
    max_num_nodes = dataset_features["max_num_nodes"]
    feat_dim = dataset_features["feat_dim"]
    print("feat_dim: %d" % feat_dim)
    graphs = train_graphs[0]
    graphs.extend(test_graphs[0])
    for graph in graphs:
        node_feat, n2n, subg = graph_to_tensor([graph], dataset_features["feat_dim"], dataset_features["edge_feat_dim"],
                                               0)
        # 需要的三个数据
        label = np.array([0, 0])
        label[graph.label] = 1

        feat = node_feat.detach().numpy()
        feat = np.concatenate([feat, np.zeros([max_num_nodes - graph.number_of_nodes, feat_dim])])

        adj = n2n.to_dense().numpy()
        adj = np.concatenate([adj, np.zeros([graph.number_of_nodes, max_num_nodes - graph.number_of_nodes])], axis=1)
        adj = np.concatenate(
            [adj, np.zeros([max_num_nodes - graph.number_of_nodes, max_num_nodes])])  # 是否需要添加self loop？

        labels.append(label)
        adjs.append(adj)
        feats.append(feat)
    with open('../data/TOX21_AR/TOX21_AR.pkl', 'wb') as fout:
        pickle.dump((np.array(adjs), np.array(feats), np.array(labels)), fout, protocol=4)
