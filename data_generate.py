import networkx as nx
import pickle
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open("data/BA-2motif/BA-2motif.pkl", 'rb') as pfile:
        (adjs, feas, labels) = pickle.load(pfile)
        pfile.close()

    graph_list = []
    for (adj, fea, label) in zip(adjs, feas, labels):
        G = nx.Graph(adj)
        G.graph['label'] = np.argmax(label)
        for node in G.nodes:
            G.nodes[node]['attribute'] = fea[node]*10
        graph_list.append(G)

    with open("data/BA-2motif/BA-2motif.p", 'wb') as pfile:
        pickle.dump(graph_list, pfile)
        pfile.close()