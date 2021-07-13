import torch
import torch.nn as nn
import numpy as np

class GraphGenNetwork(nn.Module):
    """ The Graph to model the facial landmarks extracted by the FAN. 

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - spatial: Spatial Configuration

        layout (string): must be one of the follow candidates
        - FAN_19: Is consists of 19 joints. 
        - FAN_21: Is consists of 21 joints, including two pupil landmarks.

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, num_node=68, center_landmark=32, strategy='uniform', max_hop=1, dilation=1):

        super(GraphGenNetwork, self).__init__()

        self.max_hop = max_hop
        self.dilation = dilation

        self.strategy = strategy
        # self.layout = layout

        self.num_node = num_node

        self.center = center_landmark

        self.node_selflink = np.array([[i, i] for i in range(self.num_node)]).T


    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'FAN_21':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16 ,1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1)]  #40 edges
            pupil_link = [(19, 12), (19, 6), (19, 13), (19, 9), 
                            (20, 7), (20, 14), (20, 15), (20, 11)]   # 8 pupil edges 
            self.edge = self_link + neighbor_link + pupil_link
            self.center = 10
        if layout == 'FAN_19':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16 ,1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1)]  #40 edges
            self.edge = self_link + neighbor_link
            self.center = 10
        # else:
        #     raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, hop_dis, strategy):

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1


        
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            A_result = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
            A_result = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            if hop_dis[j, self.center] == hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center] > hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            A_result = A
        else:
            raise ValueError("Do Not Exist This Strategy")

        return A_result, adjacency

    
    def forward(self, x):

        # input landmark_edges: [(1, 2), (2, 3), ... ] 68 edges (B, 2, 68)
        # output face_graph: generate Adjacency Matrix
        # self.get_edge(self.layout)

        batch_size = x.size(0)

        A_list = []
        adjacency_list = []

        for i in range(batch_size):

            edge = x[i].numpy()       #(2, 68)

            # print("edge:", edge.shape)
            # print("self.node_selflink:", self.node_selflink.shape)
            edge = np.concatenate([edge, self.node_selflink], axis=1)
            
            # if i == 0:
            #     print("x[i]:", x[i])
            #     print("edge:", edge)

            hop_dis = get_hop_distance(self.num_node, edge, max_hop=self.max_hop)

            A, adjacency = self.get_adjacency(hop_dis, self.strategy)

            # if i == 0:
            #     print("hop_dis:", hop_dis)
            #     print("adjacency:", adjacency)

            A = torch.from_numpy(A)
            adjacency = torch.from_numpy(adjacency)

            A_list.append(A)
            adjacency_list.append(adjacency)

        A_face_graph = torch.stack(A_list)
        adjacency_face_graph = torch.stack(adjacency_list)

        # print("A_face_graph:", A_face_graph.shape)

        return A_face_graph, adjacency_face_graph


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    num_edges = edge.shape[1]
    for idx_edge in range(num_edges):
        i = edge[0, idx_edge]
        j = edge[1, idx_edge]
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


if __name__ == "__main__":

    import networkx as nx
    import os
    import matplotlib.pyplot as plt
    import random

    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_node = 68
    
    model = GraphGenNetwork(num_node=num_node, center_landmark=32, strategy='spatial', max_hop=1, dilation=1)

    input_size = ((12, 2, num_node))

    x = torch.randn(input_size).uniform_(0, num_node).long()

    print("x:", x.shape)

    A_face_graph, adjacency_face_graph = model(x)

    print("A_face_graph:", A_face_graph.shape)      # (12, 2, 1, 68)

    print("adjacency_face_graph:", adjacency_face_graph.shape)      # (12, 68, 68)

    print(adjacency_face_graph[0])


    adjacency_matrix = adjacency_face_graph[0].numpy()

    print('adjacency_matrix:', adjacency_matrix)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=None, with_labels=True)
    
    graph_img_path = os.path.join('./', 'e{}_b{}_graph_{}.jpg'.format(2, 0, 'debug'))

    plt.savefig(graph_img_path)

