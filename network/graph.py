import numpy as np

class Graph():
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

    def __init__(self,
                 layout='FAN_19',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

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

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
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
