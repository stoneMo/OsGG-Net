import torch
import torch.nn as nn


class AdjacencyCriterion(nn.Module):

    def __init__(self, num_landmark=19, desired_node_freedom=2, dense_loss_weight=1, sparse_loss_weight=1):

        super(AdjacencyCriterion, self).__init__()


        self.num_landmark = num_landmark       
        self.desired_node_freedom = desired_node_freedom 
        self.dense_loss_weight = dense_loss_weight
        self.sparse_loss_weight = sparse_loss_weight


    def forward(self, adjacency_matrix):

        # print('adjacency_matrix:', adjacency_matrix[0])

        batch_size = adjacency_matrix.size(0)

        diagnoal_matrix = torch.zeros((self.num_landmark, self.num_landmark)).fill_diagonal_(1)
        diagnoal_matrix = diagnoal_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        adjacency_matrix = adjacency_matrix - diagnoal_matrix.to(adjacency_matrix.device)

        adjacency_matrix_total = adjacency_matrix.reshape(batch_size, -1)

        # mask0_a = torch.zeros_like(adjacency_matrix_total)
        # mask1_a = torch.ones_like(adjacency_matrix_total)
        # mask_ref = torch.ones_like(adjacency_matrix_total) * self.desired_node_freedom
        
        # adjacency_matrix_total = torch.where(adjacency_matrix_total>mask1_a, mask1_a, mask0_a)
        loss_row_total = torch.norm(adjacency_matrix_total, 1, dim=1)

        # print("loss_row_total:", loss_row_total[0])

        adjacency_column_sum = torch.sum(adjacency_matrix, dim=1)

        # print('adjacency_column_sum:', adjacency_column_sum[0])

        mask0 = torch.zeros_like(adjacency_column_sum)
        mask1 = torch.ones_like(adjacency_column_sum)
        mask_ref = torch.ones_like(adjacency_column_sum) * self.desired_node_freedom

        adjacency_column = torch.where(adjacency_column_sum<mask_ref, mask0, mask1)
        # print('adjacency_column:', adjacency_column[0])

        loss_column_total = torch.norm(adjacency_column, 1, dim=1)
        # print('loss_column_total:', loss_column_total[0])

        loss_total = self.sparse_loss_weight * loss_row_total - self.dense_loss_weight * loss_column_total
        # print('loss_total:', loss_total[0])

        return loss_total.sum()


if __name__ == "__main__":

    from ggn import GraphGenNetwork
    import os
    import matplotlib.pyplot as plt
    import random

    seed = 2021
    random.seed(seed)
    torch.manual_seed(seed)

    num_node = 8
    
    model = GraphGenNetwork(num_node=num_node, center_landmark=32, strategy='uniform', max_hop=1, dilation=1)

    input_size = ((12, 2, num_node))

    x = torch.randn(input_size).uniform_(0, num_node).long()

    print("x:", x.shape)

    A_face_graph, adjacency_face_graph = model(x)

    print("adjacency_face_graph:", adjacency_face_graph.shape)
    print("A_face_graph:", A_face_graph.shape)      # (12, 2, 1, 68)


    loss_adjacency_fn = AdjacencyCriterion(num_landmark=num_node, desired_node_freedom=1, dense_loss_weight=1, sparse_loss_weight=1)

    loss_adjacency = loss_adjacency_fn(adjacency_face_graph)

    print("loss_adjacency:", loss_adjacency)









