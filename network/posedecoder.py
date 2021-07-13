import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from tgcn import ConvTemporalGraphical
# from graph import Graph

from network.tgcn import ConvTemporalGraphical
from network.graph import Graph
import random

SEED = 2020

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 19)


class PoseDecoder(nn.Module):
    r"""PoseDecoder.

    Args:
        in_channels (int): Number of channels in the input data
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        device (str): 'cpu' or 'cuda'
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, C_{in})`
        - Output: #head pose (N, T, 1, 3), landmark (N, T, 19, 2)
            :math:`(N, T, 1, 3) & (N, T, 19, 2)` where
            :math:`N` is a batch size,
            :math:`T` is a number of forecasted frames for the long-term head pose forecasting task

    """

    def __init__(self, in_channels, spatial_kernel_size, num_landmark_node,
                 edge_importance_weighting, device='cpu', **kwargs):
        super().__init__()
        torch.manual_seed(SEED)
        random.seed(2020)
        
        # load graph_decoder
        # self.graph_de = Graph(layout='FAN_19', strategy='spatial')  #(3, 19, 19)
        # A_d = torch.tensor(self.graph_de.A, dtype=torch.float32, requires_grad=False, device=device)
        # self.register_buffer('A_d', A_d)

        self.A_d = torch.ones((spatial_kernel_size, num_landmark_node, num_landmark_node))

        # build networks
        # spatial_kernel_size = 3  # 3
        temporal_kernel_size = 1
        # num_landmark_node = 68
        kernel_size = (temporal_kernel_size, spatial_kernel_size) #(1, 3)
        self.data_bn = nn.BatchNorm1d(in_channels * num_landmark_node)     # 2 * 19
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}        
        self.st_gcn_networks = nn.ModuleList((

            # --------------------------------------------------PoseDecoder -----------------------------------------
            st_gcn(2, 64, (1, 1), 1, residual=False, **kwargs0),
            st_gcn(64, 64, (1, 1), 1, **kwargs),
            st_gcn(64, 64, (1, 1), 1, **kwargs),
            st_gcn(64, 64, (1, 1), 1, **kwargs),
            st_gcn(64, 128, (1, 1), 2, **kwargs),
            # -------------------------------------------------------------------------------------------------------
        ))
        
        # fcn for prediction
        self.fcn = nn.Conv2d(128, 256, kernel_size=1)
       
        # linear layer
        self.linear = nn.Linear(256, 3)
        
    
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            
            self.edge_importance = nn.ParameterList([          #decoder_pose graph edge 
                nn.Parameter(torch.ones(self.A_d.size()))
                for i in self.st_gcn_networks
            ])
            
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def decode_pose(self, z, face_graph):
        
        N, C, T, V = z.size()  # 输入tensor：(N, C, T, V)
        z = z.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        z = z.view(N, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, V, C, T)
        z = z.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)  --> (N, 64, 1, 21)
        
        z_pose = z      #(N, 64, 1, 21)
        #print(z.shape)

        # print("face_graph:", face_graph.shape)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            z_pose, A = gcn(z_pose, face_graph * importance)  # (N, C=128, T=1, V=68)
            
#         print(z_pose.shape)
            
        y = F.avg_pool2d(z_pose, z_pose.size()[2:])    
        y = y.view(N, -1, 1, 1)      #(N, 128, 1, 1)
       
        # prediction 
        y = self.fcn(y)           #(N, 256, 1, 1)
        
        y = y.view(y.size(0), -1)

        y_pose = self.linear(y)     #(N, 3)
        y_pose = y_pose.view(N, 3)
        
        return y_pose, A

    def forward(self, x, face_graph):


        # input x: (B, V, C)  ---> (B, 68, 2)
        # face_graph A: (3, 68, 68)

        # x = x.unsqueeze(1)

        # x = x.permute(0, 3, 1, 2).contiguous() #(B, T, V, C) ---->(B, C, T, V)
#         print(x.shape)
        
        y_pose, A_pred = self.decode_pose(x, face_graph)

        pre_pose = y_pose.squeeze(-1).squeeze(-1)  # (B, 3)
        
        
        return pre_pose, A_pred



class st_gcn(nn.Module):
    r"""Spatial temporal graph convolution network (ST-GCN).

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 last_layer=False,
                 decoder_layer=False,
                 re_layer=False):
        super().__init__()

        assert len(kernel_size) == 2
        #print(kernel_size[0])
        if kernel_size[0] == 3 and not decoder_layer:
            padding = ((kernel_size[0] - 1) // 2, 0)
            stride_res = 2
        elif re_layer:
            padding = (0, 0)
            stride_res = 2
        elif last_layer:
            padding = ((kernel_size[0] + 1) //2, 0) 
            stride_res = 4
        elif not last_layer and kernel_size[0] != 1: 
            padding = ((kernel_size[0] + 1) //2, 0) 
            stride_res = 2
        elif kernel_size[0] == 1:
            padding = ((kernel_size[0] - 1) // 2, 0)
            stride_res = stride

        # GCN
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # TCN
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else: 
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride_res, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        self.kernel_size = kernel_size

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        tcn = self.tcn(x)
        x = tcn + res
        
        return self.relu(x), A


if __name__ == "__main__":
    
    model = PoseDecoder(in_channels=2, spatial_kernel_size=3, num_landmark_node=68, edge_importance_weighting=False)

    input_size = ((12, 68, 2))

    x = torch.randn(input_size)

    A = torch.randn((12, 3, 68, 68))

    y, A_pred = model(x, A)

    print("y:", y.shape)      # (12, 3)
    print("A_pred:", A_pred.shape)      # (12, 3, 68, 68)