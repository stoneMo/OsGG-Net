import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

# from fcn import FCN
# from egn import gen_coordinate_softmax_integral
# from egn import EdgeGenNetwork
# from ggn_torch_mat import GraphGenNetwork
# from posedecoder import PoseDecoder

from network.fcn import FCN
from network.posedecoder import PoseDecoder
from network.egn_softmax import gen_coordinate_softmax_integral
from network.egn_softmax import EdgeGenNetwork
# from network.ggn import GraphGenNetwork
from network.ggn_torch_mat import GraphGenNetwork

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class OsGGNet(nn.Module):
    # OsGGNet with 68 output heatmaps of each landmark and 3 output layers for yaw, pitch and roll
    def __init__(self, num_landmarks=68, center_landmark=32, strategy='spatial', choice='landmark', adj_threshold=0.1):

        super(OsGGNet, self).__init__()

        self.landmark_regession_network = FCN(pretrained=True, nparts=num_landmarks)

        self.edge_generation_network = EdgeGenNetwork(num_landmarks=num_landmarks)
        
        self.graph_generation_network = GraphGenNetwork(num_node=num_landmarks, center_landmark=center_landmark, strategy=strategy, max_hop=1, dilation=1)

        self.pose_decoder = PoseDecoder(in_channels=2, spatial_kernel_size=3, num_landmark_node=num_landmarks, edge_importance_weighting=False)

        self.pose_decoder.apply(weights_init)

        self.thresh_layer = nn.Threshold(adj_threshold, 0, False)

        self.num_landmarks = num_landmarks

        self.spatial_kernel_size = 1 if strategy == 'uniform' else 3 

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x, choice='landmark'):


        # step 1
        # feed an image input x to a landmark regression network, like FCN 
        # x: (B, 3, 256, 256)
        landmark_heatmap = self.landmark_regession_network(x)  # 68 heatmaps (B, 68, 256, 256)
        pred_heatmap = landmark_heatmap     # (B, 68, 256, 256)

        # print("pred_heatmap:", pred_heatmap.shape)

        hm_width, hm_height = pred_heatmap.shape[2:4]

        # generate coordinates 
        softmax_heatmaps, pred_landmark, coordinates = gen_coordinate_softmax_integral(landmark_heatmap, self.num_landmarks, hm_width, hm_height)    # (B, 2, 68)

        # print("pred_landmark:", pred_landmark.shape)                  # (B, 2, 68)
        # print("coordinates:", coordinates.shape)      # (B, 2, 68)
        # 68 heatmaps for localizing two landmarks 
        # p_i: node 1 for regression of groundtruth landmarks
        # q_i: the highest activation value among the other 67 landmarks
        # step 2
        # linking p_i and q_i
        # generate 68 edges for 68 heatmaps
        # if choice == 'landmark+pose': 

        adjacency_face_graph = self.edge_generation_network(softmax_heatmaps, coordinates)

        # print("landmark_edges:", landmark_edges)

        # node_feature: (B, C, T, V)
        # landmark_edges: (B, 2, 68)
        # [(1, 2), (2, 3), ... ] 68 edges

        # step 3
        # merge the 68 edges to form a graph for generating adjacency matrix
        # face_graph, adjacency_face_graph = self.graph_generation_network(landmark_edges)    # 
        # face_graph: A


        # clip adjacency graph
        # TO DO

        adjacency_face_graph = self.thresh_layer(adjacency_face_graph)
        

        adjacency_face_graph = adjacency_face_graph.to(x.device)

        face_graph = adjacency_face_graph.unsqueeze(1)

        node_feature = pred_landmark.unsqueeze(2)

        # step 4
        # feed the graph into GCN to get yaw, pitch, roll
        pred_pose, _ = self.pose_decoder(node_feature, face_graph)

        pred_landmark = pred_landmark.permute(0, 2, 1)

        return pred_heatmap, pred_landmark, pred_pose, adjacency_face_graph.to(x.device), adjacency_face_graph.to(x.device)

if __name__ == "__main__":
    
    model = OsGGNet(num_landmarks=68, center_landmark=32, strategy='uniform')

    input_size = ((12, 3, 256, 256))

    x = torch.randn(input_size)

    # y_preds, y_coordinates = model(x, choice='landmark')

    y_heatmap, y_landmark, y_headpose, landmark_edges, face_graph = model(x, choice='landmark+pose')

    print("y_heatmap:", y_heatmap.shape)      # (12, 68, 64, 64)
    print("y_landmark:", y_landmark.shape)      # (12, 2, 68)

