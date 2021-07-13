import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.comm
import numpy as np

def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    # accu_x = accu_x.sum(dim=3)
    accu_y = heatmaps.sum(dim=3)
    # print("accu_x:", accu_x.shape)
    # print("accu_y:", accu_y.shape)

    # accu_y = accu_y.sum(dim=2)

    if torch.cuda.is_available():
        accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
        accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    # print("accu_x:", accu_x.shape)
    # print("accu_y:", accu_y.shape)

    return heatmaps, accu_x, accu_y

def gen_coordinate_softmax_integral(preds, num_landmarks, hm_width, hm_height):

    # global softmax
    preds = preds.reshape((preds.shape[0], num_landmarks, -1))
    preds = F.softmax(preds, 2)

    heatmaps, x, y = generate_2d_integral_preds_tensor(preds, num_landmarks, hm_width, hm_height)

    # print('x:', x.shape)
    # print('y:', y.shape)

    # integrate heatmap into joint location
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    preds = torch.cat((x, y), dim=2)   # (B, 68, 2)

    # print("preds:", preds.shape)
    preds = preds.permute(0, 2, 1)                   # (B, 2, 68)
    # reshape((preds.shape[0], 2, num_landmarks))    # (B, 2, 68)

    coords = preds.float()
    # project to original image size
    coords[:, 0, :] = (coords[:, 0, :] + 0.5) * hm_width
    coords[:, 1, :] = (coords[:, 1, :] + 0.5) * hm_height

    return heatmaps, preds, coords.long()  

pre_defined_search_range_three = {
                            0: [1, 5, 12],
                            1: [0, 12, 16],   
                            2: [16, 17,18],
                            3: [4, 15, 18], 
                            4: [3, 8, 15], 
                            5: [0, 6, 12],
                            6: [7, 13, 5], 
                            7: [8, 14, 15], 
                            8: [4, 7, 15],
                            9: [10, 11, 17], 
                            10: [9, 11, 17], 
                            11: [9, 10, 17],
                            12: [0, 5, 13], 
                            13: [6, 12, 14], 
                            14: [7, 13, 15], 
                            15: [4, 8, 14], 
                            16: [1, 9, 17],
                            17: [9, 10, 11],
                            18: [10, 11, 17]}

pre_defined_search_range_five = {
                            0: [1, 5, 12, 6, 13],
                            1: [0, 12, 16, 9, 2, 13],   
                            2: [16, 17, 18, 1, 3],
                            3: [4, 15, 18, 11, 14], 
                            4: [3, 8, 15, 14, 7], 
                            5: [0, 6, 12, 13, 1],
                            6: [7, 13, 5, 12, 14], 
                            7: [14, 15, 8, 6, 13], 
                            8: [4, 7, 15, 14, 3],
                            9: [10, 11, 17, 16, 1], 
                            10: [9, 11, 17, 16, 18], 
                            11: [9, 10, 17, 18, 3],
                            12: [0, 5, 13, 6, 1], 
                            13: [6, 12, 14, 7, 5], 
                            14: [7, 13, 15, 6, 8], 
                            15: [4, 8, 14, 7, 3], 
                            16: [1, 9, 17, 10, 2],
                            17: [9, 10, 11, 16, 18],
                            18: [10, 11, 17, 2, 3]}

# symmetric_edge = [(2,1),(2,3);(2,16),(2,18); (1,0),(3,4);(1,12),(3,15);(1,13),(3,14);(1,9),(3,11); 
#     (1,16),(3,18); (0,5),(4,8); (0,6),(4,7);(0,12),(4,15);(0,13),(4,14); (5,1),(8,3);(5,12),(8,15);
#     (5,13),(8,14); (5,6),(8,7); (9,16),(11,18); (9,17),(11,17); (9,10),(11,10); (10,16),(10,18); 
#     (12,5),(15,8); (12,1),(15,3); (12,6),(15,7); (12,13),(15,14); (13,6),(14,7);(13,7),(14,6); 
#     (15,4),(12,0); (16,17),(18,17)] 

symmetric_edge = [[(2,1),(2,3)],[(2,16),(2,18)],[(1,0),(3,4)],[(1,12),(3,15)],[(1,13),(3,14)],[(1,9),(3,11)], 
    [(1,16),(3,18)],[(0,5),(4,8)],[(0,6),(4,7)],[(0,12),(4,15)],[(0,13),(4,14)],[(5,1),(8,3)],[(5,12),(8,15)],
    [(5,13),(8,14)],[(5,6),(8,7)],[(9,16),(11,18)],[(9,17),(11,17)],[(9,10),(11,10)],[(10,16),(10,18)], 
    [(12,5),(15,8)],[(12,1),(15,3)],[(12,6),(15,7)],[(12,13),(15,14)],[(13,6),(14,7)],[(13,7),(14,6)], 
    [(15,4),(12,0)],[(16,17),(18,17)]] 

def gen_symm_mask(symmetric_edge, num_landmarks):

    symm_mask = np.zeros((num_landmarks, num_landmarks))

    for index_sym in range(len(symmetric_edge)):

        symm_edge_pair = symmetric_edge[index_sym]

        for edge in symm_edge_pair:
            i, j = edge
            symm_mask[i,j] = index_sym + 1
            symm_mask[j,i] = index_sym + 1

    return symm_mask


def gen_mask(num_landmarks):
    heatmap_mask = np.zeros((num_landmarks, num_landmarks))

    for idx in range(num_landmarks):
        search_range = pre_defined_search_range_five[idx]
    
        search_range = np.array(search_range)
        heatmap_mask[idx][search_range] = 1

    return torch.from_numpy(heatmap_mask)

def gen_edges(x, num_landmarks, coordinates, heatmap_mask):

    # input coordinates: (B, 2, 68)
    # output edge: (B, 2, 68)

    batch_size = x.size(0)
    edges = torch.zeros((batch_size, num_landmarks, num_landmarks))

    # print('coordinates:', coordinates.shape)

    for idx_batch in range(batch_size):

        heatmaps = x[idx_batch]         # (68, 256, 256)

        coordinate = coordinates[idx_batch]


        x_coordinate = coordinate[0]
        y_coordinate = coordinate[1]

        # print("x_coordinate:", x_coordinate)
        # print('y_coordinate:', y_coordinate)
        cur_img = heatmaps
        # print("cur_img:", cur_img)

        heatmap_values = cur_img[:, x_coordinate, y_coordinate]   # (68, 68)


        # heatmap_values_mask = torch.ones_like(heatmap_values)
            
        # heatmap_values_mask[idx_landmark] = 0
        # print("heatmap_values:", heatmap_values)

        # print("heatmap_mask:", heatmap_mask)
            
        heatmap_values = heatmap_values * heatmap_mask.to(heatmap_values.device)      # (68, 68)
        # print("heatmap_values_mask:", heatmap_values_mask.requires_grad)    # False
        # print("heatmap_values:", heatmap_values.requires_grad)        # True

        # print("after mask, heatmap_values:", heatmap_values)

        # _, idx_max_landmark = torch.topk(heatmap_values, k=2)
        # print("idx_max_landmark:", idx_max_landmark)

        # if idx_max_landmark[0].item() != idx_landmark: 
        #     idx_second_landmark = idx_max_landmark[0].item()
        # else:
        #     idx_second_landmark = idx_max_landmark[1].item()
        edges_batch = F.softmax(heatmap_values, dim=1)

        edges_batch_trans = edges_batch.permute(1,0)

        edges[idx_batch] = edges_batch + edges_batch_trans    # (68, 68)
        
        # print("idx_second_landmark:", idx_second_landmark)
        # print(idx_landmark, idx_second_landmark)
        # edges[idx_batch, 0, idx_landmark] = idx_landmark
        # edges[idx_batch, 1, idx_landmark] = idx_second_landmark
        
        # print(edges.requires_grad)    # False

        # print("edges:", edges.requires_grad)

    return edges


def gen_coordinate_argmax(x, num_landmarks):

    batch_size = x.size(0)
    coordinates = torch.zeros((batch_size, 2, num_landmarks))    # (B, 2, 68)

    for idx_batch in range(x.size(0)):

        heatmaps = x[idx_batch]   # (68, 256, 256)
        # get predicted keypoint coordinate (2, 68)
        for idx_landmark in range(num_landmarks):
            img = heatmaps[idx_landmark]
            # get predicted keypoint coordinate (x, y)
            M = img.argmax()
            coordinates[idx_batch, 0, idx_landmark] = M/img.size(1)
            coordinates[idx_batch, 1, idx_landmark] = M%img.size(1)

    return coordinates

class EdgeGenNetwork(nn.Module):

    def __init__(self, num_landmarks):

        super(EdgeGenNetwork, self).__init__()
        
        self.num_landmarks = num_landmarks

        self.heatmap_mask = gen_mask(num_landmarks)

    def forward(self, x, coordinates):
        
        # input landmark_heatmap: 68 heatmaps (B, 68, 256, 256)
        # output landmark_edges: [(1, 2), (2, 3), ... ] 68 edges       (B, 68, 2) 
        # output node_feature: (B, C, T, V)  ---> (B, 2, 1, 68)
        # batch_size, num_landmarks, hm_width, hm_height = x.size()

        # print("coordinates:", coordinates.shape)

        # generate edges
        landmark_edges = gen_edges(x, self.num_landmarks, coordinates, self.heatmap_mask)           # (B, 2, 68)

        # node_feature = coordinates.unsqueeze(2)     # (B, 2, 1, 68)  (B, C, T, V)

        return landmark_edges

if __name__ == "__main__":
    
    # model = EdgeGenNetwork(num_landmarks=68)

    # input_size = ((1, 68, 256, 256))

    # x = torch.randn(input_size)

    # y_feature, y_edge, y_coordinates = model(x)

    # print("y_feature:", y_feature[0])      # (12, 2, 1, 68)
    # print("y_edge:", y_edge[0])      # (12, 2, 68)
    # print("y_coordinates:", y_coordinates[0])

    symm_mask = gen_symm_mask(symmetric_edge, num_landmarks=19)

    print("symm_mask:", symm_mask)
