
# from torch.utils.serialization import load_lua
import os
import logging
import scipy.io as sio
import cv2
import math
from math import cos, sin
import time
import shutil
from PIL import Image

import matplotlib.pyplot as plt
import networkx as nx

import sys
# import keras
# import tensorflow as tf

import numpy as np
import torch


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out

def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def augment_data(image):

    rand_r = np.random.random()
    if  rand_r < 0.25:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop(image,dn)

    elif rand_r >= 0.25 and rand_r < 0.5:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_black(image,dn)

    elif rand_r >= 0.5 and rand_r < 0.75:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_white(image,dn)
    
    # if np.random.random() > 0.3:
    #     images[i] = tf.contrib.keras.preprocessing.image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
    
    return image

def experiment_config(parser, args):
    """ Handles experiment configuration and creates new dirs for model.
    """
    # check number of models already saved in 'experiments' dir, add 1 to get new model number
    run_dir = os.path.join(os.path.split(os.getcwd())[0], 'experiments')

    os.makedirs(run_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create all save dirs
    model_dir = os.path.join(run_dir, run_name)

    os.makedirs(model_dir, exist_ok=True)

    args.summaries_dir = os.path.join(model_dir, 'summaries')
    args.checkpoint_dir = os.path.join(model_dir, 'checkpoint.pt')

    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameters in .txt file
    with open(os.path.join(model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1} \n'.format(str(key), str(value)))

    # save config file used in .txt file
    with open(os.path.join(model_dir, 'config.txt'), 'w') as logs:
        # Remove the string from the blur_sigma value list
        config = parser.format_values().replace("'", "")
        # Remove the first line, path to original config file
        config = config[config.find('\n')+1:]
        logs.write('{}'.format(config))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(os.path.join(model_dir, 'trainlogs.txt')),
                                  logging.StreamHandler()])
    return args

def print_network(model, args):
    """ Utility for printing out a model's architecture.
    """
    logging.info('-'*70)  # print some info on architecture
    logging.info('{:>25} {:>27} {:>15}'.format('Layer.Parameter', 'Shape', 'Param#'))
    logging.info('-'*70)

    for param in model.state_dict():
        p_name = param.split('.')[-2]+'.'+param.split('.')[-1]
        # don't print batch norm layers for prettyness
        if p_name[:2] != 'BN' and p_name[:2] != 'bn':
            logging.info(
                '{:>25} {:>27} {:>15}'.format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    '{0:,}'.format(np.product(list(model.state_dict()[param].size())))
                )
            )
    logging.info('-'*70)

    logging.info('\nTotal params: {:,}\n\nSummaries dir: {}\n'.format(
        sum(p.numel() for p in model.parameters()),
        args.summaries_dir))

    for key, value in vars(args).items():
        if str(key) != 'print_progress':
            logging.info('--{0}: {1}'.format(str(key), str(value)))

def save_checkpoint(state, is_best, save):

    save_path = "/".join(save.split("/")[:-1])
    filename = os.path.join(save_path, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)

def visual_heatmap(idx_select, pred_heatmap, prefix, epoch, save_dir):

    pred_heatmap = pred_heatmap.detach().cpu().numpy()
    heatmaps_size = pred_heatmap[0,0].shape
    
    heatmaps = pred_heatmap[idx_select]
    # heatmap: (68, 256, 256)
    heatmap_img=np.zeros(heatmaps_size,dtype=np.float)
    for index in range(len(heatmaps)):
        heatmap_img+=heatmaps[index, :,:]*255.0
    heatmap_img_path = os.path.join(save_dir, 'e{}_b{}_heatmap_{}.jpg'.format(epoch, idx_select, prefix))
    Image.fromarray(heatmap_img).convert('RGB').save(heatmap_img_path)

def save_graph_with_labels(idx_select, face_graph, mylabels, prefix, epoch, save_dir, visual_threshold):

    # face_graph: (B, 68, 68)
    face_graph = face_graph.detach().cpu().numpy()

    adjacency_matrix = face_graph[idx_select]
    rows_mask, cols_mask = np.where(adjacency_matrix < visual_threshold)
    adjacency_matrix[rows_mask,cols_mask] = 0
    adjacency_matrix[cols_mask,rows_mask] = 0

    rows, cols = np.where(adjacency_matrix >= visual_threshold)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    
    graph_img_path = os.path.join(save_dir, 'e{}_b{}_graph_{}.jpg'.format(epoch, idx_select, prefix))

    plt.savefig(graph_img_path)

symmetric_edge = [[(2,1),(2,3)],[(2,16),(2,18)],[(1,0),(3,4)],[(1,12),(3,15)],[(1,13),(3,14)],[(1,9),(3,11)], 
    [(1,16),(3,18)],[(0,5),(4,8)],[(0,6),(4,7)],[(0,12),(4,15)],[(0,13),(4,14)],[(5,1),(8,3)],[(5,12),(8,15)],
    [(5,13),(8,14)],[(5,6),(8,7)],[(9,16),(11,18)],[(9,17),(11,17)],[(9,10),(11,10)],[(10,16),(10,18)], 
    [(12,5),(15,8)],[(12,1),(15,3)],[(12,6),(15,7)],[(12,13),(15,14)],[(13,6),(14,7)],[(13,7),(14,6)], 
    [(15,4),(12,0)],[(16,17),(18,17)]] 

def gen_symm_mask(symmetric_edge, num_landmarks, edges_mask):

    symm_mask = np.zeros((num_landmarks, num_landmarks))

    for index_sym in range(len(symmetric_edge)):

        symm_edge_pair = symmetric_edge[index_sym]

        for edge in symm_edge_pair:
            i, j = edge
            symm_mask[i,j] = index_sym + 1
            symm_mask[j,i] = index_sym + 1

    rows_mask = []
    col_mask = []

    for row, col in edges_mask:
        index_sym_pair = symm_mask[row, col]
        sym_rows, sym_col = np.where(symm_mask == index_sym_pair)
        rows_mask.extend(sym_rows.tolist())
        col_mask.extend(sym_col.tolist())

    return np.array(rows_mask), np.array(col_mask)
    
def save_graph_with_face(idx_select, face_graph, input_image, pred_landmark, prefix, epoch, save_dir, visual_threshold):

    # face_graph: (B, 68, 68)
    face_graph = face_graph.detach().cpu().numpy()
    input_image = input_image.detach().cpu()
    pred_landmark = pred_landmark.detach().cpu().numpy()

    adjacency_matrix = face_graph[idx_select]
    num_landmarks = adjacency_matrix.shape[1]

    print("adjacency_matrix:", adjacency_matrix)

    rows_mask, cols_mask = np.where(adjacency_matrix < visual_threshold)
    edges_mask = zip(rows_mask.tolist(), cols_mask.tolist())
    sym_rows_mask, sym_cols_mask = gen_symm_mask(symmetric_edge, num_landmarks, edges_mask)

    # adjacency_matrix[rows_mask,cols_mask] = 0
    # adjacency_matrix[cols_mask,rows_mask] = 0
    adjacency_matrix[sym_rows_mask,sym_cols_mask] = 0
    adjacency_matrix[sym_cols_mask,sym_rows_mask] = 0

    print("adjacency_matrix_mask:", adjacency_matrix)

    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())

    landmarks = pred_landmark[idx_select]

    img = input_image[idx_select]
    img = tensor2im(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    print("img:", img.shape)
    # draw landmarks
    for j in range(len(landmarks)):
        x = int((landmarks[j][0] + 0.5) * img.shape[1]) 
        y = int((landmarks[j][1] + 0.5) * img.shape[0])
        img = cv2.circle(img, (x,y), radius=1, color=(0, 0, 255), thickness=1)
    
    print("edges:", edges)
    # draw edges
    for row, col in edges:
        x_1 = int((landmarks[row][0] + 0.5) * img.shape[1])
        y_1 = int((landmarks[row][1] + 0.5) * img.shape[0])
        start_point = (x_1, y_1)
        x_2 = int((landmarks[col][0] + 0.5) * img.shape[1])
        y_2 = int((landmarks[col][1] + 0.5) * img.shape[0])
        end_point = (x_2, y_2)
        img = cv2.line(img, start_point, end_point, color=(0, 0, 255), thickness=1)

    graph_face_path = os.path.join(save_dir, 'e{}_b{}_graph_face_{}.jpg'.format(epoch, idx_select, prefix))

    cv2.imwrite(graph_face_path, img)

def plot_heatmap_graph(pred_graph, pred_landmark, pred_heatmap, img, gt_landmark, gt_heatmap, epoch, prefix, args):

    idx_select = np.random.randint(len(pred_graph))

    save_graph_with_labels(idx_select, pred_graph, None, 'pred_'+prefix, epoch+1, args.heatmap_dir, args.visual_threshold)

    save_graph_with_face(idx_select, pred_graph, img, pred_landmark, 'pred_'+prefix, epoch+1, args.heatmap_dir, args.visual_threshold)

    save_graph_with_face(idx_select, pred_graph, img, gt_landmark, 'gt_'+prefix, epoch+1, args.heatmap_dir, args.visual_threshold)

    # # visualize pred_heatmap: (B, 256, 256, 68)
    visual_heatmap(idx_select, pred_heatmap, 'pred_'+prefix, epoch+1, args.heatmap_dir)

    # # visualize gt_heatmap: (B, 256, 256, 68)
    visual_heatmap(idx_select, gt_heatmap, 'gt_'+prefix, epoch+1, args.heatmap_dir)

def tensor2im(input_image, imtype=np.uint8):
    
    mean = [0.485,0.456,0.406] 
    std = [0.229,0.224,0.225]  
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): 
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): 
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = np.stack([image_numpy[:,:,2], image_numpy[:,:,1], image_numpy[:,:,0]],axis=2)
        # print("image_numpy:", image_numpy.shape)
    else:  # 
        image_numpy = input_image
    return image_numpy.astype(imtype)