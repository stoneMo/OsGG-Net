#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
# torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import transforms as transforms

# import face_alignment

from network.osggnet import OsGGNet
from datasets import OsGGData

from utils import experiment_config, print_network
from train_test import pipeline

def parse_args():
    """Parse input arguments."""
    parser = configargparse.ArgumentParser(description='Head pose estimation using the One-step graph generation(OsGG) network.')
    parser.add_argument('--gpu_id', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=str)
    parser.add_argument('--epochs', dest='epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--learn_rate', dest='learn_rate', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument("--optim_type", type=str, default='Adam', help='optimizer: SGD or Adam')
    parser.add_argument("--gradient_clip", type=bool, default=True, help='Gradient clip')
    parser.add_argument('--train_dataset', dest='train_dataset', help='Train Dataset type.', default='300W_LP', type=str)
    parser.add_argument('--valid_dataset', dest='valid_dataset', help='Valid Dataset type.', default='AFLW2000', type=str)    
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--resume_path', type=str, default='EXP', help='resume experiment name')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from last saved best checkpoint')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate or not')
    parser.add_argument('--distributed', action='store_true', default=False, help='Whether or Not to Use Distributed Training')

    parser.add_argument('--num_landmark', dest='num_landmark', help='Number of landmarks.',
          default=68, type=int)
    parser.add_argument('--desired_node_freedom', dest='desired_node_freedom', help='Desired Node Freedom.',
          default=2, type=int)
    parser.add_argument('--dense_loss_weight', dest='dense_loss_weight', help='Graph dense loss weight.',
          default=1., type=float)
    parser.add_argument('--sparse_loss_weight', dest='sparse_loss_weight', help='Graph sparse loss weight.',
          default=1., type=float)

    parser.add_argument('--landmark_loss_weight', dest='landmark_loss_weight', help='Landmark loss weight.',
          default=1., type=float)
    parser.add_argument('--heatmap_loss_weight', dest='heatmap_loss_weight', help='Heatmap loss weight.',
          default=1., type=float)
    parser.add_argument('--graph_loss_weight', dest='graph_loss_weight', help='Graph loss weight.',
          default=1., type=float)

    parser.add_argument('--vis_heatmap', action='store_true', default=False, help='Whether or Not to visualize heatmap')
    parser.add_argument('--heatmap_dir', dest='heatmap_dir', help='Directory path for saving heatmaps.',
          default='./heatmap', type=str)

    parser.add_argument('--hm_loss_type', dest='hm_loss_type', help='Loss type for heatmap loss.',
          default='l1', type=str)

    parser.add_argument('--model_choice', dest='model_choice', help='Model choice.',
          default='landmark', type=str)

    parser.add_argument('--visual_threshold', dest='visual_threshold', help='Adjacency visual threshold.',
          default=0.5, type=float)
    parser.add_argument('--adj_threshold', dest='adj_threshold', help='Adjacency threshold.',
          default=0.5, type=float)

    parser.add_argument('--yaw_loss_weight', dest='yaw_loss_weight', help='Yaw loss weight.',
          default=0.001, type=float)
    parser.add_argument('--pitch_loss_weight', dest='pitch_loss_weight', help='Pitch loss weight.',
          default=0.0015, type=float)
    parser.add_argument('--roll_loss_weight', dest='roll_loss_weight', help='Roll loss weight.',
          default=0.001, type=float)

    args = parser.parse_args()
    return parser, args

def setup(distributed):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed

    Taken from https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate

    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 420
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank

def main():
    """ Main """

    # Arguments
    parser, args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Setup Distributed Training
    # device, local_rank = setup(distributed=args.distributed)

    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    # transforms 
    train_transformations = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # transforms 
    valid_transformations = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # dataset & dataloader
    train_dataset = OsGGData(dataset_name=args.train_dataset, data_dir=args.data_dir, num_landmark=args.num_landmark, transform=train_transformations, train=False)
    valid_dataset = OsGGData(dataset_name=args.valid_dataset, data_dir=args.data_dir, num_landmark=args.num_landmark, transform=valid_transformations, train=False)
    
    if args.num_landmark == 19:
        center_landmark = 10
    elif args.num_landmark == 68:
        center_landmark == 32

    # model 
    model = OsGGNet(num_landmarks=args.num_landmark, center_landmark=center_landmark, strategy='uniform', choice=args.model_choice, adj_threshold=args.adj_threshold)

    # Place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(6)  # n cpu threads / n processes per node

        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        # Only print from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        # If non Distributed use DataParallel
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

        # model.to(device)
        model = model.cuda()
        
        args.print_progress = True

    # Print Network Structure and Params
    if args.print_progress:
        print_network(model, args)  # prints out the network architecture etc
        logging.info('\ntrain: {} - valid: {}'.format(len(train_dataset), len(valid_dataset)))
    
    best_validation_loss, best_train_loss, time_dif, best_loss_yaw, best_loss_pitch, \
            best_loss_roll = pipeline(train_dataset, valid_dataset, model, args) 

    logging.info('\nbest_validation_loss: {} - best_train_loss: {}'.format(best_validation_loss, best_train_loss))
    logging.info('\nbest_loss - yaw: {} - pitch: {} - roll: {}'.format(best_loss_yaw, best_loss_pitch, best_loss_roll))
    logging.info('\ntime cost: {}'.format(time_dif))


if __name__ == "__main__":
    main()
    
# dataloader
# images 
# groundtruth gaussian heatmaps (landmarks locations) 
# yaw, pitch, roll


# model 
# step 1
# input an image
# CNN 
# FCN 
# 68 heatmaps for localizing two landmarks 
# p_i: node 1 for regression of groundtruth landmarks
# q_i: the highest activation value among the other 67 landmarks

# step 2
# linking p_i and q_i

# step 3
# merge the 68 heatmaps to form a graph

# step 4
# feed the graph into GCN to get yaw, pitch, roll


# loss
# location (gaussian heatmap for regressing the landmarks)
# cocurrence (graph distance loss)
# mse(yaw, pitch, roll)

# 