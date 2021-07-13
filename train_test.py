# sys
import os
import sys
import numpy as np
import random
import pickle
import logging

# time
import time

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# for hyperparameter
from collections import OrderedDict
from collections import namedtuple
from itertools import product

# from data_load import Dataset, data_load
# from transfer_model import transfer_model
import utils

from network.losses import AdjacencyCriterion

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

def train(epoch, model, train_data_loader, optimizer, batch_size, params, gradient_clip, max_norm, args, is_best):
    #loss_functions
    if args.hm_loss_type == "l1":
        loss_fn1 = nn.L1Loss(reduction='sum').cuda()
        loss_fn1_heatmap = nn.L1Loss().cuda()
    elif args.hm_loss_type == "mse":
        loss_fn1 = nn.MSELoss(reduction='sum').cuda()
        loss_fn1_heatmap = nn.MSELoss().cuda()
    loss_fn2 = nn.MSELoss(reduction='sum').cuda()
    # loss_fn3 = nn.MSELoss().cuda()
    loss_fn_graph = AdjacencyCriterion(num_landmark=args.num_landmark, desired_node_freedom=args.desired_node_freedom, dense_loss_weight=args.dense_loss_weight, sparse_loss_weight=args.sparse_loss_weight).cuda()

    loss_train = []
    pose_loss_train = []
    landmark_loss_train = []
    graph_loss_train = []
    heatmap_loss_train = []

    model.train()

    for idx_batch, data_batch in enumerate(train_data_loader):
        # get data
        img, gt_pose, gt_heatmap, gt_landmark, ori_landmark = data_batch
        
        img = img.float().cuda(non_blocking=True)                 # (B, 3, 256, 256)
        gt_pose = gt_pose.float().cuda(non_blocking=True)         # (B, 3)
        gt_heatmap = gt_heatmap.float().cuda(non_blocking=True)   # (B, 68, 256, 256)
        gt_landmark = gt_landmark.float().cuda(non_blocking=True) # (B, 68, 2)
        ori_landmark = ori_landmark.float().cuda(non_blocking=True) # (B, 68, 2)

        # img = img.float().cuda()                 # (B, 3, 256, 256)
        # gt_pose = gt_pose.float().cuda()         # (B, 3)
        # gt_heatmap = gt_heatmap.float().cuda()   # (B, 68, 256, 256)
        # gt_landmark = gt_landmark.float().cuda() # (B, 68, 2)

        yaw_label, pitch_label, roll_label = gt_pose[:,0], gt_pose[:,1], gt_pose[:,2]

        # forward
        # img = torch.autograd.Variable(img)

        # print("image:", img.device)

        
        pred_heatmap, pred_landmark, pred_pose, pred_edges, pred_graph = model(img)

        yaw, pitch, roll = pred_pose[:,0], pred_pose[:,1], pred_pose[:,2]          
        
        # loss
        # alpha, beta = 0.001,0.0015
        Pitch_loss = args.pitch_loss_weight * loss_fn2(pitch, pitch_label)
        Yaw_loss = args.yaw_loss_weight * loss_fn2(yaw, yaw_label)
        Roll_loss = args.roll_loss_weight * loss_fn2(roll, roll_label)

        pose_loss = (Yaw_loss + Pitch_loss + Roll_loss) /batch_size  

        # print("pred_landmark:", pred_landmark.shape)        # (B, 68, 256, 256)
        # print("gt_heatmap:", gt_heatmap.shape)

        # landmark_loss = loss_fn1(pred_landmark, gt_heatmap) /batch_size 

        # target_heatmap = torch.autograd.Variable(gt_heatmap)

        landmark_loss = (loss_fn1(pred_landmark, gt_landmark)).sum() /batch_size
        
        heatmap_loss = loss_fn1_heatmap(pred_heatmap, gt_heatmap)
        # how much landmark errors affect the final score

        # visualize landmark_edges
        # TO DO

        # visualize pred_garph
        # TO DO
        # does not make sense: genmetry restriction 
        # multi-graph: merge

        # TO DO

        graph_loss = loss_fn_graph(pred_graph) /batch_size
        
        loss = pose_loss + args.landmark_loss_weight * landmark_loss + args.heatmap_loss_weight * heatmap_loss + graph_loss * args.graph_loss_weight

        # backward
        optimizer.zero_grad()
        loss.backward()

        if gradient_clip: nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()

        # statistics
        loss_train.append(loss.data.item())
        pose_loss_train.append(pose_loss.data.item())
        landmark_loss_train.append(landmark_loss.data.item())
        graph_loss_train.append(graph_loss.data.item())
        heatmap_loss_train.append(heatmap_loss.data.item())


        # msg ='[epoch:{}] [batch:{}] [train_loss_total:{}] [landmark_loss:{}] [yaw_loss:{}] [pitch_loss:{}] [roll_loss:{}]'

        # if is_best and (idx_batch+1) % 100 == 0:

        #     # if (idx_batch+1) % 100 == 0 and (epoch+1) % 10 == 0:
        #     logging.info(msg.format(epoch+1, idx_batch+1, loss.item(), landmark_loss.item(), Yaw_loss.item(), Pitch_loss.item(), Roll_loss.item()))
        
        #     # # visualize pred_garph: (B, 1, 68, 68)
        #     # TO DO

        if idx_batch == len(train_data_loader) - 1:
            select_img = img
            select_pred_graph = pred_graph
            select_pred_landmark = pred_landmark
            select_pred_heatmap= pred_heatmap
            select_gt_heatmap = gt_heatmap
            select_gt_landmark = gt_landmark

        del img, gt_pose, gt_heatmap, gt_landmark, pred_landmark, pred_pose, yaw, pitch, roll, pred_heatmap
        del loss, pose_loss, landmark_loss, graph_loss, Yaw_loss, Pitch_loss, Roll_loss, pred_graph, pred_edges

        torch.cuda.empty_cache()

    train_loss = np.mean(loss_train)
    train_pose_loss = np.mean(pose_loss_train)
    train_landmark_loss = np.mean(landmark_loss_train)
    train_heatmap_loss = np.mean(heatmap_loss_train)
    train_graph_loss = np.mean(graph_loss_train)

    return train_loss, train_pose_loss, train_landmark_loss, train_heatmap_loss, train_graph_loss, select_img, select_pred_graph, select_pred_landmark, select_pred_heatmap, select_gt_heatmap, select_gt_landmark

def evaluate(epoch, model, valid_data_loader, batch_size, is_best, args):
    
    #valid time
    model.eval()

    loss_fn = nn.L1Loss(reduction='sum').cuda()
    if args.hm_loss_type == "l1":
        loss_fn1 = nn.L1Loss(reduction='sum').cuda()
        loss_fn1_heatmap = nn.L1Loss().cuda()
    elif args.hm_loss_type == "mse":
        loss_fn1 = nn.MSELoss(reduction='sum').cuda()
        loss_fn1_heatmap = nn.MSELoss().cuda()

    loss_fn_graph = AdjacencyCriterion(num_landmark=args.num_landmark, desired_node_freedom=args.desired_node_freedom, dense_loss_weight=args.dense_loss_weight, sparse_loss_weight=args.sparse_loss_weight).cuda()

    loss_valid_landmark = []
    loss_valid_heatmap = []
    loss_valid_mae = []
    loss_valid_yaw = []
    loss_valid_pitch = []
    loss_valid_roll = []

    loss_valid_graph = []
    
    time_diff_list = []

    for idx_batch, batch in enumerate(valid_data_loader):

        img, gt_pose, gt_heatmap, gt_landmark, ori_landmark = batch
        # get data
        img = img.float().cuda(non_blocking=True)                 # (B, 3, 256, 256)
        gt_pose = gt_pose.float().cuda(non_blocking=True)         # (B, 3)
        gt_heatmap = gt_heatmap.float().cuda(non_blocking=True)   # (B, 256, 256, 68)
        gt_landmark = gt_landmark.float().cuda(non_blocking=True) # (B, 68, 2)
        ori_landmark = ori_landmark.float().cuda(non_blocking=True) # (B, 68, 2)

        # inference
        with torch.no_grad():

            start_time = time.time()

            pred_heatmap, pred_landmark, pred_pose, pred_edges, pred_graph = model(img)

            end_time = time.time()

            time_diff = (end_time - start_time) / batch_size

            time_diff_list.append(time_diff)

            valid_loss_yaw = loss_fn(pred_pose[:,0], gt_pose[:,0]) /batch_size 
            valid_loss_pitch = loss_fn(pred_pose[:,1], gt_pose[:,1]) /batch_size 
            valid_loss_roll = loss_fn(pred_pose[:,2], gt_pose[:,2]) /batch_size 
            
            valid_loss = loss_fn(pred_pose, gt_pose) /batch_size /3 

            
            landmark_loss = loss_fn1(pred_landmark, gt_landmark) /batch_size
            heatmap_loss = loss_fn1_heatmap(pred_heatmap, gt_heatmap)


            graph_loss = loss_fn_graph(pred_graph) / batch_size



        loss_valid_mae.append(valid_loss.item())
        loss_valid_landmark.append(landmark_loss.item())
        loss_valid_heatmap.append(heatmap_loss.item())

        loss_valid_graph.append(graph_loss.item())
        
        loss_valid_yaw.append(valid_loss_yaw.item())
        loss_valid_pitch.append(valid_loss_pitch.item())
        loss_valid_roll.append(valid_loss_roll.item())

        # msg ='[epoch:{}] [batch:{}] [valid_loss_total:{}] [landmark_loss:{}] [yaw_loss:{}] [pitch_loss:{}] [roll_loss:{}] [eval_cost:{}s/sample]'
        # if is_best and (idx_batch+1) % 30 == 0:

        #     # if (idx_batch+1) % 100 == 0 and (epoch+1) % 10 == 0:
        #     logging.info(msg.format(epoch+1, idx_batch+1, valid_loss.item(), landmark_loss.item(), valid_loss_yaw.item(), valid_loss_pitch.item(), valid_loss_roll.item(), time_diff))
        
        #     # # visualize pred_garph: (B, 1, 68, 68)
        #     # TO DO

        if idx_batch == len(valid_data_loader) - 1:
            select_img = img
            select_pred_graph = pred_graph
            select_pred_landmark = pred_landmark
            select_pred_heatmap = pred_heatmap
            select_gt_heatmap = gt_heatmap
            select_gt_landmark = gt_landmark

        del img, gt_pose, gt_heatmap, gt_landmark, pred_landmark, pred_pose, pred_graph, pred_edges, pred_heatmap
        del valid_loss_yaw, valid_loss_pitch, valid_loss_roll, valid_loss

        torch.cuda.empty_cache()

    eval_time = np.mean(time_diff_list)
    valid_loss_landmark = np.mean(loss_valid_landmark)
    valid_loss_heatmap = np.mean(loss_valid_heatmap)
    valid_loss_graph = np.mean(loss_valid_graph)
    valid_loss_mae = np.mean(loss_valid_mae)
    
    loss_yaw = np.mean(loss_valid_yaw)
    loss_pitch = np.mean(loss_valid_pitch)
    loss_roll = np.mean(loss_valid_roll)

    return valid_loss_mae, loss_yaw, loss_pitch, loss_roll, valid_loss_landmark, valid_loss_heatmap, valid_loss_graph, eval_time, select_img, select_pred_graph, select_pred_landmark, select_pred_heatmap, select_gt_heatmap, select_gt_landmark

def pipeline(train_dataset, valid_dataset, model, args):
    
    hyperparams = OrderedDict(
        lr = [args.learn_rate]
        ,batch_size = [args.batch_size]
    )
    
    epoch_num = args.epochs
    require_improvement = args.epochs
    gradient_clip = args.gradient_clip

    max_norm = 10

    data_loader = dict()

    if args.evaluate:
        pipeline_mode = 'test'
    else:
        pipeline_mode = 'train'

    
    # if pretrain:
    #     transfer_model('./pretrain/PoseDecoder/pretrain_model_PoseDecoder_2.67.pkl', model, device)
    # else:
    #     model.apply(weights_init)
        
    for run in RunBuilder.get_runs(hyperparams):

        # model = model.to(run.device)
        
        # load data 
        data_loader['train'] = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=run.batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True)
        data_loader['test'] = torch.utils.data.DataLoader(
                        dataset=valid_dataset,
                        batch_size=run.batch_size,
                        shuffle=False,
                        num_workers=0)

        params = filter(lambda p: p.requires_grad, model.parameters())
        
        if args.optim_type == 'SGD':
            optimizer = optim.SGD(
                        params,
                        lr=run.lr,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.0001)
            
        if args.optim_type == 'Adam':
            optimizer = optim.Adam(
                        params,
                        lr=run.lr,
                        weight_decay=0.0001)
        

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=4e-08)

        global total_iterations
        global best_validation_loss
        global last_improvement

        total_iterations = 0
        best_validation_loss = float('inf')
        last_improvement = 0

        if args.resume:

            checkpoint = torch.load(os.path.join(args.resume_path, 'model_best.pth.tar'))

            model.load_state_dict(checkpoint['state_dict'], strict=False)

            best_validation_loss = checkpoint['best_mse']

            optimizer.load_state_dict(checkpoint['optimizer'])

            epoch = checkpoint['epoch']

            logging.info('loaded last saved best checkpoint successfully!')
        
        # Start-time used for printing time-usage below.
        start_time = time.time()

        if pipeline_mode == 'train':

            logging.info('pipeline mode:{}, batch_size:{}, learning_rate:{}'.format(pipeline_mode, run.batch_size, run.lr))
            
            is_best = False

            for epoch in range(epoch_num):
                total_iterations += 1
                
                train_loss, train_pose_loss, train_landmark_loss, train_heatmap_loss, train_graph_loss, train_img, train_pred_graph, train_pred_landmark, train_pred_heatmap, train_gt_heatmap, \
                    train_gt_landmark = train(epoch, model, data_loader['train'], optimizer, run.batch_size, params, gradient_clip, max_norm, args, is_best)

                valid_loss_mae, loss_yaw, loss_pitch, loss_roll, valid_loss_landmark, valid_loss_heatmap, valid_loss_graph, eval_time, valid_img, valid_pred_graph, valid_pred_landmark, \
                    valid_pred_heatmap, valid_gt_heatmap, valid_gt_landmark = evaluate(epoch, model, data_loader['test'], run.batch_size, is_best, args)
                
#                 scheduler.step(valid_loss_mae)
                is_best = False

                if valid_loss_mae < best_validation_loss:
                    best_validation_loss = valid_loss_mae
                    best_train_loss = train_loss
                    best_loss_yaw = loss_yaw
                    best_loss_pitch = loss_pitch
                    best_loss_roll = loss_roll
                    last_improvement = total_iterations
                    improved_str = '*'

                    is_best = True

                    if args.vis_heatmap:
                        utils.plot_heatmap_graph(train_pred_graph, train_pred_landmark, train_pred_heatmap, train_img, train_gt_landmark, train_gt_heatmap, epoch, 'train', args)
                        utils.plot_heatmap_graph(valid_pred_graph, valid_pred_landmark, valid_pred_heatmap, valid_img, valid_gt_landmark, valid_gt_heatmap, epoch, 'valid', args)
                else:
                    improved_str = ''

                #save model 
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mse': best_validation_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.checkpoint_dir)
                    # torch.save(model.state_dict(), './pretrain/PoseDecoder/pretrain_model_PoseDecoder_'+str(int(round(start_time * 1000)))+'.pkl')

                msg ='[epoch:{}] [train_loss_total:{}] [train_pose_loss:{}] [train_landmark_loss:{}] [train_heatmap_loss:{}] [train_graph_loss:{}] [valid_loss_pose:{}] {} \n [yaw_loss:{}] [pitch_loss:{}] [roll_loss:{}] [valid_loss_landmark:{}] [valid_loss_heatmap:{}] [valid_loss_graph:{}] [eval_cost:{}s/sample]'
                if (epoch+1) % 1 == 0:
                    logging.info(msg.format(epoch+1, train_loss, train_pose_loss, train_landmark_loss, train_heatmap_loss, train_graph_loss, valid_loss_mae, improved_str, loss_yaw, loss_pitch, loss_roll, valid_loss_landmark, valid_loss_heatmap, valid_loss_graph, eval_time))

                if total_iterations - last_improvement > require_improvement:
                    logging.info("No improvement found in a while, stopping optimization.")

                    # Break out from the for-loop.
                    break

                del train_img, train_pred_graph, train_pred_landmark, train_gt_heatmap, train_gt_landmark
                del valid_img, valid_pred_graph, valid_pred_landmark, valid_gt_heatmap, valid_gt_landmark


        elif pipeline_mode == 'test':

            logging.info('pipeline mode:{}, batch_size:{}'.format(pipeline_mode, run.batch_size))

            best_validation_loss, best_loss_yaw, best_loss_pitch, best_loss_roll, valid_loss_heatmap, eval_time, valid_img, valid_pred_graph, valid_pred_landmark, \
                    valid_gt_heatmap, valid_gt_landmark = evaluate(0, model, data_loader['test'], run.batch_size, False, args)

            best_train_loss = 0.

            if args.vis_heatmap:
                utils.plot_heatmap_graph(valid_pred_graph, valid_pred_landmark, valid_img, valid_gt_landmark, valid_gt_heatmap, epoch, 'valid', args)
                 
        logging.info('=============================================')
        # Ending time.
        end_time = time.time()
        
    
        # Difference between start and end-times.
        time_dif = end_time - start_time

        if best_validation_loss == float('inf'):
            best_validation_loss = 'nan'
                        
       
    return best_validation_loss, best_train_loss, time_dif, best_loss_yaw, best_loss_pitch, best_loss_roll

