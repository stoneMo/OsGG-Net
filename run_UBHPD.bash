#!/bin/bash

python main.py --gpu_id 0 \
               --epochs 300 \
               --num_landmark 19 \
               --visual_threshold 0.15 \
               --adj_threshold 0 \
               --landmark_loss_weight 10 \
               --heatmap_loss_weight 0 \
               --graph_loss_weight 10 \
               --desired_node_freedom 1\
               --sparse_loss_weight 1 \
               --dense_loss_weight 10 \
               --batch_size 32 \
               --learn_rate 1e-3 \
               --train_dataset UBHPD_train \
               --valid_dataset UBHPD_test \
               --data_dir ./data \
               --hm_loss_type mse \
               --yaw_loss_weight 0.01 \
               --pitch_loss_weight 0.015 \
               --roll_loss_weight 0.01  \
               --heatmap_dir ./heatmap \
               --vis_heatmap
