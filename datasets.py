import os
import numpy as np
import cv2
import pandas as pd
import logging
import time

import torch
from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image
# from torchvision import transforms
from matplotlib import pyplot as plt

from PIL import Image, ImageFilter
import face_alignment

from utils import *

# logging.basicConfig(level=logging.DEBUG)
def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"], d["landmark"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

def gen_landmarks(img):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    key_points = fa.get_landmarks(img)
    if isinstance(key_points, list):
        if len(key_points) > 1:
            face_id = 0
            face_point = 0
            for i, points in enumerate(key_points):
                point = points[0][0]
                if point > face_point:
                    face_point = point
                    face_id = i
            key_point = key_points[face_id]
        else:
            key_point = key_points[0]

        if len(key_points) > 0:
            landmarks = key_point
        else:
            landmarks = None
	
    return landmarks

def gen_gaussian_heatmaps(img, landmarks, down_ratio, num_points=68):

	img_h, img_w = img.shape[:2]
	landmarks = landmarks / img_h

	map_height = img_h//down_ratio
	map_width = img_w//down_ratio
	heatmap=np.zeros((map_height, map_width, num_points),dtype=np.float)
	assert(len(landmarks)==num_points)
	for p in range(len(landmarks)):
		x=landmarks[p][0]*map_width
		y=landmarks[p][1]*map_height
		for i in range(map_width):
			for j in range(map_height):
				if (x-i)*(x-i)+(y-j)*(y-j)<=4:
					# print(1.0/(1+(x-i)*(x-i)*2+(y-j)*(y-j)*2))
					heatmap[j][i][p]=1.0/(1+(x-i)*(x-i)*2+(y-j)*(y-j)*2)

	return heatmap

def visual_landmarks(image, landmarks):

    for i in range(len(landmarks)):
        x = int((landmarks[i][0] + 0.5) * image.shape[1])
        y = int((landmarks[i][1] + 0.5) * image.shape[0])
        # print(x,y)
        image = cv2.circle(image, (x,y), radius=1, color=(0, 0, 255), thickness=1)
    return image

def gaussian(img, pt):
    sigma = 0.5
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
        br[0] < 0 or br[1] < 0):
    # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def gen_landmark_label(patch_width, patch_height, joints):

    # print(patch_height)
    # print(patch_width)

    norm_joints = np.zeros_like(joints)

    norm_joints[:, 0] = joints[:, 0] / patch_width - 0.5
    norm_joints[:, 1] = joints[:, 1] / patch_height - 0.5

    # joints = joints.reshape((-1))
    # joints_vis = joints_vis.reshape((-1))
    return norm_joints

def load_data(dataset_name, data_dir):

    if dataset_name == '300W_LP':
        db_list = ['AFW.npz','AFW_Flip.npz','HELEN.npz','HELEN_Flip.npz','IBUG.npz','IBUG_Flip.npz','LFPW.npz','LFPW_Flip.npz']
        image = []
        pose = []
        landmark = []
        for i in range(0,len(db_list)):
            image_temp, pose_temp, landmark_temp = load_data_npz(os.path.join(data_dir, db_list[i]))
            image.append(image_temp)
            pose.append(pose_temp)
            landmark.append(landmark_temp)

        image = np.concatenate(image,0)
        pose = np.concatenate(pose,0)
        landmark = np.concatenate(landmark,0)
        
        # we only care the angle between [-99,99] and filter other angles
        img_data = []
        headpose_data = []
        landmark_data = []
        print(image.shape)
        print(pose.shape)
        print(landmark.shape)

        for i in range(0, pose.shape[0]):
            temp_pose = pose[i,:]
            if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                img_data.append(image[i,:,:,:])
                headpose_data.append(pose[i,:])
                landmark_data.append(landmark[i,:,:])
        img_data = np.array(img_data)
        headpose_data = np.array(headpose_data)
        landmark_data = np.array(landmark_data)
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    elif dataset_name == 'synhead_noBIWI':
        img_data, headpose_data, landmark_data = load_data_npz('../data/synhead/media/jinweig/Data2/synhead2_release/synhead_noBIWI.npz')
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    
    elif dataset_name == 'AFLW2000':
        img_data, headpose_data, landmark_data = load_data_npz(os.path.join(data_dir, 'AFLW2000.npz'))
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    elif dataset_name == 'BIWI':
        img_data, headpose_data, landmark_data = load_data_npz(os.path.join(data_dir, 'BIWI_noTrack.npz'))
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    elif dataset_name == 'BIWI_train':
        img_data, headpose_data, landmark_data = load_data_npz(os.path.join(data_dir, 'BIWI_train.npz'))
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    elif dataset_name == 'BIWI_test':
        img_data, headpose_data, landmark_data = load_data_npz(os.path.join(data_dir, 'BIWI_test.npz'))
        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        return img_data, headpose_data, landmark_data

    else:
        print('dataset_name is wrong!!!')
        return

class OsGGData(Dataset):
    def __init__(self, dataset_name, data_dir, num_landmark, transform=None, train=True):
        
        
        logging.info("Loading data...")

        start_time = time.time()

        img_data, headpose_data, landmark_data = load_data(dataset_name, data_dir)

        print(img_data.shape)
        print(headpose_data.shape)
        print(landmark_data.shape)

        self.img_data = img_data
        self.headpose_data = headpose_data
        self.landmark_data = landmark_data

        self.transform = transform
        self.train = train

        self.down_ratio = 4
        self.crop_dim = 256

        self.num_landmark = num_landmark

        if self.num_landmark == 19:
            self.landmarks_list = [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54]
        elif self.num_landmark == 68:
            self.landmarks_list = range(self.num_landmark)

        end_time = time.time()

        logging.info("time for initializing dataset: {}".format(end_time-start_time))
    
    def make_gaussian(self, landmarks, pt, down_ratio):
        (h,w) = pt
        masks = np.zeros((self.num_landmark, h//down_ratio, w//down_ratio), dtype=np.float32)

        for idx in range(self.num_landmark):
            # if int(anno[idx+1][2]) == 1:
            masks[idx] = gaussian(masks[idx], (int(round(landmarks[idx][0]/down_ratio)), int(round(landmarks[idx][1]/down_ratio))))

        return masks

    def __getitem__(self, index):


        # start_time = time.time()

        img = self.img_data[index]
        # ori_img = Image.fromarray(ori_img).resize((self.crop_dim, self.crop_dim))
        # ori_img = np.asarray(ori_img)

        # print("ori_img:", ori_img.shape)

        # gt_landmark = gen_landmarks(ori_img)
        # while gt_landmark == None:
        #     index = index + 1
        # ori_img = self.img_data[index]

        # time1 = time.time()
        # logging.info("time for loading img: {}".format(time1-start_time))

        # if self.train:
        #     img = augment_data(ori_img)
        # else:
        #     img = ori_img

        gt_pose = self.headpose_data[index]

        # time2 = time.time()
        # logging.info("time for getting pose: {}".format(time2-time1))

        gt_landmark = self.landmark_data[index]

        # time3 = time.time()
        # logging.info("time for getting landmark: {}".format(time3-time2))

        gt_landmark = gt_landmark[np.array(self.landmarks_list),:]

        # print("gt_landmark:", gt_landmark[:,0])

        # normalization
        target_landmark = gen_landmark_label(img.shape[1], img.shape[0], gt_landmark)

        gt_heatmap = self.make_gaussian(gt_landmark, img.shape[:2], self.down_ratio)         # (68, 256, 256)
        # gt_heatmap = gen_gaussian_heatmaps(ori_img, gt_landmark, self.down_ratio)
        # print("gt_landmark:", gt_landmark[:,0])
        # time4 = time.time()
        # logging.info("time for getting heatmap: {}".format(time4-time3))

        # print("gt_heatmap_before:", gt_heatmap.shape)

        if self.transform is not None:
            (img, gt_heatmap) = self.transform((img, gt_heatmap.transpose(1,2,0)))

        # print("gt_heatmap_after:", gt_heatmap.shape)          # (68, 256, 256)
        # print("gt_landmark:", gt_landmark[:,0])
 
        ori_landmark = gt_landmark

        # print("ori_landmark:", ori_landmark.shape)

        # print("img:", img.shape)


        # time5 = time.time()
        # logging.info("time for transformation: {}".format(time5-time4))
        
        return img, gt_pose, gt_heatmap.transpose(2,0,1), target_landmark, ori_landmark

    def __len__(self):
        return len(self.img_data)


if __name__ == "__main__":

    import transforms as transforms
    import utils

    # transforms 
    train_transformations = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_dir = './data'
    dataset_name = 'AFLW2000'

    isPlot = True

    num_landmark = 19

    # dataset & dataloader
    train_dataset = OsGGData(dataset_name=dataset_name, data_dir=data_dir, num_landmark=num_landmark, transform=train_transformations, train=False)

    print("train_dataset:", len(train_dataset))

    for i in range(len(train_dataset)):

        if i == 2:

            img, gt_pose, gt_heatmap, target_landmark, ori_landmark = train_dataset[i]

            print("img:", img.shape)
            print("gt_pose:", gt_pose.shape)
            print("gt_heatmap:", gt_heatmap.shape)        # (num_landmark, 64, 64)
            # print("vis_landmark:", vis_landmark[:,0])
            print("target_landmark:", target_landmark[:,0])
            print("ori_landmark:", ori_landmark[:,0])

            img = utils.tensor2im(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) 
            # img = cv2.utils.dumpInputArray(img)

            if isPlot:
                # cv2.imshow('check',img)
                # k=cv2.waitKey(500)

                # img = img.permute(1,2,0).numpy()
                print(type(img))
                print(img.shape)
                # img = Image.fromarray(img*255.0)

                cv2.imwrite(str(i)+'.jpg', img)

                img_landmarks = visual_landmarks(img, target_landmark)

                cv2.imwrite(str(i)+'_landmarks_'+str(num_landmark)+'.jpg', img_landmarks)
                
                heatmap_img=np.zeros((64,64),dtype=np.float)
                for index in range(num_landmark):
                    heatmap_img+=gt_heatmap[index,:,:]*255.0
                # print(heatmap_img)

                Image.fromarray(heatmap_img).convert('RGB').save('{}_heatmaps_'.format(i)+str(num_landmark)+'.jpg')

                print("img:", img.shape)



    
