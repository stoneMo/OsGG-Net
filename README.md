
# OsGG-Net: One-step Graph Generation Network for Unbiased Head Pose Estimation**

[accepted to 2021 ACM MM]


![alt text](https://github.com/stoneMo/OsGG-Net/blob/main/imgs/title_image.png?raw=true)

## Abstract

Head pose estimation is a crucial problem that involves the prediction of the Euler angles of a human head in an image. Previous approaches predict head poses through landmarks detection, which can be applied to multiple downstream tasks. However, previous landmark-based methods can not achieve comparable performance to the current landmark-free methods due to lack of modeling the complex nonlinear relationships between the geometric distribution of landmarks and head poses. Another reason for the performance bottleneck is that there exists biased underlying distribution of the 3D pose angles in the current head pose benchmarks. In this work, we propose **OsGG-Net**, a One-step Graph Generation Network for estimating head poses from a single image by generating a landmark-connection graph to model the 3D angle associated with the landmark distribution robustly. To further ease the angle-biased issues caused by the biased data distribution in learning the graph structure, we propose the UnBiased Head Pose Dataset, called UBHPD, and a new unbiased metric, namely UBMAE, for unbiased head pose estimation. We conduct extensive experiments on various benchmarks and UBHPD where our method achieves the state-of-the-art results in terms of the commonly-used MAE metric and our proposed UBMAE. Comprehensive ablation studies also demonstrate the effectiveness of each part in our approach.

## Requirements


To install requirements, you can:

```
pip install -r requirements.txt

```

Note that our implementation is based on Python 3.7, and PyTorch deep learning framework, trained on NVIDIA Titan RTX GPU in Ubuntu 16.04 system.

## Codes

There are three different section of this project. 
1. Data pre-processing
2. Training and testing 

We will go through the details in the following sections.

### 1. Data pre-processing

In this work, we demonstrate the effective of the proposed OsGG-Net for head pose estimation on BIWI Kinect Head Pose Database. The BIWI dataset contains 24 videos of 20 subjects in the controlled environment. There are a total of roughly 15, 000 frames in the dataset. The 300W across Large Poses (300W-LP) dataset is synthesized with expanding 558 61,225 samples across large poses in the 300W dataset with flipping to 122,450 samples. The AFLW2000 dataset provides 560 ground-truth 3D faces and the corresponding 68 landmarks for the first 2, 000 images of the AFLW dataset, where the faces in the dataset have large pose variations with various illumination conditions and expressions. 

```

# Running on BIWI dataset

cd data_preprocessing
python create_db_biwi.py

# Running on AFLW dataset

cd data_preprocessing
bash run_created_db_300W_AFLW.bash 

```

#### Download the datasets

+ [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
+ [300W-LP+AFLW](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
+ [UBHPD](https://drive.google.com/drive/folders/1A6mraNo3cZBy4Xps9HdxMXnshbG1tmzr?usp=sharing)


### 2. Training and testing 
```

# Running on BIWI dataset

bash run_BIWI.bash


# Running on AFLW dataset

bash run_AFLW.bash


# Running on UBHPD dataset

bash run_UBHPD.bash

```

Just remember to check which dataset type you want to use in the shell and you are good to go. Note that we calculate the MAE of yaw, pitch, roll independently, and average them into one single MAE for evaluation.


## Cite

```

@inproceedings{mo2021osggnet,
  title={OsGG-Net: One-step Graph Generation Network for Unbiased Head Pose Estimation},
  author={Shentong Mo and Miao Xin},
  booktitle={Proceedings of the 29th ACM Int'l Conference on Multimedia},
  year={2021}
}

```