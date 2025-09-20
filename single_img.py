## Enviroment Setup
import os, sys
WORKING_DIR = '/home/dipcik/avatar/flame-head-tracker-temiz'
os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
print("Current Working Directory: ", os.getcwd())

## Computing Device
device = 'cuda:0'
import torch
torch.cuda.set_device(device) # this will solve the problem that OpenGL not on the same device with torch tensors

import matplotlib.pyplot as plt
import numpy as np

from tracker_base import Tracker

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task', 
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'use_matting': True,           # use image/video matting to remove background
    'optimize_fov': True,          # whether to optimize the camera FOV, NOTE: this feature is still experimental
    'device': device,
}

tracker = Tracker(tracker_cfg)

img_path = './assets/FFHQ/00002.png'

# if realign == True, 
# img will be replaced by the realigned image
ret_dict = tracker.load_image_and_run(img_path, realign=True, photometric_fitting=False) 

# plot(ret_dict)

# check the shapes of returned results
for key in ret_dict:
    if key == 'fov': print('FOV = ', ret_dict[key])
    else: print(f'{key} {ret_dict[key].shape}')
