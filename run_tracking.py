import os, sys
WORKING_DIR = '/home/dipcik/avatar/flame-head-tracker-temiz'
os.chdir(WORKING_DIR)

device = 'cuda:0'
import torch
torch.cuda.set_device(device)

from tracker_video import track_video

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker_v2_with_blendshapes.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth',
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'use_matting': True,
    'optimize_fov': True,
    'device': device,
    # Video-specific settings
    'video_path': './get_video/subclip_40s.mp4',
    'original_fps': 25,
    'subsample_fps': 25,
    'save_path': './get_video/tracking_results',
    'photometric_fitting': False,
    'realign': True,
    'batch_size': 8,
}

track_video(tracker_cfg)
print('Tracking complete!')
