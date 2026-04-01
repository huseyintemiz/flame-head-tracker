import os, sys
WORKING_DIR = '/home/dipcik/phdprojects/flame-head-tracker'
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
    'video_path': './SUDFace/video/mov_subj1_NA',
    'original_fps': 60,
    'subsample_fps': 60,
    'save_path': './SUDFace/flame',
    'photometric_fitting': False,
    'realign': True,
    'batch_size': 32,
    'slim_save': True,  # False to save full npz with images/parsing
}

track_video(tracker_cfg)
print('Tracking complete!')
