import os, sys, glob, time, shutil, subprocess
WORKING_DIR = '/home/dipcik/phdprojects/flame-head-tracker'
os.chdir(WORKING_DIR)

device = 'cuda:0'
import torch
torch.cuda.set_device(device)

from tracker_video import track_video

VIDEO_DIR = './SUDFace/video_full'
SAVE_PATH = './SUDFace/flame'


def get_expected_frames(video_path):
    """Get expected frame count from video file."""
    out = subprocess.check_output([
        'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_path
    ]).decode().strip()
    return int(out) if out else 0


def is_complete(result_dir, video_path):
    """Check if tracking results are complete."""
    if not os.path.exists(result_dir):
        return False
    npz_count = len([f for f in os.listdir(result_dir) if f.endswith('.npz')])
    expected = get_expected_frames(video_path)
    return npz_count >= expected


# Get all videos sorted numerically by subject number
videos = sorted(glob.glob(os.path.join(VIDEO_DIR, 'mov_subj*')),
                key=lambda x: (int(os.path.basename(x).split('_')[1].replace('subj', '')),
                               os.path.basename(x).split('_')[2]))

# Filter subjects 1 to 50
videos = [v for v in videos
          if int(os.path.basename(v).split('_')[1].replace('subj', '')) <= 50]

print(f'Found {len(videos)} videos (subj1-50)')

# Skip completed, remove incomplete
todo = []
for i, v in enumerate(videos):
    video_name = os.path.basename(v).split('.')[0]
    result_dir = os.path.join(SAVE_PATH, video_name)
    if is_complete(result_dir, v):
        print(f'SKIP {video_name} (complete)')
    else:
        if os.path.exists(result_dir):
            print(f'REMOVING incomplete {video_name}')
            shutil.rmtree(result_dir)
        todo.append((v, i + 1))

print(f'\nTotal: {len(videos)} videos, {len(videos) - len(todo)} done, {len(todo)} remaining\n')

for video_path, idx in todo:
    video_name = os.path.basename(video_path).split('.')[0]
    print(f'[{idx}/{len(videos)}] Processing {video_name}...')
    t0 = time.time()

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
        'video_path': video_path,
        'original_fps': 60,
        'subsample_fps': 60,
        'save_path': SAVE_PATH,
        'photometric_fitting': False,
        'realign': True,
        'batch_size': 64,
        'slim_save': True,
    }

    try:
        track_video(tracker_cfg)
        elapsed = time.time() - t0
        print(f'[{idx}/{len(videos)}] DONE {video_name} ({elapsed:.1f}s)')
    except Exception as e:
        print(f'[{idx}/{len(videos)}] FAILED {video_name}: {e}')

print('All done!')
