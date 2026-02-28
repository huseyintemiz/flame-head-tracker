import os, sys, time
WORKING_DIR = '/home/dipcik/phdprojects/flame-head-tracker'
os.chdir(WORKING_DIR)

device = 'cuda:0'
import torch
torch.cuda.set_device(device)

# ─── GPU/CPU diagnostic ───
print("=" * 70)
print("DEVICE DIAGNOSTICS")
print("=" * 70)
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device      : {torch.cuda.get_device_name(device)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device   : {torch.cuda.current_device()}")
    mem = torch.cuda.get_device_properties(device)
    print(f"GPU memory       : {mem.total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available, running on CPU!")
print("=" * 70)

# ─── timing helpers ───
_timers = {}

def tic(label):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    _timers[label] = time.perf_counter()

def toc(label):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - _timers[label]
    print(f"[TIME] {label:.<55s} {elapsed:8.3f}s")
    return elapsed

# ─── start total timer ───
tic("TOTAL")
tic("Imports & setup")

from tracker_base import Tracker
from tracker_video import _generate_compare_image, _generate_output_video

from utils.video_utils import video_to_images
from utils.matting_utils import matting_video_frames
from utils.general_utils import check_nan_in_dict
import numpy as np
import cv2
from tqdm import tqdm

toc("Imports & setup")

# ─── config ───
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
    'video_path': './get_video/subclip_31s_part2.mp4',
    'original_fps': 25,
    'subsample_fps': 25,
    'save_path': './get_video/tracking_results_timed',
    'photometric_fitting': False,
    'realign': True,
    'batch_size': 8,
}

video_path = tracker_cfg['video_path']
save_path = tracker_cfg['save_path']
photometric_fitting = tracker_cfg['photometric_fitting']
realign = tracker_cfg['realign']
batch_size = max(8, tracker_cfg['batch_size'])
output_fps = tracker_cfg.get('output_fps', tracker_cfg.get('subsample_fps', 25))

video_base_name = os.path.basename(video_path)
video_name = video_base_name.split('.')[0]
result_save_path = os.path.join(save_path, video_name)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load video frames
# ═══════════════════════════════════════════════════════════════
tic("1. Video loading (video_to_images)")
frames = video_to_images(
    video_path=tracker_cfg['video_path'],
    original_fps=tracker_cfg['original_fps'],
    subsample_fps=tracker_cfg['subsample_fps'],
)
t_video_load = toc("1. Video loading (video_to_images)")
print(f"   Loaded {len(frames)} frames, shape={frames[0].shape}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Initialize Tracker (loads all models)
# ═══════════════════════════════════════════════════════════════
tic("2. Tracker initialization (all models)")
tracker = Tracker(tracker_cfg)
tracker.set_landmark_detector('mediapipe')
t_tracker_init = toc("2. Tracker initialization (all models)")

# ─── check which models are on GPU ───
print("\n" + "=" * 70)
print("MODEL DEVICE CHECK")
print("=" * 70)

def check_model_device(name, model):
    try:
        p = next(model.parameters())
        print(f"  {name:.<45s} {p.device}")
    except StopIteration:
        print(f"  {name:.<45s} (no parameters)")
    except AttributeError:
        print(f"  {name:.<45s} (not a nn.Module)")

check_model_device("FLAME model", tracker.flame)
check_model_device("FLAME texture model", tracker.flametex)
check_model_device("FLAME texture renderer", tracker.flame_texture_render)
check_model_device("Ear landmarker", tracker.ear_landmarker)
check_model_device("DECA model", tracker.deca)
check_model_device("MICA model", tracker.mica)
check_model_device("Video matting model", tracker.video_matting_model)
print(f"  {'Face parser (BiSeNet)':.<45s}", end=" ")
try:
    p = next(tracker.face_parser.model.parameters())
    print(p.device)
except:
    print("unknown")
print(f"  {'Mediapipe face landmarker':.<45s} CPU (always)")
print(f"  {'FAN face alignment':.<45s} CPU (always)")
print("=" * 70 + "\n")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Video matting
# ═══════════════════════════════════════════════════════════════
if tracker.use_matting:
    tic("3. Video matting (all frames)")
    frames = matting_video_frames(tracker.video_matting_model, frames)
    t_matting = toc("3. Video matting (all frames)")
    print(f"   {len(frames)} frames matted, avg={t_matting/len(frames):.4f}s/frame")
else:
    t_matting = 0.0
    print("[TIME] 3. Video matting............................... SKIPPED")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Canonical shape estimation
# ═══════════════════════════════════════════════════════════════
shape_code = None
texture = None

tic("4. Canonical shape estimation (total)")
if not photometric_fitting:
    MAX_SAMPLE_SIZE = 3
    frames_subset = frames[:MAX_SAMPLE_SIZE]

    t_per_sample = []
    with torch.no_grad():
        mean_shape_code = np.zeros([1, tracker.NUM_SHAPE_COEFFICIENTS], dtype=np.float32)
        counter = 0
        for i in range(len(frames_subset)):
            tic(f"   4a. prepare_intermediate_data frame {i}")
            img = frames_subset[i]
            in_dict = tracker.prepare_intermediate_data_from_image(img, realign=realign)
            t_s = toc(f"   4a. prepare_intermediate_data frame {i}")
            t_per_sample.append(t_s)
            if in_dict is not None:
                mean_shape_code += in_dict['shape']
                counter += 1
        if counter > 0:
            mean_shape_code /= counter
    shape_code = mean_shape_code
else:
    frames_subset = [frames[i] for i in np.linspace(0, len(frames) - 1, batch_size, dtype=int)]
    tic("   4a. photometric canonical estimation")
    batch_ret_dict, batch_valid_indices = tracker.run(
        img=frames_subset, realign=realign,
        photometric_fitting=photometric_fitting,
        shape_code=None, texture=None,
        temporal_smoothing=False, estimate_canonical=True
    )
    toc("   4a. photometric canonical estimation")
    if 'texture' in batch_ret_dict.keys():
        texture_save_path = os.path.join(result_save_path, 'texture.png')
        if not os.path.exists(texture_save_path):
            img_texture = (np.transpose(batch_ret_dict['texture'][0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(texture_save_path, cv2.cvtColor(img_texture, cv2.COLOR_RGB2BGR))
        texture = np.copy(batch_ret_dict['texture'][:1])
        del batch_ret_dict['texture']
    shape_code = batch_ret_dict['shape'][:1]

t_canon = toc("4. Canonical shape estimation (total)")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Track all frames (main loop) — with per-batch breakdown
# ═══════════════════════════════════════════════════════════════
print(f"\n>>> Tracking video: {video_path}")
print(f"    Total frames: {len(frames)}, batch_size: {batch_size}")
print("=" * 70)

tic("5. Track all frames (total)")

i = 0
total_steps = len(frames)
batch_count = 0

# accumulators for per-step timing
t_tracker_run_total = 0.0
t_save_npz_total = 0.0
t_compare_img_total = 0.0
t_batch_times = []

pbar = tqdm(total=total_steps, desc="Tracking")
while i < total_steps:
    tic(f"batch_{batch_count}")

    batch_frames = frames[i:i + batch_size]

    # ── tracker.run (prepare_intermediate + fitting) ──
    tic("   5a. tracker.run (batch)")
    batch_ret_dict, batch_valid_indices = tracker.run(
        img=batch_frames, realign=realign,
        photometric_fitting=photometric_fitting,
        shape_code=shape_code, texture=texture,
        temporal_smoothing=True,
    )
    t_run = toc("   5a. tracker.run (batch)")
    t_tracker_run_total += t_run

    if 'texture' in batch_ret_dict.keys():
        del batch_ret_dict['texture']

    # ── save results + compare images ──
    tic("   5b. Save npz + compare images (batch)")
    for j, idx in enumerate(batch_valid_indices):
        fid = i + idx

        ret_dict = {}
        for key in batch_ret_dict:
            ret_dict[key] = batch_ret_dict[key][j:j + 1]

        tic("      save_npz")
        save_file_path = os.path.join(result_save_path, f'{fid}.npz')
        np.savez_compressed(save_file_path, **ret_dict)
        t_save_npz_total += toc("      save_npz")

        tic("      compare_img")
        with torch.no_grad():
            loaded_params = np.load(save_file_path)
            result_img = _generate_compare_image(loaded_params, tracker=tracker, ret_dict=ret_dict)
            cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)
        t_compare_img_total += toc("      compare_img")

    t_batch = toc(f"batch_{batch_count}")
    t_batch_times.append((i, len(batch_frames), t_batch))

    i += batch_size
    batch_count += 1
    pbar.update(batch_size)

pbar.close()
t_track_all = toc("5. Track all frames (total)")

# ═══════════════════════════════════════════════════════════════
# STEP 6: Generate output video
# ═══════════════════════════════════════════════════════════════
tic("6. Generate output video (ffmpeg)")
_generate_output_video(result_save_path, video_name, output_fps, video_path=video_path)
t_video_gen = toc("6. Generate output video (ffmpeg)")

# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════
t_total = toc("TOTAL")

print("\n")
print("=" * 70)
print("RUNTIME SUMMARY")
print("=" * 70)
print(f"  Total frames processed     : {len(frames)}")
print(f"  Batch size                 : {batch_size}")
print(f"  Number of batches          : {batch_count}")
print(f"  Photometric fitting        : {photometric_fitting}")
print(f"  Device                     : {device}")
print("-" * 70)
print(f"  1. Video loading           : {t_video_load:8.3f}s")
print(f"  2. Tracker initialization  : {t_tracker_init:8.3f}s")
print(f"  3. Video matting           : {t_matting:8.3f}s  ({t_matting/max(len(frames),1):.4f}s/frame)")
print(f"  4. Canonical shape estim.  : {t_canon:8.3f}s")
print(f"  5. Track all frames        : {t_track_all:8.3f}s")
print(f"     ├── tracker.run total   : {t_tracker_run_total:8.3f}s  ({t_tracker_run_total/max(batch_count,1):.3f}s/batch)")
print(f"     ├── save .npz total     : {t_save_npz_total:8.3f}s")
print(f"     └── compare images total: {t_compare_img_total:8.3f}s")
print(f"  6. Output video (ffmpeg)   : {t_video_gen:8.3f}s")
print("-" * 70)
print(f"  TOTAL                      : {t_total:8.3f}s  ({t_total/60:.1f} min)")
print(f"  Avg time per frame (track) : {t_track_all/max(len(frames),1):.4f}s")
print(f"  Effective FPS (track only) : {len(frames)/max(t_track_all,0.001):.2f}")
print("=" * 70)

# ─── per-batch breakdown ───
print("\nPER-BATCH TIMING:")
print(f"  {'Batch':>5s}  {'Start':>6s}  {'Frames':>6s}  {'Time(s)':>8s}  {'FPS':>6s}")
for start_idx, n_frames, bt in t_batch_times:
    fps = n_frames / bt if bt > 0 else float('inf')
    print(f"  {start_idx//batch_size:>5d}  {start_idx:>6d}  {n_frames:>6d}  {bt:>8.3f}  {fps:>6.2f}")

# ─── GPU memory usage ───
if torch.cuda.is_available():
    print(f"\nGPU MEMORY:")
    print(f"  Peak allocated : {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    print(f"  Peak reserved  : {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")
    print(f"  Current alloc  : {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

print("\nTracking complete!")
