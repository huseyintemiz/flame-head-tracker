# FLAME Head Tracker — Video Pipeline

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Video File (.mp4)                         │
│              original_fps=25  subsample_fps=25  batch_size=32       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 1: VIDEO LOADING & PREPROCESSING                 │
│                                                                     │
│  ┌──────────────────┐        ┌──────────────────────────────┐      │
│  │ video_to_images() │        │  matting_video_frames()      │      │
│  │                   │        │  (RobustVideoMatting)        │      │
│  │  OpenCV decode    │───────▶│                              │      │
│  │  BGR → RGB        │        │  Recurrent state across      │      │
│  │  Subsample FPS    │        │  frames for temporal         │      │
│  │                   │        │  consistency                 │      │
│  │  → List[H,W,3]   │        │  → List[H,W,3] (white bg)   │      │
│  └──────────────────┘        └──────────────┬───────────────┘      │
└─────────────────────────────────────────────┼───────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│          STAGE 2: CANONICAL SHAPE & TEXTURE ESTIMATION              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  if photometric_fitting=False (Landmark-Only):              │   │
│  │                                                             │   │
│  │    Sample 3 frames → MICA encoder → shape [1,300] each     │   │
│  │    Average → mean_shape_code [1,300]                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  if photometric_fitting=True (Photometric):                 │   │
│  │                                                             │   │
│  │    Sample batch_size frames uniformly across video          │   │
│  │    → tracker.run(photometric=True, estimate_canonical=True) │   │
│  │    → Average shape [1,300]                                  │   │
│  │    → Extract texture [1,3,256,256] from first valid frame   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output: shape_code [1,300]  (+texture [1,3,256,256] if photo)     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 3: BATCH VIDEO TRACKING LOOP                     │
│                                                                     │
│  for i in range(0, total_frames, batch_size):                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  batch_frames = frames[i : i+batch_size]                    │   │
│  │                         │                                   │   │
│  │                         ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │        PER-FRAME PREPROCESSING (parallel)            │   │   │
│  │  │                                                      │   │   │
│  │  │  For each frame in batch:                            │   │   │
│  │  │    ├─ MediaPipe → 478 lmks + 68 lmks + blendshapes  │   │   │
│  │  │    ├─ Image Align → [256,256,3]                      │   │   │
│  │  │    ├─ Face Parsing → [512,512]                       │   │   │
│  │  │    ├─ Ear Detection → 20 lmks/ear (optional)        │   │   │
│  │  │    └─ DECA → initial exp, pose, tex, light           │   │   │
│  │  └──────────────────────┬───────────────────────────────┘   │   │
│  │                         │                                   │   │
│  │                         ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │        BATCH OPTIMIZATION ON GPU                     │   │   │
│  │  │                                                      │   │   │
│  │  │  Step 1: Rigid Camera Pose (1500 iters)              │   │   │
│  │  │    ├─ FLAME forward [N,5023,3]                       │   │   │
│  │  │    ├─ Extract lmks [N,108,3]                         │   │   │
│  │  │    ├─ Perspective projection → NDC                   │   │   │
│  │  │    └─ L2 loss vs GT landmarks                        │   │   │
│  │  │                                                      │   │   │
│  │  │  Step 2: Fine-Tuning                                 │   │   │
│  │  │    ├─ Landmark: 200 iters (exp+jaw+eyes)             │   │   │
│  │  │    └─ Photometric: 400 iters (+shape+tex+light)      │   │   │
│  │  │                                                      │   │   │
│  │  │  temporal_smoothing=True:                            │   │   │
│  │  │    └─ ||exp[t] - exp[t-1]||² regularization          │   │   │
│  │  └──────────────────────┬───────────────────────────────┘   │   │
│  │                         │                                   │   │
│  │                         ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │        PER-FRAME SAVE & VISUALIZE                    │   │   │
│  │  │                                                      │   │   │
│  │  │  For each valid frame j in batch:                    │   │   │
│  │  │    ├─ Save {fid}.npz (all FLAME params)              │   │   │
│  │  │    └─ Save {fid}_compare.jpg (5-panel image)         │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STAGE 4: VIDEO COMPOSITION                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Collect all *_compare.jpg → sorted by frame number         │   │
│  │                                                              │   │
│  │  ffmpeg -framerate 25                                        │   │
│  │    -i {path}/%d_compare.jpg     (compare frames)             │   │
│  │    -i {video_path}              (source audio)               │   │
│  │    -c:v libx264 -c:a aac                                    │   │
│  │    -shortest                                                 │   │
│  │    → {video_name}_compare.mp4                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT FILES                                │
│                                                                     │
│  output/{video_name}/                                               │
│    ├── 0.npz              ← FLAME params frame 0                   │
│    ├── 0_compare.jpg      ← 5-panel visualization frame 0          │
│    ├── 1.npz                                                       │
│    ├── 1_compare.jpg                                                │
│    ├── ...                                                          │
│    ├── N.npz                                                        │
│    ├── N_compare.jpg                                                │
│    └── {video_name}_compare.mp4  ← final comparison video w/ audio │
└─────────────────────────────────────────────────────────────────────┘
```

## 5-Panel Comparison Image Detail

```
┌─────────────────────────────────────────────────────────────────────┐
│                    {fid}_compare.jpg [256 × 1280]                   │
│                                                                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│  │          │ ● 68 face│          │          │          │          │
│  │ Original │ ● 10 eye │ 40% img  │ Textured │ Frontal  │          │
│  │  Image   │ ● 20+20  │ 60% mesh │  FLAME   │  View    │          │
│  │          │   ear    │ blended  │  Mesh    │ pose=0   │          │
│  │ [256²]   │ [256²]   │ [256²]   │ [256²]   │ [256²]   │          │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘          │
│   Panel 1    Panel 2    Panel 3    Panel 4    Panel 5              │
└─────────────────────────────────────────────────────────────────────┘
```

## NPZ File Contents (per frame)

```
{fid}.npz
  ├── shape           [1, 300]         # identity shape coefficients
  ├── exp             [1, 100]         # expression coefficients
  ├── head_pose       [1, 3]           # head rotation (yaw, pitch, roll)
  ├── jaw_pose        [1, 3]           # jaw opening rotation
  ├── eye_pose        [1, 6]           # eye ball rotation (L+R)
  ├── tex             [1, 50]          # texture coefficients
  ├── light           [1, 9, 3]        # SH lighting (9 bands × RGB)
  ├── texture         [1, 3, 256, 256] # texture map (photometric only)
  ├── cam             [1, 6]           # camera pose (rot3 + trans3)
  ├── fov             [1]              # field of view (degrees)
  ├── K               [1, 3, 3]        # camera intrinsics matrix
  ├── img_resized     [1, 256, 256, 3] # input image (resized)
  ├── parsing         [1, 512, 512]    # face parsing mask
  ├── lmks_68         [1, 68, 3]       # 2D face landmarks
  ├── lmks_eyes       [1, 10, 3]       # 2D eye landmarks
  ├── lmks_ears       [1, 40, 3]       # 2D ear landmarks (optional)
  └── blendshape_scores [1, 52]        # FACs blendshapes
```

## Batch Processing Flow

```
Total Frames: N (e.g. 3944 for chunk_012)
Batch Size: 32

Batch 0:  frames[  0: 32]  → GPU optimize → save 0.npz..31.npz
Batch 1:  frames[ 32: 64]  → GPU optimize → save 32.npz..63.npz
Batch 2:  frames[ 64: 96]  → GPU optimize → save 64.npz..95.npz
  ...
Batch K:  frames[K*32: N]  → GPU optimize → save remaining.npz

Each batch:
  ┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────┐
  │ Preproc │──▶│ Rigid Fit   │──▶│ Fine-Tune   │──▶│ Save │
  │ 32 imgs │   │ 1500 iters  │   │ 200/400 it  │   │ .npz │
  │ CPU+GPU │   │ GPU         │   │ GPU         │   │ .jpg │
  └─────────┘   └─────────────┘   └─────────────┘   └──────┘
```

## Recovery Mode (Existing Results)

```
If output/{video_name}/ already has .npz files:

  ┌────────────┐    ┌───────────────┐    ┌──────────────┐
  │ Load .npz  │───▶│ Regenerate    │───▶│ Compose      │
  │ files      │    │ _compare.jpg  │    │ output video │
  │ (skip fit) │    │ from params   │    │ with ffmpeg  │
  └────────────┘    └───────────────┘    └──────────────┘

  → Fast re-visualization without re-tracking
```

## Models & GPU Memory

| Model | Device | Approx VRAM |
|-------|--------|-------------|
| RobustVideoMatting | GPU | ~200 MB |
| MediaPipe Landmarker | CPU (TFLite) | - |
| DECA Encoder | GPU | ~500 MB |
| MICA Encoder | GPU | ~300 MB |
| Face Parsing | GPU | ~200 MB |
| Ear Landmarker | GPU | ~50 MB |
| FLAME Decoder | GPU | ~100 MB |
| PyTorch3D Renderer | GPU | ~500 MB |
| **Total (approx)** | | **~2 GB** |

## Timing Estimates (RTX 5090, batch=32)

| Stage | Time per batch |
|-------|---------------|
| Frame preprocessing | ~2-3s |
| Rigid camera fitting (1500 iters) | ~15-20s |
| Landmark fine-tuning (200 iters) | ~3-5s |
| Photometric fine-tuning (400 iters) | ~8-12s |
| Save + visualize | ~1-2s |
| **Total per batch (landmark)** | **~25-30s** |
| **Total per batch (photometric)** | **~30-40s** |
