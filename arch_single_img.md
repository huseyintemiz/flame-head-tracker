# FLAME Head Tracker — Single Image Pipeline

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: RGB Image [H, W, 3]                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DATA PREPARATION                       │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│  │  Background   │   │  MediaPipe Face  │   │   DECA Encoder    │   │
│  │  Matting      │   │  Landmarker v2   │   │                   │   │
│  │  (ResNet50)   │   │                  │   │  ┌─────────────┐  │   │
│  │              │   │  478 dense lmks   │   │  │ exp  [1,100]│  │   │
│  │  img → alpha │   │  52 blendshapes   │   │  │ pose [1,6]  │  │   │
│  │  composite   │   │  68 sparse lmks   │   │  │ tex  [1,50] │  │   │
│  │  w/ white bg │   │  10 eye lmks      │   │  │ light[1,9,3]│  │   │
│  └──────┬───────┘   └────────┬─────────┘   │  └─────────────┘  │   │
│         │                    │              └────────┬──────────┘   │
│         ▼                    ▼                       │              │
│  ┌──────────────┐   ┌──────────────────┐            │              │
│  │ Image Align  │   │  Ear Landmarker  │            │              │
│  │ (Affine)     │   │  (CNN, optional) │            │              │
│  │ → [256,256]  │   │  → 20 lmks/ear   │            │              │
│  └──────┬───────┘   └────────┬─────────┘            │              │
│         │                    │                       │              │
│         ▼                    ▼                       │              │
│  ┌──────────────┐   ┌──────────────────┐            │              │
│  │ Face Parsing │   │  MICA Encoder    │            │              │
│  │ (19 classes) │   │  (ArcFace)       │            │              │
│  │ → [512,512]  │   │  → shape [1,300] │            │              │
│  └──────┬───────┘   └────────┬─────────┘            │              │
│         │                    │                       │              │
└─────────┼────────────────────┼───────────────────────┼──────────────┘
          │                    │                       │
          ▼                    ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INITIAL COEFFICIENTS DICT                           │
│                                                                     │
│   shape [1,300]  exp [1,100]  head_pose [1,3]  jaw_pose [1,3]      │
│   tex [1,50]  light [1,9,3]  gt_landmarks [1,68,3]                 │
│   gt_eye_landmarks [1,10,3]  gt_ear_landmarks [1,20,3]             │
│   blendshape_scores [1,52]   parsing [1,512,512]                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          │                                           │
          ▼                                           ▼
┌─────────────────────────┐             ┌─────────────────────────────┐
│  PATH A: Landmark-Only  │             │  PATH B: Photometric        │
│  (faster)               │             │  (slower, more accurate)    │
└────────┬────────────────┘             └────────────┬────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2: RIGID CAMERA POSE FITTING                     │
│                      (1500 iterations)                              │
│                                                                     │
│  Optimized params:                                                  │
│    d_camera_rotation [N,3]                                          │
│    d_camera_translation [N,3]                                       │
│    d_fov [N] (optional)                                             │
│                                                                     │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                │
│  │   FLAME    │───▶│ Perspective│───▶│  NDC       │                │
│  │  Forward   │    │ Projection │    │ Conversion │                │
│  │ [N,5023,3] │    │ [N,108,3]  │    │ [-1,1]     │                │
│  └────────────┘    └────────────┘    └─────┬──────┘                │
│                                            │                        │
│  Loss: L2(projected_lmks, gt_lmks)         │                        │
│    - face lmks × 300                       │                        │
│    - jawline   × 200                       │                        │
│    - ear lmks  (yaw-weighted)              │                        │
└────────────────────────────────────────────┼────────────────────────┘
                                             │
         ┌───────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│             STAGE 3A: LANDMARK FINE-TUNING (200 iters)              │
│                     (if landmark-only)                              │
│                                                                     │
│  Optimized:  d_exp [N,100]  d_jaw [N,3]  eye_pose [N,6]           │
│                                                                     │
│  Loss:                                                              │
│    - L1(face lmks) × 500                                           │
│    - L1(eye lmks)  × 500                                           │
│    - ||exp||²      × 0.025  (regularization)                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
         ┌────────────────────────┘
         │
┌────────┼────────────────────────────────────────────────────────────┐
│        ▼   STAGE 3B: PHOTOMETRIC FINE-TUNING (400 iters)           │
│                     (if photometric)                                │
│                                                                     │
│  Additional optimized params:                                       │
│    d_shape [N,300]  d_tex [N,50]  d_texture [N,3,256,256]          │
│    d_light [N,9,3]  + camera params                                │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐                   │
│  │ FLAMETex │──▶│ Renderer │──▶│ Rendered Img │                   │
│  │ [N,3,    │   │(PyTorch  │   │ [N,3,256,256]│                   │
│  │  256,256]│   │  3D)     │   │              │                   │
│  └──────────┘   └──────────┘   └──────┬───────┘                   │
│                                       │                             │
│  Loss:                                │                             │
│    - L1(pixels, gt) × 2   (within face mask)                       │
│    - L2(lmks)       × 1                                            │
│    - L2(eye lmks)   × 2                                            │
│    - ||d_shape||²   × 0.1                                          │
│    - ||exp||²       × 1e-4                                         │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: OUTPUT & VISUALIZATION                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              5-Panel Comparison [256 × 1280]                 │   │
│  │                                                              │   │
│  │  ┌─────────┬──────────┬──────────┬──────────┬──────────┐    │   │
│  │  │Original │Landmarks │ Mesh     │ Textured │ Frontal  │    │   │
│  │  │ Image   │ Overlay  │ Overlay  │ Mesh     │ View     │    │   │
│  │  │         │ 68+10+40 │ 40%+60%  │ Full     │ pose=0   │    │   │
│  │  │[256×256]│[256×256] │[256×256] │[256×256] │[256×256] │    │   │
│  │  └─────────┴──────────┴──────────┴──────────┴──────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Output dict:                                                       │
│    shape [N,300]  exp [N,100]  head_pose [N,3]  jaw_pose [N,3]     │
│    eye_pose [N,6]  tex [N,50]  light [N,9,3]  cam [N,6]           │
│    fov [N]  K [N,3,3]  texture [N,3,256,256]                      │
│    img_rendered [N,256,256,3]  shape_rendered [N,256,256,3]        │
└─────────────────────────────────────────────────────────────────────┘
```

## Models Used

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **RobustVideoMatting** | Background removal | [H,W,3] RGB | Alpha matte [H,W] |
| **MediaPipe Face Landmarker v2** | Face detection + landmarks | [H,W,3] RGB | 478 dense lmks, 52 blendshapes |
| **DECA** | Initial FLAME coefficients | [224,224,3] face crop | exp, pose, tex, light |
| **MICA** | Shape identity code | [224,224,3] + ArcFace | shape [1,300] |
| **Ear Landmarker** | Ear keypoints (optional) | [368,368,3] | 20 lmks/ear |
| **Face Parsing** | Semantic face segmentation | [512,512,3] | 19-class mask |
| **FLAME** | 3D Morphable Face Model | shape+exp+pose params | 5023 vertices |
| **FLAMETex** | Texture generation | tex coefficients | [3,256,256] texture map |
| **PyTorch3D Renderer** | Mesh rendering | vertices + texture | [256,256,3] rendered image |

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Render size | 256 × 256 |
| Default FOV | 20° |
| Rigid fitting iterations | 1500 |
| Landmark fine-tuning iterations | 200 |
| Photometric fine-tuning iterations | 400 |
| Face landmark weight | 300 |
| Jawline landmark weight | 200 |
| Expression regularization (lmk) | 0.025 |
| Expression regularization (photo) | 1e-4 |
| Photometric loss weight | 2.0 |
