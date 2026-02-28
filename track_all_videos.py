"""
Batch FLAME Head Tracker — run tracking on multiple video files.

Usage:
    python track_all_videos.py                          # process all chunks in default dir
    python track_all_videos.py --input_dir path/to/videos
    python track_all_videos.py --input_dir path/to/videos --save_path ./output
    python track_all_videos.py video1.mp4 video2.mp4    # process specific files
"""

import os
import sys
import glob
import argparse
from time import time

WORKING_DIR = '/home/dipcik/phdprojects/flame-head-tracker'
os.chdir(WORKING_DIR)

import torch
torch.cuda.set_device('cuda:0')

from tracker_video import track_video


def get_tracker_cfg(video_path, save_path='./output', device='cuda:0', batch_size=32):
    return {
        # model paths
        'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task',
        'flame_model_path': './models/FLAME2020/generic_model.pkl',
        'flame_lmk_embedding_path': './models/landmark_embedding.npy',
        'ear_landmarker_path': './models/ear_landmarker.pth',
        'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
        'face_parsing_model_path': './models/79999_iter.pth',
        'template_mesh_file_path': './models/head_template.obj',
        'result_img_size': 512,
        'use_matting': True,
        'optimize_fov': False,
        'device': device,

        # video settings
        'original_fps': 25,
        'subsample_fps': 25,
        'output_fps': 25,
        'photometric_fitting': True,
        'video_path': video_path,
        'save_path': save_path,
        'batch_size': batch_size,
        'realign': True,
    }


def main():
    parser = argparse.ArgumentParser(description='Batch FLAME Head Tracker')
    parser.add_argument('videos', nargs='*', help='Specific video files to process')
    parser.add_argument('--input_dir', default='get_video/subclips_yolo_30s',
                        help='Directory containing video files (default: get_video/subclips_yolo)')
    parser.add_argument('--save_path', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pattern', default='*.mp4', help='Glob pattern for videos (default: *.mp4)')
    args = parser.parse_args()

    # Collect video files
    if args.videos:
        video_files = args.videos
    else:
        video_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} videos to process")
    print(f"Output directory: {args.save_path}")
    print(f"Device: {args.device}, Batch size: {args.batch_size}")
    print("=" * 60)

    total_start = time()
    results = []

    for i, video_path in enumerate(video_files):
        fname = os.path.basename(video_path)
        print(f"\n[{i+1}/{len(video_files)}] Processing: {fname}")
        print("-" * 40)

        cfg = get_tracker_cfg(
            video_path=video_path,
            save_path=args.save_path,
            device=args.device,
            batch_size=args.batch_size,
        )

        dt = time()
        try:
            track_video(cfg)
            elapsed = time() - dt
            results.append((fname, elapsed, "OK"))
            print(f"  Done: {elapsed/60:.2f} min")
        except Exception as e:
            elapsed = time() - dt
            results.append((fname, elapsed, f"FAILED: {e}"))
            print(f"  FAILED after {elapsed/60:.2f} min: {e}")

    # Summary
    total_elapsed = time() - total_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = sum(1 for _, _, s in results if s == "OK")
    failed = len(results) - ok
    for fname, elapsed, status in results:
        print(f"  {fname}: {elapsed/60:.2f} min — {status}")
    print(f"\nTotal: {ok} OK, {failed} failed")
    print(f"Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} hours)")


if __name__ == '__main__':
    main()
