"""Motion-interpolate recorded videos from 10 fps -> 30 fps for smooth playback
at REAL-TIME speed (the sim is 10 Hz, so raw recordings are 10 fps = jerky).
Keeps the true 100 s duration; ffmpeg synthesizes intermediate frames.

Usage: python -m carla_rl.scripts.smooth_videos [dir]
"""

import glob
import os
import subprocess
import sys

import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
src_dir = sys.argv[1] if len(sys.argv) > 1 else r'carla_rl\logs\pp_demo_video'

for src in sorted(glob.glob(os.path.join(src_dir, '*.mp4'))):
    if src.endswith('_smooth.mp4'):
        continue
    dst = src[:-4] + '_smooth.mp4'
    print(f'interpolating {os.path.basename(src)} -> {os.path.basename(dst)}', flush=True)
    # blend (simple frame mix) is robust — mci motion-interpolation corrupted a
    # long clip once; faststart puts the moov atom up front for playback.
    subprocess.run(
        [ffmpeg, '-y', '-i', src, '-vf', 'minterpolate=fps=30:mi_mode=blend',
         '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
print('SMOOTH VIDEOS DONE', flush=True)
