"""Export original or comma-separated example ids using a bundled encoder."""
import argparse
import os
from pathlib import Path
import subprocess
import imageio_ffmpeg
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--examples', default='original')
args=parser.parse_args()
env={**os.environ,'FFMPEG_BIN':imageio_ffmpeg.get_ffmpeg_exe(), 'FLYOCR_EXAMPLES':args.examples}
subprocess.run(['npm','run','video'],cwd=root/'viewer',env=env,check=True)
