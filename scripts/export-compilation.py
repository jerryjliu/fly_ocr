"""Render the final compilation with the same saved inference as the viewer."""
import os
from pathlib import Path
import subprocess
import imageio_ffmpeg
root=Path(__file__).resolve().parents[1]
subprocess.run(['npm','run','compilation'],cwd=root/'viewer',env={**os.environ,'FFMPEG_BIN':imageio_ffmpeg.get_ffmpeg_exe()},check=True)
