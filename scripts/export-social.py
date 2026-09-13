"""Render the social cut, using an installed Chrome or Playwright Chromium."""
import os
from pathlib import Path
import subprocess
import imageio_ffmpeg

root=Path(__file__).resolve().parents[1]
env={**os.environ,'FFMPEG_BIN':imageio_ffmpeg.get_ffmpeg_exe()}
# The machine-specific path is discovered, never embedded in public artifacts.
chrome=Path('/Applications')/'Google Chrome.app'/'Contents'/'MacOS'/'Google Chrome'
if chrome.exists():env.setdefault('FLYOCR_CHROME',str(chrome))
subprocess.run(['npm','run','social'],cwd=root/'viewer',env=env,check=True)
