"""Decode each new MP4 fully and extract visible frames from the encoded file."""
from pathlib import Path
import argparse
import re
import subprocess
import imageio_ffmpeg
from flyocr.common import read_json, save_json, digest_file

root = Path(__file__).resolve().parents[1]
encoder = imageio_ffmpeg.get_ffmpeg_exe()
parser = argparse.ArgumentParser()
parser.add_argument('--letters',action='store_true',help='Verify the expanded-letter demos')
parser.add_argument('--calibrated',action='store_true',help='Verify the calibrated-input letter demos')
args = parser.parse_args()
results = []
spec = 'calibrated-examples.json' if args.calibrated else 'letter-examples.json' if args.letters else 'more-examples.json'
for item in read_json(root/'examples/microsoft-2025'/spec)['examples']:
    name = item['id']; directory = root/'demo/examples'/name
    video = directory/'video.mp4'; manifest = read_json(directory/'video-manifest.json')
    assert digest_file(video) == manifest['video_sha256']
    assert digest_file(directory/'run.json') == manifest['run_sha256']
    # Full decode, not merely a container metadata probe.
    decoded = subprocess.run([encoder, '-v', 'error', '-i', str(video), '-map', '0:v:0',
        '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    progress = dict(line.split('=',1) for line in decoded.stdout.splitlines() if '=' in line)
    assert int(progress['frame']) == 960 and progress['progress'] == 'end'
    assert abs(int(progress['out_time_us'])/1e6 - 40) < .1
    metadata = subprocess.run([encoder, '-hide_banner', '-i', str(video)], capture_output=True, text=True).stderr
    assert '1920x1080' in metadata and '24 fps' in metadata and 'h264' in metadata
    target = root/'reports/video-decoded'/name; target.mkdir(parents=True, exist_ok=True)
    subprocess.run([encoder, '-v', 'error', '-y', '-i', str(video), '-vf',
        r'select=eq(n\,0)+eq(n\,360)+eq(n\,959)', '-fps_mode', 'vfr', str(target/'%02d.png')], check=True)
    result = {'id': name, 'video_sha256': digest_file(video), 'frames_decoded': int(progress['frame']),
        'seconds': int(progress['out_time_us'])/1e6, 'width':1920, 'height':1080, 'fps':24,
        'run_sha256': manifest['run_sha256'], 'decoded_frames': [0,360,959], 'decode_errors': decoded.stderr}
    results.append(result); print(name, '960 frames decoded', flush=True)
save_json(root/('reports/letters-v2/video-verification.json' if args.calibrated else 'reports/letters/video-verification.json' if args.letters else 'reports/more-examples/video-verification.json'), {'videos':results})
