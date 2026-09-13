"""Fully decode release videos and verify their source/renderer fingerprints."""
from pathlib import Path
import imageio_ffmpeg
from flyocr.common import read_json,digest_file,save_json
results=[]
paths=[(Path('demo/flyocr-demo.mp4'),Path('demo/video-manifest.json'))]
paths += [(p/'video.mp4',p/'video-manifest.json') for p in sorted(Path('demo/examples').iterdir()) if (p/'video.mp4').exists()]
if Path('demo/compilation/fly-ocr.mp4').exists():paths += [(Path('demo/compilation/fly-ocr.mp4'),Path('demo/compilation/manifest.json'))]
for video,manifest in paths:
 m=read_json(manifest)
 assert digest_file(video)==m['video_sha256']
 assert digest_file('viewer/lib/replay.ts')==m['renderer_sha256']
 assert digest_file('viewer/lib/retina.ts')==m['atlas_renderer_sha256']
 if 'source' in m:assert digest_file(m['source'])==m['run_sha256']
 else:
  assert digest_file('viewer/scripts/compilation.ts')==m['compilation_renderer_sha256']
  for chapter in m['chapters']:
   if 'run_sha256' in chapter:assert digest_file(Path('demo/examples')/chapter['id']/'run.json')==chapter['run_sha256']
 stream=imageio_ffmpeg.read_frames(str(video));meta=next(stream);stream.close()
 frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
 assert frames==m['frames'] and abs(seconds-m['seconds'])<.1
 assert meta['size']==(1920,1080) and meta['fps']==24
 results.append({'video':str(video),'frames':frames,'seconds':seconds,'size':list(meta['size']),'fps':meta['fps'],'sha256':m['video_sha256'],'full_decode':True})
 print('Verified',video,frames,'frames',flush=True)
save_json('reports/public-release/media-verification.json',{'videos':results,'total_decoded_frames':sum(v['frames'] for v in results)})
