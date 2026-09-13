"""Verify the social MP4, its edit coverage and preservation of the original."""
from pathlib import Path
import json
import imageio_ffmpeg
from flyocr.common import read_json,digest_file,save_json

manifest=read_json('demo/social/manifest.json')
video=Path('demo/social/fly-ocr-social.mp4')
assert digest_file(video)==manifest['video_sha256']
for path,expected in manifest['source_sha256'].items():
    assert digest_file(path)==expected,path

coverage=[]
for chapter in manifest['chapters']:
    path=Path('demo/examples')/chapter['id']/'run.json'
    assert digest_file(path)==chapter['run_sha256'],str(path)
    run=read_json(path)
    n=len(run['events'])
    # Mirror the documented edit arithmetic, independently of the renderer.
    revealed=[min(n,int((frame/manifest['fps'])/(chapter['seconds']-2.4)*n)+1)
              for frame in range(chapter['seconds']*manifest['fps'])]
    assert revealed[0]==1 and revealed[-1]==n
    assert set(revealed)==set(range(1,n+1)),chapter['id']
    outputs={}
    for event in run['events']:
        outputs[event['row_id']]=outputs.get(event['row_id'],'')+event.get('prefix','')+event['character']
    assert all(outputs.get(row['row_id'],'')==row['raw'] for row in run['rows'])
    coverage.append({'id':chapter['id'],'events':n,'every_event_visible':True,'first_frame_prediction':run['events'][0]['character']})

stream=imageio_ffmpeg.read_frames(str(video));meta=next(stream);stream.close()
frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
assert frames==manifest['frames']==2520 and abs(seconds-manifest['seconds'])<.1
assert meta['size']==(1920,1080) and meta['fps']==24
original=read_json('demo/compilation/manifest.json')
assert digest_file('demo/compilation/fly-ocr.mp4')==original['video_sha256']
assert original['video_sha256']=='aeb4ccd9544199178f794366a9127a786ec753347eb174dc600d01fb215edcf1'
result={'video':str(video),'frames':frames,'seconds':seconds,'fps':meta['fps'],'size':list(meta['size']),
        'sha256':manifest['video_sha256'],'full_decode':True,'coverage':coverage,
        'original_compilation_unchanged':True,'original_video_sha256':original['video_sha256']}
save_json('reports/public-release/social-media-verification.json',result)
print(json.dumps(result,indent=2))
