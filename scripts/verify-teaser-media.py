"""Verify the rendered event trace, MP4 and preservation of earlier videos."""
from pathlib import Path
import json
import imageio_ffmpeg
from flyocr.common import read_json,digest_file,save_json

manifest=read_json('demo/teaser/manifest.json')
video=Path('demo/teaser/fly-ocr-teaser.mp4')
assert digest_file(video)==manifest['video_sha256']
for path,expected in manifest['source_sha256'].items():
    assert digest_file(path)==expected,path

coverage=[]
assert [c['id'] for c in manifest['chapters']]==['calibrated-labels','income-table','calibrated-labels-more']
for chapter in manifest['chapters']:
    path=Path('demo/examples')/chapter['id']/'run.json'
    assert digest_file(path)==chapter['run_sha256'],str(path)
    run=read_json(path)
    event_ids=[e['id'] for e in run['events']]
    trace=chapter['frame_event_ids']
    assert len(trace)==chapter['seconds']*manifest['fps']
    assert trace[0]==event_ids[chapter['startEvent']-1] and trace[-1]==event_ids[-1]
    assert set(trace)==set(event_ids[chapter['startEvent']-1:])
    indices={event_id:i for i,event_id in enumerate(event_ids)}
    positions=[indices[event_id] for event_id in trace]
    assert all(0<=b-a<=1 for a,b in zip(positions,positions[1:]))
    outputs={}
    for event in run['events']:
        outputs[event['row_id']]=outputs.get(event['row_id'],'')+event.get('prefix','')+event['character']
    assert all(outputs.get(row['row_id'],'')==row['raw'] for row in run['rows'])
    coverage.append({'id':chapter['id'],'frames':len(trace),'first_event':trace[0],
                     'last_event':trace[-1],'selected_glyphs':len(set(trace)),'rendered_trace_verified':True})

stream=imageio_ffmpeg.read_frames(str(video));meta=next(stream);stream.close()
frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
assert frames==manifest['frames']==960 and abs(seconds-40)<.1
assert meta['size']==(1920,1080) and meta['fps']==24
previous={
    'demo/compilation/fly-ocr.mp4':'aeb4ccd9544199178f794366a9127a786ec753347eb174dc600d01fb215edcf1',
    'demo/social/fly-ocr-social.mp4':'9a758dd1a7f5d31be0af769cbb280757e1b1d342f9c4e12bbe942177b6aa2cdc',
}
for path,expected in previous.items():assert digest_file(path)==expected,path
result={'video':str(video),'frames':frames,'seconds':seconds,'fps':meta['fps'],'size':list(meta['size']),
        'sha256':manifest['video_sha256'],'full_decode':True,'coverage':coverage,
        'previous_videos_unchanged':previous}
save_json('reports/public-release/teaser-media-verification.json',result)
print(json.dumps(result,indent=2))
