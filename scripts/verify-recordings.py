"""Verify every bundled recording without loading the source connectome."""
from pathlib import Path
from flyocr.demo.schema import verify
from flyocr.common import read_json,save_json
cards={read_json(p)['artifact_id']:p.parent for p in Path('artifacts').glob('*/model-card.json')}
results=[]
for path in [Path('demo/run'),*sorted(Path('demo/examples').iterdir())]:
 if not (path/'run.json').exists():continue
 run=read_json(path/'run.json');r=verify(path,cards[run['artifact_id']]);results.append({'run':str(path),**r})
save_json('reports/public-release/replay-verification.json',{'recordings':results,'events':sum(r['verified_events'] for r in results),'network_required':False,'graph_required':False})
print('Verified',len(results),'recordings and',sum(r['verified_events'] for r in results),'events')
