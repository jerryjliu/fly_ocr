"""macOS integration check: enforce no network, PDF source or evaluation truth reads."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from flyocr.common import read_json, save_json

root=Path.cwd().resolve()
if not shutil.which('sandbox-exec'):
    raise SystemExit('This optional enforcement check requires macOS sandbox-exec. Regular inference is portable.')
truth=root/'examples/microsoft-2025/evaluation.json'
assert truth.exists()
profile='(version 1)(allow default)(deny network*)(deny file-read* (subpath '+json.dumps(str(root/'examples'))+'))(deny file-read* (literal '+json.dumps(str(root/'data/source/microsoft-2025.pdf'))+'))'
check="""import socket,sys
try:
 open(sys.argv[1]).read()
except PermissionError: pass
else: raise RuntimeError('Truth was readable')
try:
 socket.socket().connect(('127.0.0.1',9))
except PermissionError: pass
else: raise RuntimeError('Network was permitted')
print('Truth and networking denied by OS sandbox')
"""
subprocess.run(['sandbox-exec','-p',profile,sys.executable,'-c',check,str(truth)],check=True)
subprocess.run(['sandbox-exec','-p',profile,sys.executable,'-m','flyocr.cli','recognize','data/pdf-raster/crop.png','--output','output/isolated-run'],check=True)
expected=read_json('demo/run/run.json');actual=read_json('output/isolated-run/run.json')
for a,b in zip(expected['events'],actual['events'],strict=True):
 for k in ['id','character','counts','probabilities','sha256']:
  if a[k]!=b[k]: raise ValueError('Isolated recognition changed: '+k)
save_json('reports/isolated-inference.json',{'events':len(actual['events']),'predictions_identical':True,'neural_counts_identical':True,'network_denied':True,'evaluation_truth_denied':True,'original_pdf_denied':True,'mechanism':'macOS sandbox-exec','artifact_id':actual['artifact_id']})
print('All isolated predictions and neural counts match.')
