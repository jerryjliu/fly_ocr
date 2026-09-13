"""Create a clean, allowlisted public export; never copy local Git history.

No file content is silently rewritten. The audit fails on suspicious material;
review findings locally before retrying. Source data and model artifacts retain
separate licenses. This is a targeted hygiene check, not a security guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOP={'src','tests','scripts','assets','artifacts','configs','demo','docs','examples','licenses','reports','research','viewer','.github'}
ROOT_FILES={'README.md','MODEL_CARD.md','THIRD_PARTY_NOTICES.md','LICENSE','pyproject.toml','uv.lock','.python-version','.gitignore','.gitattributes','CITATION.cff'}
BLOCKED_PARTS={'.git','.venv','__pycache__','.pytest_cache','.cache','node_modules','dist','.openai','.wrangler','.next','.vinext','video-frames','video-decoded'}
BLOCKED_SUFFIX={'.log','.pyc','.so','.dylib','.pem','.key','.tsbuildinfo'}
BLOCKED_REPORTS={'viewer-audit-initial.json','viewer-audit.json','video-contact-sheet.jpg'}
PATTERNS={
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
 'openai_token':rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
 'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'slack_token':rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
 'credential_url':rb'https?://[^\s/"<>:]+:[^\s/"<>]+@',
 'private_home_path':rb'(?:/Users/|/home/)[A-Za-z0-9_.-]+/',
}
def files():
 for p in ROOT.rglob('*'):
  rel=p.relative_to(ROOT)
  if not p.is_file() or p.is_symlink():continue
  if len(rel.parts)==1:
   if rel.name not in ROOT_FILES:continue
  elif rel.parts[0] not in TOP:continue
  if any(part in BLOCKED_PARTS for part in rel.parts):continue
  if p.suffix in BLOCKED_SUFFIX or p.name=='.DS_Store' or p.name.startswith('.env'):continue
  if rel.parts[:2] in [('viewer','public')] and rel.parts[2] not in ['fonts','favicon.svg']:continue
  if rel.parts[0]=='viewer' and p.name.startswith('.') and p.name!='.gitignore':continue
  if rel.parts[0]=='reports' and p.name in BLOCKED_REPORTS:continue
  if p.name in {'release-manifest.json','release-manifest.py'}:continue  # replaced by current export manifest
  yield p,rel

def audit(selected):
 findings=[]
 for p,rel in selected:
  if p.stat().st_size>=100*1024**2:findings.append({'file':str(rel),'reason':'exceeds GitHub file limit'})
  data=p.read_bytes()
  if p.suffix=='.npz':
   with zipfile.ZipFile(p) as archive:
    data+=b''.join(archive.read(name) for name in archive.namelist())
  # Patterns scan raw files, including binary strings; no matched secrets printed.
  for name,pattern in PATTERNS.items():
   if re.search(pattern,data):findings.append({'file':str(rel),'reason':name})
 return findings

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='output/public-release');parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
 selected=sorted(files(),key=lambda x:str(x[1]));findings=audit(selected)
 if findings:print(json.dumps({'findings':findings},indent=2));raise SystemExit('Public export audit failed')
 print('Audit passed:',len(selected),'files; no credential patterns or private home paths found')
 if args.audit_only:return
 output=(ROOT/args.output).resolve()
 if ROOT not in output.parents or not output.is_relative_to(ROOT/'output'):raise ValueError('Export must be under ignored output/')
 package=output/'fly_ocr'
 if package.exists():raise ValueError('Export directory exists; choose a new --output to preserve it')
 package.mkdir(parents=True)
 hashes={}
 for p,rel in selected:
  target=package/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
  hashes[str(rel)]=hashlib.sha256(target.read_bytes()).hexdigest()
 manifest={'format':1,'files':hashes,'files_count':len(hashes),'audit':{'credential_patterns':list(PATTERNS),'findings':0,'private_history_included':False},'graph_id':json.loads((ROOT/'artifacts/graph-manifest.json').read_text())['graph_id']}
 (package/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 archive=output/'fly_ocr-public.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(package.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(output))
 print('Exported',package,'and',archive)
if __name__=='__main__':main()
