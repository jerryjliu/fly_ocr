"""Validation-only PCA/RBF readout comparison using cached neural responses."""
from pathlib import Path
import importlib.util
import argparse
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from flyocr.common import read_json,save_json

spec=importlib.util.spec_from_file_location('refinement','scripts/refine-letter-readout.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--components',type=int,default=64);args=parser.parse_args()
 components=args.components
 if components not in (64,128,256):raise ValueError('Choose 64, 128, or 256 components')
 output=Path('artifacts/letter-kernel' if components==64 else 'artifacts/letter-kernel-'+str(components));output.mkdir(parents=True,exist_ok=True)
 card=read_json('artifacts/letters/model-card.json');neurons=read_json('artifacts/letters/selection.json')['neurons']
 protocol={'source_artifact_id':card['artifact_id'],'inputs':'recorded brain spike counts only','components':components,'whiten':True,'gamma_multipliers':[.25,1.,4.],'ridge_strengths':[.001,.01,.1], 'model_selection':'validation accuracy only','test_access':False,'note':'Exploratory refinement after observing earlier PDF/test errors. No PDF inputs or labels enter fitting.'}
 save_json(output/'protocol.json',protocol)
 tx,ty=module.features('train',neurons);vx,vy=module.features('validation',neurons)
 scale=StandardScaler().fit(tx)
 x=np.clip(scale.transform(tx),-10,10);v=np.clip(scale.transform(vx),-10,10)
 begin=time.perf_counter()
 with threadpool_limits(limits=4):
  pca=PCA(n_components=components,whiten=True,svd_solver='randomized',random_state=817).fit(x)
  x=pca.transform(x);v=pca.transform(v)
  d=cdist(x,x,'sqeuclidean');vd=cdist(v,x,'sqeuclidean')
  targets=np.eye(len(card['characters']))[ty]
  trials=[];best=card['fit']['validation_accuracy']
  print('PCA explained variance',float(pca.explained_variance_ratio_.sum()),flush=True)
  for multiplier in [.25,1.,4.]:
   gamma=multiplier/components
   kernel=np.exp(-gamma*d);vkernel=np.exp(-gamma*vd)
   for alpha in [.001,.01,.1]:
    k=kernel.copy();k.flat[::len(k)+1]+=alpha
    dual=cho_solve(cho_factor(k,lower=True,overwrite_a=True,check_finite=False),targets,check_finite=False)
    logits=vkernel@dual;accuracy=float((logits.argmax(1)==vy).mean())
    trial={'gamma_multiplier':multiplier,'alpha':alpha,'validation_accuracy':accuracy};trials.append(trial)
    print(trial,flush=True)
    if accuracy>best:
     best=accuracy
     temperatures=[.03,.05,.08,.12,.2,.3,.5,1.]
     temperature=min(temperatures,key=lambda t:float(-np.log(np.maximum(softmax(logits/t,axis=1)[np.arange(len(vy)),vy],1e-15)).mean()))
     np.savez_compressed(output/'readout.npz',mean=scale.mean_,scale=scale.scale_,pca_mean=pca.mean_,pca_components=pca.components_,pca_variance=pca.explained_variance_,kernel_basis=x,kernel_dual=dual,kernel_gamma=np.array(gamma),temperature=np.array(temperature),classes=np.arange(len(card['characters'])),neurons=np.array(neurons,np.int32))
     save_json(output/'chosen.json',{**trial,'temperature':temperature})
    save_json(output/'validation.json',{'previous_validation_accuracy':card['fit']['validation_accuracy'],'best_accuracy':best,'selected':'rbf' if (output/'readout.npz').exists() else card.get('readout_type','linear'),'trials':trials,'test_access':False,'wall_seconds':time.perf_counter()-begin})
 print('Best validation accuracy',best,flush=True)

if __name__=='__main__':main()
