"""Validation-only comparison of a small MLP with the frozen linear readout.

No simulator, PDF, held-out images, or held-out labels are opened while choosing
the head. The already recorded downstream spike features are the sole inputs.
"""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from flyocr.common import read_json,save_json,digest_file,identity
from flyocr.readout.train import probabilities


def features(split,neurons):
    sha=digest_file(Path('data/glyphs-letters')/(split+'.npz'))
    for d in Path('data/letter-features').iterdir():
        protocol=read_json(d/'protocol.json')
        if protocol['source_sha256']!=sha or protocol['neurons']!=neurons:continue
        parts=[]
        for path in sorted(d.glob('*.npz')):
            if read_json(path.with_suffix('.json'))['sha256']!=digest_file(path):raise ValueError('Feature hash mismatch')
            parts.append(np.load(path)['features'])
        data=np.load(Path('data/glyphs-letters')/(split+'.npz'))
        result=np.concatenate(parts)
        if len(result)!=len(data['labels']):raise ValueError('Feature collection is incomplete')
        return result,data['labels']
    raise ValueError('No matching feature cache')


def main():
    output=Path('artifacts/letter-refinement');output.mkdir(parents=True,exist_ok=True)
    card=read_json('artifacts/letters/model-card.json')
    if card.get('readout_type')=='mlp' and Path('artifacts/letters/recognition.json').exists():
        print('The refined checkpoint is already complete.');return
    selection=read_json('artifacts/letters/selection.json')
    configs=[{'hidden':64,'weight_decay':.01},{'hidden':128,'weight_decay':.01},{'hidden':128,'weight_decay':.001}]
    protocol={'source_linear_artifact_id':card['artifact_id'],'inputs':'recorded spike counts only','configs':configs,
        'optimizer':'AdamW','learning_rate':.002,'max_epochs':120,'patience':18,'batch_size':256,
        'dropout':.1,'seed':817,'feature_clip':[-10,10],'model_selection':'validation accuracy; keep linear on ties',
        'test_and_pdf_access':False}
    save_json(output/'protocol.json',protocol)
    tx,ty=features('train',selection['neurons']);vx,vy=features('validation',selection['neurons'])
    scaler=StandardScaler().fit(tx)
    train=np.clip(scaler.transform(tx),-10,10).astype(np.float32)
    val=np.clip(scaler.transform(vx),-10,10).astype(np.float32)
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4)
    x=torch.tensor(train,device=device);y=torch.tensor(ty.astype(np.int64),device=device)
    v=torch.tensor(val,device=device);vy_t=torch.tensor(vy.astype(np.int64),device=device)
    overall=card['fit']['validation_accuracy'];chosen=None;trials=[];begin=time.perf_counter()
    for config in configs:
        torch.manual_seed(817);rng=np.random.default_rng(817)
        net=nn.Sequential(nn.Linear(train.shape[1],config['hidden']),nn.ReLU(),nn.Dropout(.1),nn.Linear(config['hidden'],len(card['characters']))).to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=config['weight_decay'])
        best=-1.;best_state=None;best_epoch=0;history=[]
        for epoch in range(1,121):
            net.train()
            order=rng.permutation(len(train))
            for lo in range(0,len(order),256):
                idx=torch.tensor(order[lo:lo+256],device=device)
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(net(x[idx]),y[idx]);loss.backward();optimizer.step()
            net.eval()
            with torch.no_grad():accuracy=float((net(v).argmax(1)==vy_t).float().mean().cpu())
            history.append(accuracy)
            if accuracy>best:
                best=accuracy;best_state={k:t.detach().cpu().clone() for k,t in net.state_dict().items()};best_epoch=epoch
            if epoch%10==0:print(config,'epoch',epoch,'best validation',round(best,4),flush=True)
            if epoch-best_epoch>=18:break
        trial={**config,'best_epoch':best_epoch,'epochs':epoch,'validation_accuracy':best,'history':history}
        trials.append(trial)
        if best>overall:
            overall=best
            chosen={'mean':scaler.mean_,'scale':scaler.scale_,
                'hidden_coef':best_state['0.weight'].numpy(),'hidden_intercept':best_state['0.bias'].numpy(),
                'coef':best_state['3.weight'].numpy(),'intercept':best_state['3.bias'].numpy(),
                'classes':np.arange(len(card['characters'])),'neurons':np.array(selection['neurons'],np.int32)}
            np.savez_compressed(output/'readout.npz',**chosen)
            # Serialization must preserve the selected predictions exactly.
            net.load_state_dict(best_state);net.eval()
            with torch.no_grad():expected=net(v).argmax(1).cpu().numpy()
            actual=probabilities(chosen,vx).argmax(1)
            if not np.array_equal(expected,actual):raise ValueError('Serialized MLP predictions differ from Torch')
            save_json(output/'chosen.json',trial)
        save_json(output/'validation.json',{'linear_accuracy':card['fit']['validation_accuracy'],'best_accuracy':overall,
            'selected':'mlp' if chosen else 'linear','trials':trials,'device':device,'wall_seconds':time.perf_counter()-begin,
            'test_and_pdf_access':False,'serialization_predictions_exact':True})
    print('Validation-selected readout:','MLP' if chosen else 'linear',f'{overall:.1%}',flush=True)


if __name__=='__main__':main()
