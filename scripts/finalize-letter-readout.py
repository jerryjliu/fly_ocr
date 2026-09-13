"""Freeze the validation-selected head, then evaluate the shared held-out split."""
from pathlib import Path
import shutil
import numpy as np
from flyocr.common import read_json,save_json,digest_file,identity
from flyocr.readout.train import predict,probabilities
from flyocr.eval.metrics import classification


def main():
    directory=Path('artifacts/letters'); refinement=Path('artifacts/letter-refinement')
    if (directory/'recognition.json').exists() and read_json(directory/'model-card.json').get('readout_type')=='mlp':
        print('The refined checkpoint is already finalized.');return
    validation=read_json(refinement/'validation.json')
    if len(validation['trials'])!=3:raise ValueError('Validation comparison is incomplete')
    if not (directory/'recognition.json').exists():raise ValueError('Original held-out feature extraction has not finished')
    if validation['selected']=='linear':return
    original=Path('artifacts/letters-linear')
    if original.exists():raise ValueError('Refinement was already finalized; do not overwrite the reference')
    shutil.copytree(directory,original)
    card=read_json(original/'model-card.json');card.pop('artifact_id')
    chosen=read_json(refinement/'chosen.json')
    shutil.copy(refinement/'readout.npz',directory/'readout.npz')
    card.update({'readout_type':'mlp','hidden_units':chosen['hidden'],
        'readout_sha256':digest_file(directory/'readout.npz'),
        'fit':{'validation_accuracy':validation['best_accuracy'],'selection':'linear vs three small MLP candidates, validation only',
               'hidden_units':chosen['hidden'],'weight_decay':chosen['weight_decay'],'epoch':chosen['best_epoch']},
        'learned':'training standardization and a one-hidden-layer ReLU readout on downstream spike counts',
        'refinement_protocol_sha256':digest_file(refinement/'protocol.json')})
    card['artifact_id']=identity(card);save_json(directory/'model-card.json',card)
    # Head selection is now frozen, before these test labels are opened.
    test=np.load(original/'heldout-predictions.npz',allow_pickle=False)
    payload=dict(np.load(directory/'readout.npz',allow_pickle=False))
    ex,y=test['features'],test['labels'];p=probabilities(payload,ex);pred=payload['classes'][p.argmax(1)]
    result=read_json(original/'recognition.json')
    result.update({'artifact_id':card['artifact_id'],'readout_type':'mlp','neural':classification(y,pred,np.arange(len(card['characters']))),
        'validation':card['fit'],'linear_reference_artifact_id':read_json(original/'model-card.json')['artifact_id'],
        'linear_reference_accuracy':result['neural']['accuracy'],'refinement_seconds':validation['wall_seconds']})
    chars=card['characters'];classes=np.arange(len(chars))
    for name,alphabet in {'uppercase':'ABCDEFGHIJKLMNOPQRSTUVWXYZ','lowercase':'abcdefghijklmnopqrstuvwxyz','digits':'0123456789','punctuation':',.-()$'}.items():
        mask=np.isin(y,[chars.index(c) for c in alphabet]);result['by_group'][name]=classification(y[mask],pred[mask],classes)
    groups=read_json('data/glyphs-letters/test.json')
    for font in sorted({g['family'] for g in groups}):
        mask=np.array([g['family']==font for g in groups]);result['by_test_font'][font]=classification(y[mask],pred[mask],classes)
    result['wall_seconds']+=validation['wall_seconds']
    np.savez_compressed(directory/'heldout-predictions.npz',labels=y,predictions=pred,probabilities=p,features=ex)
    save_json(directory/'recognition.json',result)
    print('Held-out accuracy',result['neural']['accuracy'],flush=True)
    for name,v in result['by_group'].items():print(name,v['correct'],'/',v['n'],v['accuracy'],flush=True)


if __name__=='__main__':main()
