import copy
import numpy as np
import pytest
from flyocr.vision.atlas import make_atlas, validate_atlas, load_atlas
from flyocr.common import save_json


def test_atlas_rejects_changed_identity_and_receptor_order(tmp_path):
    ids=np.array([101,202,303]); retina=np.array([0,2]); uv=np.array([[.1,.2],[.8,.7]],np.float32)
    atlas=make_atlas(ids,retina,uv,['L','R'],'test-graph')
    for name,a in [('ids',ids),('retina',retina),('uv',uv)]: np.save(tmp_path/(name+'.npy'),a)
    save_json(tmp_path/'manifest.json',{'graph_id':'test-graph','retinal_inputs':2})
    save_json(tmp_path/'retinal-atlas.json',atlas)
    assert load_atlas(tmp_path)==atlas
    bad=copy.deepcopy(atlas);bad['eye'][0]='R'
    with pytest.raises(ValueError,match='checksum'):validate_atlas(bad)
    reordered=make_atlas(ids,retina[::-1],uv,['L','R'],'test-graph')
    save_json(tmp_path/'retinal-atlas.json',reordered)
    with pytest.raises(ValueError,match='order'):load_atlas(tmp_path)
    with pytest.raises(ValueError,match='another graph'):validate_atlas(atlas,graph_id='other')
