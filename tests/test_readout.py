import numpy as np
import pytest
from flyocr.readout.train import fit_readout, predict, probabilities


def test_serialized_linear_readout_and_training_only_scaling(tmp_path):
    x = np.array([[0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3]], float)
    y = np.array([0, 1, 0, 1, 0, 1])
    p, fit = fit_readout(x, y, x+np.array([0, 100]), y)
    assert np.allclose(p["mean"], x.mean(0))
    np.savez(tmp_path/"model.npz", **p)
    loaded = np.load(tmp_path/"model.npz", allow_pickle=False)
    assert np.array_equal(predict(p, x), predict(loaded, x))
    assert np.allclose(probabilities(p, x).sum(1), 1)
    with pytest.raises(ValueError): probabilities(p, np.ones((2, 3)))
    with pytest.raises(ValueError): probabilities(p, np.array([[np.nan, 1]]))


def test_portable_mlp_scores_match_torch_reference():
    from pathlib import Path
    path=Path('artifacts/letters/readout.npz')
    if not path.exists():pytest.skip('Requires trained letter readout')
    torch=pytest.importorskip('torch')
    payload=dict(np.load(path,allow_pickle=False))
    if 'hidden_coef' not in payload:pytest.skip('Selected head is linear')
    x=np.load('artifacts/letters/heldout-predictions.npz')['features'][:8]
    def tensor(v):return torch.tensor(v,dtype=torch.float64)
    t=tensor(np.clip((x-payload['mean'])/payload['scale'],-10,10))
    hidden=torch.relu(t@tensor(payload['hidden_coef']).T+tensor(payload['hidden_intercept']))
    expected=torch.softmax(hidden@tensor(payload['coef']).T+tensor(payload['intercept']),dim=1).numpy()
    assert np.allclose(probabilities(payload,x),expected,rtol=1e-9,atol=1e-11)
