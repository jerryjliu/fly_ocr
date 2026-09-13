import numpy as np
import pytest
from flyocr.vision.coverage import balanced_retina
from flyocr.vision.retina import sample_retina
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.readout.train import probabilities


def test_calibration_reveals_missing_stroke_and_preserves_shared_sites():
    # A bottom stroke is wholly invisible to a map concentrated in the upper half.
    sites = np.stack(np.meshgrid(np.linspace(0,1,5), np.linspace(0,.4,5)), axis=-1).reshape(-1,2)
    old = np.repeat(sites, 2, axis=0).astype(np.float32)
    new, metadata = balanced_retina(old)
    blank = np.full((48,48),255,np.uint8); stroke = blank.copy(); stroke[45:,:] = 0
    assert np.array_equal(sample_retina(blank,old),sample_retina(stroke,old))
    assert not np.array_equal(sample_retina(blank,new),sample_retina(stroke,new))
    assert np.array_equal(new[::2],new[1::2])
    assert np.array_equal(new,balanced_retina(old)[0])
    assert not metadata["uses_labels"] and not metadata["learned"]


def test_input_adapter_keeps_internal_wiring_and_has_distinct_identity():
    graph = {"ptr":np.array([0,1,2,2],np.int64), "post":np.array([1,2],np.int32),
        "weight":np.array([12.,6.],np.float32), "retina":np.array([0],np.int32),
        "lamina":np.array([1],np.int32), "uv":np.array([[.1,.1]],np.float32),
        "candidates":np.array([1,2],np.int32)}
    config = StimulusConfig(sensory_gain=20,lamina_drive=20,invert=True,half_saturation=.2)
    old = Brain(graph,config); new = Brain(graph,config,retina_uv=np.array([[.9,.9]],np.float32))
    assert old.model_id != new.model_id
    assert np.array_equal(graph["uv"],[[np.float32(.1),np.float32(.1)]])
    for key in ["ptr","post","weight"]:
        assert new.graph[key] is graph[key]
    with pytest.raises(ValueError, match="eye coordinates"):
        Brain(graph,config,retina_uv=np.array([[2.,0.]],np.float32))


def test_saved_two_layer_sqrt_readout_matches_torch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)
    net = torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.ReLU(),
                              torch.nn.Linear(6,3),torch.nn.ReLU(),torch.nn.Linear(3,2)).double()
    counts = np.array([[0,1,4,9],[25,16,0,100]],np.float64)
    mean = np.array([1.,2.,.5,3.]); scale = np.array([.5,1.,2.,.7])
    with torch.no_grad():
        inp = ((torch.sqrt(torch.tensor(counts))-torch.tensor(mean))/torch.tensor(scale)).clamp(-10,10)
        expected = net(inp).softmax(dim=1).numpy()
    payload = {"mean":mean,"scale":scale,"count_transform":np.array("sqrt"),
        "hidden_coef":net[0].weight.detach().numpy(),"hidden_intercept":net[0].bias.detach().numpy(),
        "hidden2_coef":net[2].weight.detach().numpy(),"hidden2_intercept":net[2].bias.detach().numpy(),
        "coef":net[4].weight.detach().numpy(),"intercept":net[4].bias.detach().numpy()}
    assert np.allclose(probabilities(payload,counts),expected,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match="negative"):
        probabilities(payload,-counts)
    with pytest.raises(ValueError,match="Unknown"):
        probabilities({**payload,"count_transform":np.array("unknown")},counts)
