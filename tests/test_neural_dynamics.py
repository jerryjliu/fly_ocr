import numpy as np
import pytest
from flyocr.brain.model import Brain, StimulusConfig


def graph(n=3):
    return {"ptr": np.array([0, 1, 2, 2], np.int64), "post": np.array([1, 2], np.int32), "weight": np.array([12., -6.], np.float32)}


def test_analytic_subthreshold():
    a = {"ptr": np.array([0, 0], np.int64), "post": np.array([], np.int32), "weight": np.array([], np.float32)}
    b = Brain(a)
    b.advance_drive([5.], 10)
    assert b.state["v"][0] == pytest.approx(-52+5*(1-np.exp(-.5)), abs=2e-4)
    assert b.state["counts"][0] == 0


def test_split_step_and_complete_restore():
    a = Brain(graph()); b = Brain(graph())
    one = a.advance_drive([30., 5., 8.], 100)
    split = sum((b.advance_drive([30., 5., 8.], 25) for _ in range(4)))
    assert np.array_equal(one, split)
    assert np.allclose(a.state["v"], b.state["v"], atol=2e-4)
    snap = a.snapshot()
    expected = a.advance_drive([10., 30., 0.], 40)
    a.advance_drive([80., 0., 80.], 100)
    a.restore(snap)
    assert np.array_equal(expected, a.advance_drive([10., 30., 0.], 40))
    with pytest.raises(ValueError): a.restore({"v": snap["v"]})


def test_brian2_independent_reference():
    brian = pytest.importorskip("brian2")
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = .1*brian.ms
    neurons = brian.NeuronGroup(3, """
       dv/dt = (-52*mV-v+I+g)/(20*ms) : volt (unless refractory)
       dg/dt = -g/(5*ms) : volt (unless refractory)
       I : volt
    """, threshold="v > -45*mV", reset="v=-52*mV; g=0*mV", refractory=2.2*brian.ms, method="exact")
    neurons.v = -52*brian.mV
    neurons.I = [30, 5, 8]*brian.mV
    synapses = brian.Synapses(neurons, neurons, "w : volt", on_pre="g_post += w", delay=1.8*brian.ms)
    synapses.connect(i=[0, 1], j=[1, 2])
    synapses.w = [12., -6.]*brian.mV
    monitor = brian.SpikeMonitor(neurons)
    brian.run(100*brian.ms)
    native = Brain(graph())
    counts = native.advance_drive([30, 5, 8], 100)
    assert np.array_equal(counts, monitor.count[:])
    assert np.allclose(native.state["v"], neurons.v[:]/brian.mV, atol=.003)


def test_invalid_native_inputs_rejected():
    g = graph(); g["post"] = np.array([8, 1], np.int32)
    with pytest.raises(ValueError): Brain(g)
    b = Brain(graph())
    with pytest.raises(ValueError): b.advance_drive([np.nan, 0, 0], 10)
    with pytest.raises(ValueError): b.advance_drive([0, 0, 0], .15)
    with pytest.raises(ValueError): StimulusConfig(bins=3).validate()
