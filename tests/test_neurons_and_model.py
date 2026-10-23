import torch
import pytest
from doomrsnn.neurons import lif_step


def test_no_spike_under_threshold():
    spike, membrane=lif_step(torch.tensor([0.1]),torch.tensor([0.0]),torch.tensor([0.0]))
    assert spike.item()==0 and membrane.item()==pytest.approx(0.1)


def test_spikes_at_threshold():
    spike, membrane=lif_step(torch.tensor([0.4]),torch.tensor([0.0]),torch.tensor([0.0]))
    assert spike.item()==1 and membrane.item()==pytest.approx(0.4)


def test_subtractive_reset():
    spike,mem=lif_step(torch.tensor([0.1]),torch.tensor([0.8]),torch.tensor([1.0]))
    assert mem.item()==pytest.approx(0.1,abs=1e-6)
    assert spike.item()==0


def test_surrogate_derivative_finite():
    x=torch.tensor([0.4],requires_grad=True)
    spike,_=lif_step(x,torch.tensor([0.0]),torch.tensor([0.0]))
    spike.sum().backward()
    assert x.grad is not None and x.grad.item()>0


def test_model_forward_shapes(tiny_model):
    logits,value,state,d=tiny_model.forward_step(torch.rand(2,3,16,16))
    assert logits.shape==(2,6) and value.shape==(2,)
    assert len(state.membranes)==len(state.spikes)==4
    assert d['spike_rate'].shape==(2,)


def test_sequential_replay_matches_stepwise(tiny_model):
    tiny_model.eval()
    frames=torch.rand(3,1,3,16,16)
    with torch.no_grad():
        full_logits,full_values,full_state,_=tiny_model.replay(frames)
        state=None;lo=[];va=[]
        for im in frames:
            lg,v,state,_=tiny_model.forward_step(im,state)
            lo.append(lg);va.append(v)
    assert torch.allclose(full_logits,torch.stack(lo))
    assert torch.allclose(full_values,torch.stack(va))
    assert all(torch.equal(a,b) for a,b in zip(full_state.membranes,state.membranes))


def test_explicit_state_is_not_mutated(tiny_model):
    x=torch.rand(1,3,16,16)
    _,_,state,_=tiny_model.forward_step(x)
    snap=state.clone()
    _=tiny_model.forward_step(x,state)
    assert all(torch.equal(a,b) for a,b in zip(state.membranes,snap.membranes))
    assert all(torch.equal(a,b) for a,b in zip(state.spikes,snap.spikes))


def test_replay_rejects_wrong_shape(tiny_model):
    with pytest.raises(ValueError):tiny_model.replay(torch.rand(3,3,16,16))


def test_recurrent_state_reaches_future_frames(tiny_model):
    # Nonzero carried state should change future outputs or model's recurrent state.
    x=torch.ones(1,3,16,16)
    _,_,state1,_=tiny_model.forward_step(x)
    _,_,state2,_=tiny_model.forward_step(x,state1)
    _,_,reset_state,_=tiny_model.forward_step(x,None)
    assert not all(torch.equal(a,b) for a,b in zip(state2.membranes,reset_state.membranes))


def test_backward_through_convolutional_spikes(tiny_model):
    frames=torch.rand(2,1,3,16,16)
    logits,value,_,_=tiny_model.replay(frames)
    (logits.square().mean()+value.square().mean()).backward()
    assert tiny_model.convs[0].weight.grad is not None
    assert torch.isfinite(tiny_model.convs[0].weight.grad).all()
    assert tiny_model.recurrent.weight.grad is not None
