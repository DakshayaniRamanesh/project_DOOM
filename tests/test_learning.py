import numpy as np
import torch
import pytest
from doomrsnn.train import gae_advantages,_clone_episode_train,update_actor_critic,collect_rl_rollout
from doomrsnn.checkpoints import save_checkpoint,load_checkpoint


def test_gae_single_terminal():
    adv,ret=gae_advantages([10.0],[3.0],next_value=100,done=[True],gamma=.99,lam=.95)
    assert adv.tolist()==pytest.approx([7.0])
    assert ret.tolist()==pytest.approx([10.0])


def test_gae_nonterminal_bootstrap():
    adv,ret=gae_advantages([1.0],[2.0],next_value=5.0,done=[False],gamma=.9,lam=.95)
    assert adv.tolist()==pytest.approx([3.5]); assert ret.tolist()==pytest.approx([5.5])


def test_gae_two_step():
    adv,ret=gae_advantages([-4,99],[40,60],0,[False,True],.99,.95)
    assert adv.tolist()==pytest.approx([15.4+.99*.95*39,39]);assert ret[0]>0


def test_behavior_clone_training_on_sequence(tiny_model):
    opt=torch.optim.Adam(tiny_model.parameters(),lr=1e-3)
    ep={'images':np.random.default_rng(3).random((3,3,16,16),dtype=np.float32),
        'labels':np.array([1,2,3])}
    loss,n,correct=_clone_episode_train(tiny_model,opt,ep,'cpu',2,True)
    assert np.isfinite(loss) and n==3 and 0<=correct<=3


def test_ppo_update_uses_temporal_replay(tiny_model,small_cfg):
    model=tiny_model
    rng=np.random.default_rng(3)
    obs=rng.random((3,3,16,16),dtype=np.float32)
    with torch.no_grad():
        logits,values,_,_=model.replay(torch.tensor(obs).unsqueeze(1))
        dist=torch.distributions.Categorical(logits=logits[:,0,:])
        actions=dist.sample()
    tr={'observations':list(obs),'actions':actions.tolist(),
        'old_logprobs':dist.log_prob(actions).tolist(),
        'values':values[:,0].tolist(),'rewards':[0,0,1],
        'dones':[False,False,True],'next_value':0.0}
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    report=update_actor_critic(small_cfg,model,opt,tr,None,'cpu')
    assert np.isfinite(report['actor_loss'])
    assert np.isfinite(report['value_loss'])
    assert report['entropy']>0


def test_checkpoint_roundtrip_and_mismatch(tiny_model,small_cfg,tmp_path):
    path=tmp_path/'model.pt'
    x=torch.rand(1,3,16,16)
    with torch.no_grad():expected=tiny_model.forward_step(x)[0].clone()
    save_checkpoint(path,tiny_model,small_cfg,'smoke')
    for p in tiny_model.parameters():p.data.add_(0.1)
    load_checkpoint(path,tiny_model)
    with torch.no_grad():actual=tiny_model.forward_step(x)[0]
    assert torch.allclose(expected,actual)
    # A different architecture must reject the checkpoint.
    from doomrsnn.model import ConvRSNN
    altered={**small_cfg['model'], 'timesteps':3}
    other=ConvRSNN(altered)
    with pytest.raises(ValueError):load_checkpoint(path,other)


def test_missing_checkpoint(tiny_model,tmp_path):
    with pytest.raises(FileNotFoundError):load_checkpoint(tmp_path/'missing.pt',tiny_model)

class FakeEnv:
    def __init__(self):self.n=0
    def step(self,action):
        self.n+=1
        return np.ones((3,16,16),dtype=np.float32)*(self.n/10),-1.0,self.n==2,{'episode_time':4*self.n}


def test_rollout_and_explicit_bootstrap(tiny_model):
    env=FakeEnv();obs=np.zeros((3,16,16),dtype=np.float32)
    t,next_obs,state,initial,done,info=collect_rl_rollout(env,tiny_model,obs,None,'cpu',5)
    assert done and len(t['actions'])==2
    assert t['next_value']==0.0 and next_obs is not None


def test_wilson_interval():
    from doomrsnn.evaluate import wilson_interval
    lo,hi=wilson_interval(81,100)
    assert 0 < lo < .81 < hi < 1
    assert wilson_interval(0,100)[0] >= 0
    assert wilson_interval(100,100)[1] <= 1.00000000001
