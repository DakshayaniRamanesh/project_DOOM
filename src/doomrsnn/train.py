"""Behavior cloning, DAgger collection and sequence-correct recurrent PPO/A2C.

Training losses are always re-evaluated by replaying trajectories in temporal order
starting with the saved state. No batch-of-frames shortcuts or hidden state mutation.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .dataset import episodes,load_episode,save_episode,episode_split
from .checkpoints import save_checkpoint
from .environment import DoomBasic
from .teacher import BasicTeacher
from .seeding import seed_all


def device_tensor(image,device):
    return torch.as_tensor(image,device=device,dtype=torch.float32).unsqueeze(0)


def collect_expert(cfg,folder,count=3,base_seed=10000):
    teacher=BasicTeacher()
    paths=[]
    with DoomBasic(cfg,labels=True) as env:
        for episode in range(count):
            obs=env.reset(seed=base_seed+episode*7)
            all_obs=[]; labels=[];actions=[]; rewards=[];done=False
            while not done:
                advice=teacher.act(env)
                action=advice.action
                all_obs.append(obs); labels.append(action); actions.append(action)
                obs,reward,done,info=env.step(action);rewards.append(reward)
            kill=bool(env.game.get_game_variable(env.vzd.GameVariable.KILLCOUNT)>0)
            paths.append(save_episode(folder,base_seed+episode*7,all_obs,labels,actions,rewards,
                                      {'mode':'teacher','seed':base_seed+episode*7,'kill':kill,
                                       'episode_reward':sum(rewards),'action_repeat':cfg['action_repeat']}))
            print(f'Teacher episode {episode+1}/{count}: kill={kill} reward={sum(rewards):.1f}, decisions={len(actions)}',flush=True)
    return paths


def collect_dagger(cfg,model,folder,count,base_seed,device='cpu',mode='greedy'):
    teacher=BasicTeacher();paths=[];model.eval()
    with DoomBasic(cfg,labels=True) as env:
        for episode in range(count):
            epseed=base_seed+episode*7
            obs=env.reset(seed=epseed)
            state=None;all_obs=[];expert=[];actions=[];rewards=[]; done=False
            while not done:
                advice=teacher.act(env)  # read-only query at SAME state
                with torch.no_grad():
                    logits,_,state,_=model.forward_step(device_tensor(obs,device),state)
                    dist=Categorical(logits=logits)
                    action=int((dist.probs.argmax(-1) if mode=='greedy' else dist.sample()).item())
                all_obs.append(obs);expert.append(advice.action);actions.append(action)
                obs,reward,done,info=env.step(action);rewards.append(reward)
            kill=bool(env.game.get_game_variable(env.vzd.GameVariable.KILLCOUNT)>0)
            paths.append(save_episode(folder,epseed,all_obs,expert,actions,rewards,
                        {'mode':'dagger','seed':epseed,'kill':kill,'episode_reward':sum(rewards),
                         'action_repeat':cfg['action_repeat'],
                         'student_teacher_agreement':float(np.mean(np.asarray(actions)==np.asarray(expert)))}))
            print(f'DAgger {episode+1}/{count}: kill={kill}, agreement={np.mean(np.asarray(actions)==np.asarray(expert)):.2%}',flush=True)
    return paths


def _clone_episode_train(model,opt,episode,device,sequence_length,train=True,max_grad_norm=0.5):
    """Per-episode streaming TBPTT: carry numerical state; detach graphs at chunk boundaries."""
    frames=episode['images']; labels=episode['labels']; state=None
    total_loss=0.0; correct=0; count=0
    model.train(train)
    for start in range(0,len(labels),sequence_length):
        end=min(start+sequence_length,len(labels))
        imgs=torch.as_tensor(frames[start:end],dtype=torch.float32,device=device).unsqueeze(1)
        y=torch.as_tensor(labels[start:end],dtype=torch.long,device=device)
        with torch.set_grad_enabled(train):
            logits,_,state,_=model.replay(imgs,state)
            lg=logits[:,0,:]
            loss=F.cross_entropy(lg,y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
                opt.step()
        total_loss+=loss.item()*(end-start)
        correct+=int((lg.argmax(-1)==y).sum().item()); count+=end-start
        state=state.detach()  # numerical memory preserved across chunks
    return total_loss,max(count,1),correct


def train_imitation(cfg,model,dataset_folder,epochs,device='cpu',output='checkpoints/bc_best.pt',extra_folders=None):
    paths=episodes(dataset_folder)
    for folder in extra_folders or []:
        paths+=episodes(folder)
    train_paths,val_paths=episode_split(paths,float(cfg['imitation']['val_fraction']),int(cfg['seed']))
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg['learning_rate']),weight_decay=float(cfg.get('weight_decay',0)))
    rng=np.random.default_rng(cfg['seed']);best=float('inf')
    history=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(train_paths))
        sums=np.zeros(3)
        for idx in order:
            loss,count,correct=_clone_episode_train(model,opt,load_episode(train_paths[int(idx)]),device,
                int(cfg['imitation']['sequence_length']),True,float(cfg['max_grad_norm']))
            sums+=loss,count,correct
        val=np.zeros(3)
        for p in val_paths:
            loss,count,correct=_clone_episode_train(model,opt,load_episode(p),device,
                int(cfg['imitation']['sequence_length']),False)
            val+=loss,count,correct
        row={'epoch':epoch,'train_loss':float(sums[0]/sums[1]),'train_accuracy':float(sums[2]/sums[1]),
             'val_loss':float(val[0]/val[1]),'val_accuracy':float(val[2]/val[1])}
        history.append(row);print(json.dumps(row),flush=True)
        if row['val_loss']<best:
            best=row['val_loss']
            save_checkpoint(output,model,cfg,'behavioral_cloning',opt,{'epoch':epoch,'validation_loss':best})
    result=Path(output).with_suffix('.history.json');result.parent.mkdir(parents=True,exist_ok=True)
    result.write_text(json.dumps(history,indent=2),encoding='utf-8')
    return history


def gae_advantages(rewards,values,next_value,done,gamma,lam):
    """Numpy GAE. done marks true terminal end of current recorded segment."""
    rewards=np.asarray(rewards,dtype=np.float64)
    values=np.asarray(values,dtype=np.float64)
    done=np.asarray(done,dtype=np.bool_)
    if not(len(rewards)==len(values)==len(done)):raise ValueError('Mismatched trajectory fields')
    advantages=np.zeros_like(rewards); running=0.0
    for t in range(len(rewards)-1,-1,-1):
        nonterminal=not done[t]
        next_v=float(next_value) if t==len(rewards)-1 else values[t+1]
        delta=rewards[t]+gamma*next_v*nonterminal-values[t]
        running=delta+gamma*lam*nonterminal*running
        advantages[t]=running
    return advantages,advantages+values


def collect_rl_rollout(env,model,obs,state,device,rollout_steps,reward_scale=0.01):
    """Return a temporally ordered on-policy rollout and uncommitted initial state."""
    initial_state=None if state is None else state.detach().clone()
    trajectory={'observations':[],'actions':[],'old_logprobs':[],'values':[],
                'rewards':[],'dones':[],'raw_rewards':[]}
    model.eval();done=False
    for _ in range(rollout_steps):
        with torch.no_grad():
            logits,v,state,_=model.forward_step(device_tensor(obs,device),state)
            dist=Categorical(logits=logits)
            action=dist.sample()
            trajectory['observations'].append(obs)
            trajectory['actions'].append(int(action.item()))
            trajectory['old_logprobs'].append(float(dist.log_prob(action).item()))
            trajectory['values'].append(float(v.item()))
        next_obs,reward,done,info=env.step(int(action.item()))
        trajectory['rewards'].append(float(reward)*reward_scale)
        trajectory['raw_rewards'].append(float(reward))
        trajectory['dones'].append(done)
        obs=next_obs
        if done:break
    if done:
        next_v=0.0
    else:
        with torch.no_grad():
            # Pure explicit-state call. This does NOT advance policy rollout state.
            _,next_val,_,_=model.forward_step(device_tensor(obs,device),state)
            next_v=float(next_val.item())
    trajectory['next_value']=next_v
    return trajectory,obs,state.detach(),initial_state,done,info


def update_actor_critic(cfg,model,opt,trajectory,initial_state,device):
    """Sequence-correct on-policy PPO: replay whole ordered rollout for each epoch.

    Never interpret time as a batch dimension. No minibatch shuffling.
    Episode lengths <= rollout_steps; forward replay starts at saved initial state.
    """
    rl=cfg['rl']; gamma=float(rl['gamma']); lam=float(rl['gae_lambda'])
    advantages,returns=gae_advantages(trajectory['rewards'],trajectory['values'],trajectory['next_value'],
                                      trajectory['dones'],gamma,lam)
    advantages=torch.as_tensor(advantages,dtype=torch.float32,device=device)
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8) if len(advantages)>1 else advantages
    ret=torch.as_tensor(returns,dtype=torch.float32,device=device)
    acts=torch.as_tensor(trajectory['actions'],device=device,dtype=torch.long)
    old_lp=torch.as_tensor(trajectory['old_logprobs'],device=device,dtype=torch.float32)
    frames=torch.as_tensor(np.stack(trajectory['observations']),device=device,dtype=torch.float32).unsqueeze(1)
    algo=rl.get('algorithm','ppo').lower()
    steps=int(rl.get('ppo_epochs',3)) if algo=='ppo' else 1
    if algo not in ('ppo','a2c'):raise ValueError('Only ppo and a2c supported')
    losses=[]
    for _ in range(steps):
        model.train()
        state=None if initial_state is None else initial_state.detach().clone()
        logits,vals,_,rates=model.replay(frames,state)
        logits=logits[:,0,:];vals=vals[:,0]
        dist=Categorical(logits=logits)
        lp=dist.log_prob(acts)
        if algo=='ppo':
            ratio=(lp-old_lp).exp()
            clip=float(rl.get('clip_epsilon',0.2))
            pg_loss=-torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages).mean()
        else:pg_loss=-(lp*advantages).mean()
        v_loss=F.smooth_l1_loss(vals,ret)
        entropy=dist.entropy().mean()
        loss=pg_loss+float(rl['value_coefficient'])*v_loss-float(rl['entropy_coefficient'])*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite RL loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['max_grad_norm']))
        opt.step()
        losses.append({'actor_loss':float(pg_loss.item()),'value_loss':float(v_loss.item()),
                       'entropy':float(entropy.item()),'spike_rate':float(rates.mean().item())})
    return losses[-1]


def train_rl(cfg,model,device,episodes_count,output='checkpoints/rl_latest.pt',start_episode=0,optimizer=None):
    opt=optimizer or torch.optim.Adam(model.parameters(),lr=float(cfg['learning_rate']))
    rl=cfg['rl'];log=[];validation_every=25;best_val=(-float('inf'),-float('inf'))
    # Checkpoint selection based on fixed validation seeds, NEVER rolling training reward.
    from .evaluate import evaluate_policy
    with DoomBasic(cfg,labels=False) as env:
        for ep in range(start_episode,start_episode+episodes_count):
            obs=env.reset(seed=int(cfg['seed'])+ep*7)
            state=None;done=False;episode_reward=0.0;episode_steps=0;last_info={}
            while not done:
                tr,obs,state,initial,done,last_info=collect_rl_rollout(env,model,obs,state,device,
                    int(rl['rollout_steps']),float(rl['reward_scale']))
                metrics=update_actor_critic(cfg,model,opt,tr,initial,device)
                episode_reward+=sum(tr['raw_rewards']);episode_steps+=len(tr['actions'])
            kill=int(env.game.get_game_variable(env.vzd.GameVariable.KILLCOUNT)>0)
            row={'episode':ep+1,'reward':episode_reward,'decisions':episode_steps,'kill':kill,**metrics}
            log.append(row);print(json.dumps(row),flush=True)
            save_checkpoint(output,model,cfg,'rl',opt,{'episode':ep+1,'latest_metrics':row})
            if (ep+1)%validation_every==0:
                # Avoid unsupported use of a fake checkpoint by evaluator: temporary real checkpoint.
                snapshot=str(Path(output).with_name('validation_snapshot.pt'))
                save_checkpoint(snapshot,model,cfg,'validation_snapshot')
                s,_,_=evaluate_policy(cfg,model=model,checkpoint=snapshot,policy='student',episodes=20,
                       seed=400001,device=device,mode='greedy')
                key=(s['kill_rate'],s['mean_reward'])
                if key>best_val:
                    best_val=key
                    save_checkpoint(str(Path(output).with_name('rl_best.pt')),model,cfg,'rl_best',opt,
                                    {'episode':ep+1,'validation':s})
                print('Validation:',json.dumps(s),flush=True)
    path=Path(output).with_suffix('.trainlog.json');path.write_text(json.dumps(log,indent=2))
    return log
