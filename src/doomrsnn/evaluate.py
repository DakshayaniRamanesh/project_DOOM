import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from .actions import Action
from .environment import DoomBasic
from .teacher import BasicTeacher


def wilson_interval(successes, n, z=1.959963984540054):
    """Wilson 95% interval for a binary success fraction."""
    if n <= 0: raise ValueError('Sample size must be positive')
    phat = successes / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n))/denom
    half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denom
    return float(center-half), float(center+half)


def evaluate_policy(cfg,model=None,checkpoint=None,policy='student',episodes=20,seed=30001,device='cpu',
                    mode='greedy',visible=False,output_path=None,return_frames=False):
    if policy=='student' and model is None:raise ValueError('Student policy requires a model')
    if policy=='student' and not checkpoint:raise ValueError('Student requires an explicit verified checkpoint')
    if policy=='student' and not Path(checkpoint).is_file():raise FileNotFoundError(f'Student checkpoint not found: {checkpoint}')
    if mode not in ('greedy','stochastic'): raise ValueError('mode must be greedy or stochastic')
    if policy not in ('student','teacher','random','always_shoot','always_left_shoot'):
        raise ValueError(f'Unsupported policy {policy}')
    rng=np.random.default_rng(seed)
    teacher=BasicTeacher()
    rows=[];samples=[]
    if model is not None:model.eval()
    with DoomBasic(cfg,labels=(policy=='teacher'),visible=visible) as env:
        for ep in range(int(episodes)):
            episode_seed=seed+ep*7
            obs=env.reset(seed=episode_seed)
            state=None; rewards=[]; actions=[]; spike_rates=[]; probs=[];frames=[]
            done=False
            while not done:
                if policy=='student':
                    with torch.no_grad():
                        inp=torch.as_tensor(obs,device=device).unsqueeze(0)
                        logits,value,state,diag=model.forward_step(inp,state)
                        dist=Categorical(logits=logits)
                        action=int((torch.argmax(logits,-1) if mode=='greedy' else dist.sample()).item())
                        probs.append(dist.probs[0].cpu().numpy().tolist())
                        spike_rates.append(float(diag['spike_rate'].item()))
                elif policy=='teacher':action=teacher.act(env).action
                elif policy=='random':action=int(rng.integers(0,6))
                elif policy=='always_shoot':action=int(Action.SHOOT)
                else:action=int(Action.LEFT_SHOOT)
                if return_frames and len(frames)<500:
                    raw=env.game.get_state().screen_buffer
                    raw=np.asarray(raw)
                    frames.append(raw.transpose(1,2,0).copy() if raw.ndim==3 and raw.shape[0]==3 else raw.copy())
                obs,reward,done,info=env.step(action)
                actions.append(action);rewards.append(reward)
            rows.append({'policy':policy,'mode':mode,'episode':ep,'seed':episode_seed,
                         'kill':int(env.game.get_game_variable(env.vzd.GameVariable.KILLCOUNT)>0),
                         'reward':float(sum(rewards)),'decisions':len(actions),
                         'spike_rate':float(np.mean(spike_rates)) if spike_rates else 0.0,
                         'shots':int(sum(a in (3,4,5) for a in actions)),
                         **{f'action_{i}':int(actions.count(i)) for i in range(6)}})
            if return_frames:samples.append(frames)
    if output_path:
        output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
        with output_path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    success=sum(r['kill'] for r in rows)
    interval=wilson_interval(success,len(rows))
    summary={'policy':policy,'episodes':len(rows),'kills':success,'kill_rate':success/len(rows),
             'kill_wilson95_low':interval[0],'kill_wilson95_high':interval[1],
             'mean_reward':float(np.mean([r['reward'] for r in rows])),
             'mean_decisions':float(np.mean([r['decisions'] for r in rows])),
             'mean_spike_rate':float(np.mean([r['spike_rate'] for r in rows]))}
    return summary,rows,samples
