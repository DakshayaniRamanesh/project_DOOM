"""Per-episode trajectories with immutable observational labels and metadata."""
from pathlib import Path
import json
import numpy as np


def save_episode(folder,episode_id,observations,teacher_actions,executed_actions,rewards,metadata):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    obs=np.asarray(observations,dtype=np.float32)
    expert=np.asarray(teacher_actions,dtype=np.int64)
    executed=np.asarray(executed_actions,dtype=np.int64)
    rewards=np.asarray(rewards,dtype=np.float32)
    if obs.ndim!=4 or obs.shape[1]!=3:
        raise ValueError(f'Observations must have [T,3,H,W], got {obs.shape}')
    if len(obs)==0 or not (len(obs)==len(expert)==len(executed)==len(rewards)):
        raise ValueError('All trajectory arrays must have equal nonzero time length')
    if not np.isfinite(obs).all() or (obs < 0).any() or (obs > 1).any():
        raise ValueError('Observations must be finite and normalized')
    if (expert<0).any() or (expert>=6).any() or (executed<0).any() or (executed>=6).any():
        raise ValueError('Action index outside canonical six-action mapping')
    path=folder/f'episode_{int(episode_id):06d}.npz'
    if path.exists():raise FileExistsError(f'Will not overwrite {path}')
    np.savez_compressed(path, images=np.rint(obs*255).astype(np.uint8),
                        teacher_actions=expert,executed_actions=executed,rewards=rewards,
                        metadata=np.array(json.dumps(metadata,sort_keys=True)))
    return path


def load_episode(path):
    with np.load(path,allow_pickle=False) as data:
        images=data['images'].astype(np.float32)/255.0
        label=data['teacher_actions'].astype(np.int64)
        actions=data['executed_actions'].astype(np.int64)
        rewards=data['rewards'].astype(np.float32)
        meta=json.loads(str(data['metadata'].item()))
    if len(images)==0 or not(len(images)==len(label)==len(actions)==len(rewards)):
        raise ValueError(f'Corrupt episode {path}')
    return {'images':images,'labels':label,'actions':actions,'rewards':rewards,'metadata':meta}


def episodes(folder):
    paths=sorted(Path(folder).glob('*.npz'))
    if not paths: raise FileNotFoundError(f'No episode .npz files found in {folder}')
    return paths


def episode_split(paths,val_fraction=0.15,seed=42):
    paths=list(paths)
    if len(paths)<2:raise ValueError('At least two episodes are needed for train/validation split')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(paths))
    count=max(1,int(round(len(paths)*val_fraction)))
    count=min(len(paths)-1,count)
    return [paths[i] for i in order[count:]],[paths[i] for i in order[:count]]


def dataset_report(folder):
    paths=episodes(folder)
    counts=np.zeros(6,np.int64);n=0;success=0
    for p in paths:
        e=load_episode(p)
        n+=len(e['labels'])
        counts+=np.bincount(e['labels'],minlength=6)
        success+=bool(e['metadata'].get('kill',False))
    return {'episodes':len(paths),'decisions':int(n),'teacher_label_counts':counts.tolist(),
            'recorded_success_rate':success/len(paths)}
