"""Single CLI: python -m doomrsnn.cli <command> --config configs/base.yaml ..."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from .config import load_config
from .seeding import seed_all,torch_device
from .model import ConvRSNN
from .checkpoints import load_checkpoint
from .dataset import dataset_report


def parser():
    p=argparse.ArgumentParser(description='Convolutional LIF-RSNN ViZDoom project')
    p.add_argument('--config',default='configs/base.yaml',help='YAML configuration file')
    p.add_argument('--device',default=None,help='auto, cpu, cuda, cuda:0, etc.')
    p.add_argument('--scenario',default=None,help='Path to basic.cfg')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('smoke',help='Synthetic model forward/backward/serialization check')
    sub.add_parser('inspect-env',help='Run one real ViZDoom reset and report labels')
    collect=sub.add_parser('collect',help='Collect teacher-controlled demonstrations')
    collect.add_argument('--episodes',type=int,default=3);collect.add_argument('--seed',type=int,default=11001)
    collect.add_argument('--output',default='data/teacher')
    inspect=sub.add_parser('inspect-data',help='Summarize a demonstration folder')
    inspect.add_argument('--data',required=True)
    bc=sub.add_parser('bc',help='Behavioral cloning from teacher and/or DAgger trajectories')
    bc.add_argument('--data',default='data/teacher');bc.add_argument('--additional-data',nargs='*',default=[])
    bc.add_argument('--epochs',type=int,default=None);bc.add_argument('--checkpoint',default=None)
    bc.add_argument('--output',default='checkpoints/bc_best.pt')
    dagger=sub.add_parser('dagger',help='DAgger collection and retraining iterations')
    dagger.add_argument('--checkpoint',required=True);dagger.add_argument('--teacher-data',default='data/teacher')
    dagger.add_argument('--iterations',type=int,default=2);dagger.add_argument('--episodes-per-iteration',type=int,default=20)
    dagger.add_argument('--epochs-per-iteration',type=int,default=5);dagger.add_argument('--output-dir',default='data/dagger')
    rl=sub.add_parser('rl',help='Sequence-aware A2C/PPO fine-tuning')
    rl.add_argument('--checkpoint',required=True);rl.add_argument('--episodes',type=int,default=100)
    rl.add_argument('--output',default='checkpoints/rl_latest.pt')
    ev=sub.add_parser('evaluate',help='Real ViZDoom benchmark')
    ev.add_argument('--checkpoint',default=None);ev.add_argument('--episodes',type=int,default=100)
    ev.add_argument('--seed',type=int,default=900001)
    ev.add_argument('--policies',nargs='+',default=['student','random','always_shoot','always_left_shoot'])
    ev.add_argument('--mode',choices=('greedy','stochastic'),default='greedy')
    ev.add_argument('--output-dir',default='runs/evaluation');ev.add_argument('--visible',action='store_true')
    record=sub.add_parser('record',help='Record trained autonomous gameplay as video')
    record.add_argument('--checkpoint',required=True);record.add_argument('--seed',type=int,default=990001)
    record.add_argument('--episodes',type=int,default=3);record.add_argument('--output',default='runs/demo.mp4')
    return p


def main():
    args=parser().parse_args()
    cfg=load_config(args.config)
    if args.scenario:cfg['scenario']=args.scenario
    if args.device:cfg['device']=args.device
    device=torch_device(cfg.get('device','auto'))
    seed_all(int(cfg['seed']))
    if args.cmd=='inspect-data':
        print(json.dumps(dataset_report(args.data),indent=2));return
    if args.cmd=='inspect-env':
        from .environment import DoomBasic
        from .teacher import BasicTeacher
        with DoomBasic(cfg,labels=True) as env:
            observation=env.reset(seed=cfg['seed'])
            state=env.label_state()
            print('Observation',observation.shape,'target labels:',[
                {'name':str(x.object_name),'x':int(x.x),'width':int(x.width)} for x in state.labels])
            print('Teacher recommendation:',BasicTeacher().act(env));return
    if args.cmd=='collect':
        from .train import collect_expert
        collect_expert(cfg,args.output,args.episodes,args.seed);return
    # Smaller architecture for local smoke; no checkpoint created for actual gameplay.
    if args.cmd=='smoke':
        import copy
        small=copy.deepcopy(cfg['model'])
        small.update({'height':16,'width':16,'conv_channels':[4,8,8],
                      'pool_hw':[1,2],'hidden_size':12,'timesteps':2})
        m=ConvRSNN(small).to(device)
        inp=torch.rand(3,1,3,16,16,device=device)
        logits,vals,_,rates=m.replay(inp)
        loss=logits.square().mean()+vals.square().mean()+0.1*logits.mean()
        loss.backward()
        grad=sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
        print(json.dumps({'result':'synthetic smoke passed','logits_shape':list(logits.shape),
                         'values_shape':list(vals.shape),'loss':float(loss.item()),
                         'parameters_with_finite_gradients':int(grad),'total_parameters':len(list(m.parameters())),
                         'model_stats':m.model_stats()},indent=2));return
    model=ConvRSNN(cfg['model']).to(device)
    print('Device:',device,'Model:',json.dumps(model.model_stats()))
    if args.cmd=='bc':
        from .train import train_imitation
        if args.checkpoint:load_checkpoint(args.checkpoint,model,device)
        train_imitation(cfg,model,args.data,args.epochs or cfg['imitation']['epochs'],device,
                        args.output,args.additional_data);return
    if args.cmd=='dagger':
        from .train import collect_dagger,train_imitation
        load_checkpoint(args.checkpoint,model,device)
        folders=[]
        for iteration in range(args.iterations):
            folder=str(Path(args.output_dir)/f'round_{iteration+1:02d}')
            collect_dagger(cfg,model,folder,args.episodes_per_iteration,120001+iteration*10000,device)
            folders.append(folder)
            ckpt=str(Path('checkpoints')/f'dagger_round_{iteration+1:02d}.pt')
            train_imitation(cfg,model,args.teacher_data,args.epochs_per_iteration,device,ckpt,folders)
            load_checkpoint(ckpt,model,device)
        return
    if args.cmd=='rl':
        from .train import train_rl
        load_checkpoint(args.checkpoint,model,device)
        train_rl(cfg,model,device,args.episodes,args.output);return
    if args.cmd=='evaluate':
        from .evaluate import evaluate_policy
        out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
        for policy in args.policies:
            if policy=='student':
                if not args.checkpoint:raise ValueError('--checkpoint is required for student evaluation')
                load_checkpoint(args.checkpoint,model,device)
            summary,_,_=evaluate_policy(cfg,model if policy=='student' else None,
                checkpoint=args.checkpoint if policy=='student' else None,policy=policy,
                episodes=args.episodes,seed=args.seed,device=device,mode=args.mode,
                visible=args.visible,output_path=out/f'{policy}.csv')
            print(json.dumps(summary),flush=True)
        return
    if args.cmd=='record':
        import imageio.v2 as imageio
        from .evaluate import evaluate_policy
        load_checkpoint(args.checkpoint,model,device)
        summary,_,episodes=evaluate_policy(cfg,model=model,checkpoint=args.checkpoint,
            policy='student',episodes=args.episodes,seed=args.seed,device=device,
            return_frames=True)
        frames=[f for ep in episodes for f in ep]
        if not frames:raise RuntimeError('No gameplay frames were captured')
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        with imageio.get_writer(output,fps=12,codec='libx264',macro_block_size=1) as writer:
            for frame in frames:writer.append_data(frame)
        print(json.dumps({'video':str(output),'captured_frames':len(frames),'evaluation':summary},indent=2))
        return
    raise RuntimeError('Unhandled command')

if __name__=='__main__':main()
