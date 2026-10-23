from pathlib import Path
import torch


def save_checkpoint(path,model,cfg,stage,optimizer=None,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content={'format':1,'stage':str(stage),'config':cfg,'model':model.state_dict(),
             'optimizer': None if optimizer is None else optimizer.state_dict(), 'extra':extra or {}}
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(content,tmp)
    tmp.replace(path)


def load_checkpoint(path,model,device='cpu',optimizer=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Checkpoint missing: {path}')
    content=torch.load(path,map_location=device,weights_only=False)
    if content.get('format')!=1:raise ValueError('Unsupported checkpoint format')
    saved=content['config']['model']
    if saved!=model.config:
        raise ValueError(f'Model config mismatch; saved={saved}, current={model.config}')
    model.load_state_dict(content['model'], strict=True)
    if optimizer is not None:
        if content.get('optimizer') is None:raise ValueError('No optimizer state to resume')
        optimizer.load_state_dict(content['optimizer'])
    return content
