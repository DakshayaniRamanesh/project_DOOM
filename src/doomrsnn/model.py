"""Convolutional spiking visual encoder and explicitly stateful recurrent LIF actor-critic.

No internal mutable recurrent state: forward_step returns a new state. This enables
correct sequence replay and prevents a bootstrap-value query from modifying policy state.
"""
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .neurons import lif_step


@dataclass
class State:
    membranes: Tuple[torch.Tensor, ...]  # conv1, conv2, conv3, recurrent
    spikes: Tuple[torch.Tensor, ...]

    def detach(self):
        return State(tuple(t.detach() for t in self.membranes), tuple(t.detach() for t in self.spikes))

    def clone(self):
        return State(tuple(t.clone() for t in self.membranes), tuple(t.clone() for t in self.spikes))


class ConvRSNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        cin = int(config['in_channels'])
        channels = [int(x) for x in config['conv_channels']]
        if len(channels) != 3:
            raise ValueError('Exactly three conv channels are required')
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, stride=2, padding=1)
                                    for a,b in zip([cin]+channels[:-1],channels)])
        self.spatial_pool = tuple(config['pool_hw'])
        self.hidden = int(config['hidden_size'])
        self.timesteps = int(config['timesteps'])
        self.beta = float(config['beta'])
        self.threshold = float(config['threshold'])
        self.slope = float(config['surrogate_slope'])
        self.input_projection = nn.Linear(channels[-1] * self.spatial_pool[0] * self.spatial_pool[1], self.hidden)
        self.recurrent = nn.Linear(self.hidden, self.hidden, bias=False)
        self.actor = nn.Linear(self.hidden, int(config['actions']))
        self.critic = nn.Linear(self.hidden, 1)
        self._shapes = self._conv_shapes(int(config['height']), int(config['width']),channels)
        for module in self.convs:
            nn.init.kaiming_uniform_(module.weight, a=0.5)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.5)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    @staticmethod
    def _conv_shapes(h,w,channels):
        result=[]
        for c in channels:
            h=(h+1)//2
            w=(w+1)//2
            result.append((c,h,w))
        return result

    def initial_state(self, batch, device, dtype=torch.float32):
        shapes = [(batch,*shape) for shape in self._shapes] + [(batch,self.hidden)]
        membranes = tuple(torch.zeros(shape,device=device,dtype=dtype) for shape in shapes)
        spikes = tuple(torch.zeros_like(x) for x in membranes)
        return State(membranes, spikes)

    def forward_step(self, image, state: State | None = None):
        if image.ndim != 4 or image.shape[1:] != (self.config['in_channels'],self.config['height'],self.config['width']):
            raise ValueError(f'Expected [B,{self.config["in_channels"]},{self.config["height"]},{self.config["width"]}], received {tuple(image.shape)}')
        if state is None:
            state=self.initial_state(image.shape[0],image.device,image.dtype)
        if state.membranes[-1].shape[0] != image.shape[0]:
            raise ValueError('Recurrent state has an incorrect batch size')
        membrane=list(state.membranes)
        spike=list(state.spikes)
        aggregate=torch.zeros((image.shape[0],self.hidden),device=image.device,dtype=image.dtype)
        spike_counts = torch.zeros_like(aggregate)
        for _ in range(self.timesteps):
            signal=image
            for i, conv in enumerate(self.convs):
                current=conv(signal)
                spike[i], membrane[i]=lif_step(current,membrane[i],spike[i],self.beta,self.threshold,self.slope)
                signal=spike[i]
            pooled=F.adaptive_avg_pool2d(signal, self.spatial_pool).flatten(1)
            current=self.input_projection(pooled) + self.recurrent(spike[3])
            spike[3],membrane[3]=lif_step(current,membrane[3],spike[3],self.beta,self.threshold,self.slope)
            aggregate=aggregate+spike[3]
            spike_counts=spike_counts+spike[3].detach()
        features=aggregate/self.timesteps
        logits=self.actor(features)
        values=self.critic(features).squeeze(-1)
        diagnostic={'spike_rate': spike_counts.mean(dim=-1)/self.timesteps, 'feature':features.detach()}
        return logits,values,State(tuple(membrane),tuple(spike)),diagnostic

    def replay(self, images, state=None):
        """Process [time,batch,channels,height,width] in temporal order."""
        if images.ndim != 5:
            raise ValueError('Expected [time,batch,C,H,W]')
        logits=[];values=[]; rates=[]
        for frame in images.unbind(0):
            lg,v,state,d=self.forward_step(frame,state)
            logits.append(lg); values.append(v);rates.append(d['spike_rate'])
        return torch.stack(logits),torch.stack(values),state,torch.stack(rates)

    def model_stats(self):
        return {'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'spiking_neurons':sum(c*h*w for c,h,w in self._shapes)+self.hidden,
                'hidden_neurons':self.hidden,
                'timesteps':self.timesteps}
