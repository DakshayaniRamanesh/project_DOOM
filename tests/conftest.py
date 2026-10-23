import pytest
from doomrsnn.model import ConvRSNN

@pytest.fixture
def small_cfg():
    return {'seed':42,'learning_rate':1e-3,'max_grad_norm':0.5,'action_repeat':4,
            'imitation':{'sequence_length':2,'val_fraction':0.5},
            'rl':{'algorithm':'ppo','rollout_steps':2,'gamma':0.99,'gae_lambda':0.95,
                  'ppo_epochs':2,'clip_epsilon':0.2,'entropy_coefficient':0.01,
                  'value_coefficient':0.5,'reward_scale':0.01},
            'model':{'in_channels':3,'height':16,'width':16,'conv_channels':[4,6,8],
                     'pool_hw':[1,2],'hidden_size':12,'timesteps':2,'beta':0.5,
                     'threshold':0.4,'surrogate_slope':5.0,'actions':6}}

@pytest.fixture
def tiny_model(small_cfg):
    return ConvRSNN(small_cfg['model'])
