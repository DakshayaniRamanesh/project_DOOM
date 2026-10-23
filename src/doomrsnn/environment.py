"""Real ViZDoom wrapper; intentionally no dummy-environment fallback."""
from pathlib import Path
import numpy as np
from PIL import Image
from .actions import BUTTON_NAMES,button_vector


def default_scenario():
    try:
        import vizdoom
    except ImportError as e:
        raise RuntimeError('ViZDoom is not installed in this Python environment') from e
    packaged=getattr(vizdoom,'scenarios_path',None)
    candidates=[]
    if packaged is not None: candidates.append(Path(packaged)/'basic.cfg')
    candidates.append(Path(vizdoom.__file__).resolve().parent/'scenarios'/'basic.cfg')
    for candidate in candidates:
        if candidate.exists(): return candidate
    raise FileNotFoundError(f'No ViZDoom BASIC config found in {candidates}; provide --scenario')


def image_tensor(screen, height, width):
    """Return RGB CHW float32 array in [0,1] from CHW or HWC ViZDoom buffers."""
    if screen is None:
        raise ValueError('Cannot preprocess a missing screen buffer')
    arr=np.asarray(screen)
    if arr.ndim==3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
        arr=arr.transpose(1,2,0)
    if arr.ndim==2: arr=np.repeat(arr[:,:,None],3,axis=2)
    if arr.ndim!=3: raise ValueError(f'Unsupported screen shape {arr.shape}')
    if arr.shape[-1]==1:arr=np.repeat(arr,3,axis=2)
    if arr.shape[-1]==4:arr=arr[:,:,:3]
    if arr.shape[-1]!=3: raise ValueError(f'Expected 3 RGB channels, got {arr.shape}')
    if arr.dtype!=np.uint8:
        arr=np.clip(arr,0,255).astype(np.uint8)
    rgb=np.asarray(Image.fromarray(arr).resize((width,height),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
    return np.ascontiguousarray(rgb.transpose(2,0,1))


class DoomBasic:
    def __init__(self,cfg,labels=False,visible=None):
        self.cfg=cfg;self.labels=labels;self.visible=cfg.get('visible',False) if visible is None else visible
        self.game=None
        self.seed=None

    def __enter__(self):
        try:
            import vizdoom as vzd
        except ImportError as e:
            raise RuntimeError('Install vizdoom and point --scenario to basic.cfg') from e
        self.vzd=vzd
        scenario=Path(self.cfg.get('scenario') or default_scenario()).expanduser().resolve()
        if not scenario.exists(): raise FileNotFoundError(scenario)
        self.game=vzd.DoomGame()
        self.game.load_config(str(scenario))
        self.game.set_window_visible(bool(self.visible))
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.clear_available_buttons()
        for name in BUTTON_NAMES:self.game.add_available_button(getattr(vzd.Button,name))
        self.game.set_episode_timeout(int(self.cfg.get('episode_timeout',300)))
        if self.labels:
            self.game.set_labels_buffer_enabled(True)
        self.game.init()
        return self

    def __exit__(self,*args):
        if self.game:self.game.close()
        self.game=None

    def reset(self,seed=None):
        if seed is not None:
            self.seed=int(seed)
            # ViZDoom seeds are game-engine implementation-specific; distinct seeds must be verified externally.
            self.game.set_seed(self.seed)
        self.game.new_episode()
        state=self.game.get_state()
        if state is None:raise RuntimeError('ViZDoom returned no state at episode start')
        return self._obs(state)

    def _obs(self,state):
        return image_tensor(state.screen_buffer,int(self.cfg['model']['height']),int(self.cfg['model']['width']))

    def label_state(self):
        if not self.labels: raise RuntimeError('Teacher-only labels were not enabled')
        return self.game.get_state()

    def step(self,action):
        reward=float(self.game.make_action(button_vector(action),int(self.cfg['action_repeat'])))
        done=bool(self.game.is_episode_finished())
        state=None if done else self.game.get_state()
        obs=None if state is None else self._obs(state)
        return obs,reward,done,{'episode_time':int(self.game.get_episode_time()),
                                'total_reward':float(self.game.get_total_reward()),
                                'truncated':bool(done and self.game.get_episode_time()>=int(self.cfg.get('episode_timeout',300)))}
