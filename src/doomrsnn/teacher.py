"""Privileged BASIC teacher; never pass label objects into a student's forward method."""
from dataclasses import dataclass
from .actions import Action

@dataclass
class LabelDecision:
    action: int
    target_x: float | None
    width: float | None
    reason: str


class BasicTeacher:
    def __init__(self,alignment_fraction=0.055,target_names=('cacodemon','doomimp','imp')):
        self.alignment_fraction=float(alignment_fraction)
        self.target_names=tuple(x.lower() for x in target_names)

    def predict(self,labels,screen_width):
        candidates=[]
        for label in labels or []:
            name=str(getattr(label,'object_name','')).lower()
            if not any(token in name for token in self.target_names):
                continue
            x=float(getattr(label,'x',0))
            w=float(getattr(label,'width',0))
            if w <= 0:continue
            candidates.append((w,x+w/2,name))
        if not candidates:
            return LabelDecision(int(Action.NOOP),None,None,'target_missing')
        w,x,_=max(candidates)
        normalized=(x-screen_width/2)/max(1.0,screen_width)
        if normalized < -self.alignment_fraction:
            return LabelDecision(int(Action.LEFT),x,w,'align_left')
        if normalized > self.alignment_fraction:
            return LabelDecision(int(Action.RIGHT),x,w,'align_right')
        return LabelDecision(int(Action.SHOOT),x,w,'aligned_shoot')

    def act(self,game):
        state=game.label_state()
        if state is None: raise RuntimeError('Teacher queried at a finished episode')
        frame=state.screen_buffer
        width=frame.shape[-1] if frame.shape[0] in (1,3,4) else frame.shape[1]
        return self.predict(getattr(state,'labels',()),width)
