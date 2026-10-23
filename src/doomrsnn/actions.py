from enum import IntEnum
import numpy as np

class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    SHOOT = 3
    LEFT_SHOOT = 4
    RIGHT_SHOOT = 5

BUTTON_NAMES = ('MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK')
BUTTON_VALUES = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1]], dtype=np.int32)


def button_vector(action: int):
    if not 0 <= int(action) < len(Action):
        raise ValueError(f'Invalid action {action}')
    return BUTTON_VALUES[int(action)].tolist()
