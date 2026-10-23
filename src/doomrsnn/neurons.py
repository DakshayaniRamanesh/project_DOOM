import torch
from torch.autograd import Function

class SurrogateSpike(Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        derivative = 1.0 / (1.0 + slope * x.abs()).square()
        return grad_output * derivative, None


def lif_step(current, membrane, previous_spike, beta=0.5, threshold=0.4, slope=5.0):
    # Subtractive reset on the NEXT timestep, consistent for train and inference.
    next_membrane = beta * membrane + current - threshold * previous_spike
    spike = SurrogateSpike.apply(next_membrane - threshold, slope)
    return spike, next_membrane
