import math

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    """
    Returns the cosine-annealed learning rate for the requested step.
    """
    # Write code here
    lr=min_lr
    t1=(base_lr-min_lr)
    t1=t1/2
    angle=math.pi*current_step
    angle=angle/total_steps
    t1=t1*(1+math.cos(angle))
    lr=lr+t1
    return lr