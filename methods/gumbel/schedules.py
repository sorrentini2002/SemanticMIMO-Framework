
import math

def compute_tau(step, tau_max=1.0, tau_min=0.3, num_steps=10000, anneal_mode="linear"):
    """
    Compute Gumbel-Softmax temperature tau based on annealing schedule.
    
    Args:
        step: int, current global step
        tau_max: float, initial temperature
        tau_min: float, minimum temperature
        num_steps: int, duration of annealing (in steps)
        anneal_mode: str, "linear", "cosine", or "exp"
    
    Returns:
        float: current tau
    """
    if step >= num_steps:
        return tau_min
        
    t = step / float(num_steps) # 0 to 1
    
    if anneal_mode == 'linear':
        # Linear decay from tau_max to tau_min
        return tau_max - t * (tau_max - tau_min)
        
    elif anneal_mode == 'cosine':
        # Cosine decay
        # Starts at tau_max, ends at tau_min
        cosine_decay = 0.5 * (1 + math.cos(math.pi * t))
        return tau_min + (tau_max - tau_min) * cosine_decay
        
    elif anneal_mode == 'exp':
        # Exponential decay
        # tau = tau_max * (tau_min / tau_max)^t
        return tau_max * (tau_min / tau_max)**t
        
    else:
        # constant (or default fallback)
        return tau_max
