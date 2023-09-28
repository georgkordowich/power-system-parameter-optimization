import torch
import numpy as np


# MomentumOptimizer
class CustomOptimizer(torch.optim.Optimizer):
    """
    Optimizer that reduces the learning rate each step, automatically choses a reasonable learning rate and enforces
    a maximum relative step size for the parameter optimization.
    """
    # Init Method:
    def __init__(self, params, max_step=0.1, decay=0.95):
        self.decay = decay
        # Max step can be a list of values (one for each parameter), or a single value.
        try:
            max_steps = [max_step[i] * abs(p.data.real) for i, p in enumerate(params)]
        except TypeError:
            max_steps = [max_step * abs(p.data.real) for p in params]

        # lrs will be initialized later
        lrs = [None for _ in params]

        defaults = dict(lr=lrs, max_steps=max_steps)
        super(CustomOptimizer, self).__init__(params, defaults=defaults)

        # store the last parameters in case the simulation becomes unstable
        self.last_params = [p.data.clone() for p in params]

    def step(self):
        # first, check if any gradient is NaN. This can happen in cases of unstable simulations.
        if np.any([[torch.isnan(p.grad.real) for p in group['params']] for group in self.param_groups]):
            print('NaN gradient detected. Using last parameters and reducing learning rate.')
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    p.data = self.last_params[p_idx]
                    group['lr'][p_idx] *= self.decay
            return
        else:
            # update last parameters
            self.last_params = [p.data.clone() for p in self.param_groups[0]['params']]

        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                if group['lr'][p_idx] is None:
                    # Determine a reasonable learning rate for the first step.
                    group['lr'][p_idx] = group['max_steps'][p_idx] / abs(p.grad.real.data)
                else:
                    # reduce learning rate in each step, so that the relative step size is always smaller than max_step.
                    group['max_steps'][p_idx] *= self.decay
                    group['lr'][p_idx] = min(group['lr'][p_idx]*self.decay, group['max_steps'][p_idx] / abs(p.grad.real.data))

                # Update the parameters so that the relative step size is smaller than max_step.
                p.data -= torch.clamp(group['lr'][p_idx] * p.grad.real.data, min=-group['max_steps'][p_idx], max=group['max_steps'][p_idx])
