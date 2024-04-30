from typing import Callable, Dict, Optional, Tuple
from gluonts.torch.distributions.distribution_output import DistributionOutput

class DiffusionOutput(DistributionOutput):
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
