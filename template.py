from torch import nn
from .mws import MWS

# Prior
class Pc(nn.Module):
    def forward(self, i, c=None):
        """
        i: batch of task indices
        c: batch of latent variables

        Returns a pair (c, log_prob)
        If c is not already given, sample from prior
        """
        return c, log_prob

# Recognition model (encoder)
class Rc(nn.Module):
    def forward(self, i, x, c=None):
        """
        i: batch of task indices
        x: batch of observations
        c: batch of latent variables

        Returns a pair (c, log_prob)
        If c is not already given, sample from prior
        """
        return c, log_prob

# Decoder
class Px(nn.Module):
    def forward(self, i, c, x=None):
        """
        i: batch of task indices
        c: batch of latent variables
        x: batch of observations

        Returns a pair (x, log_prob)
        If x is not already given, sample from decoder
        """
        return x, log_prob

prior = Pc()
decoder = Px()
encoder = Rc()

mws = MWS(encoder, decoder, prior, nTasks=...)
