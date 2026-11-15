"""Deep learning reduced order model (DL-ROM) of the CFF on the fault.

An autoencoder compresses the CFF field into a few latent variables, and the reduced
network maps the parameters and the time to the same latent variables. The CFF is
predicted as decoder(reduced_network(parameters, time)).
"""

import torch
from torch import nn

N_INPUTS = 10
N_CELLS = 416
LATENT_SIZE = 10


def fully_connected(sizes):
    """Fully connected layers of the given sizes, each followed by a PReLU."""
    layers = []
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(size_in, size_out))
        layers.append(nn.PReLU())
    network = nn.Sequential(*layers)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    return network


class DLROM(nn.Module):
    def __init__(self, training_inputs, training_cff):
        """The training data give the ranges used to scale inputs and CFF to [0, 1]."""
        super().__init__()
        self.encoder = fully_connected([N_CELLS, 18, LATENT_SIZE])
        self.decoder = fully_connected([LATENT_SIZE, 18, 18, 18, N_CELLS])
        self.reduced_network = fully_connected([N_INPUTS, 40, 40, 40, 40, LATENT_SIZE])

        # buffers are saved together with the weights, but they are not trained
        self.register_buffer("inputs_min", training_inputs.min(dim=0).values)
        self.register_buffer("inputs_max", training_inputs.max(dim=0).values)
        self.register_buffer("cff_min", training_cff.min(dim=0).values)
        self.register_buffer("cff_max", training_cff.max(dim=0).values)

    def scale_inputs(self, inputs):
        return (inputs - self.inputs_min) / (self.inputs_max - self.inputs_min)

    def scale_cff(self, cff):
        return (cff - self.cff_min) / (self.cff_max - self.cff_min)

    def unscale_cff(self, cff_scaled):
        return cff_scaled * (self.cff_max - self.cff_min) + self.cff_min

    def forward(self, inputs):
        """Predict the CFF in MPa from the 9 parameters and the time."""
        latent = self.reduced_network(self.scale_inputs(inputs))
        return self.unscale_cff(self.decoder(latent))
