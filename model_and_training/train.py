"""Train the DL-ROM on the training set and save it in the results folder.

Three terms are minimized together:
reconstruction: the autoencoder must reproduce the CFF field it receives
latent:         the reduced network must give the same latent variables as the encoder
regularization: mean absolute value of the weights
"""

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dlrom import DLROM
from src.data.torch_dataset import FaultStabilityDataset

SEED = 42
N_EPOCHS = 701
BATCH_SIZE = 3
LEARNING_RATE = 0.5e-3
EPOCH_OF_HALVED_LEARNING_RATE = 500
REGULARIZATION_WEIGHT = 1e-4


def losses(model, inputs, cff):
    cff_scaled = model.scale_cff(cff)
    latent_encoder = model.encoder(cff_scaled)
    latent_reduced = model.reduced_network(model.scale_inputs(inputs))
    mse = torch.nn.functional.mse_loss
    reconstruction = mse(model.decoder(latent_encoder), cff_scaled)
    latent = mse(latent_reduced, latent_encoder.detach())
    return reconstruction, latent


def regularization(model):
    mean_absolute_weights = [weights.abs().mean() for weights in model.parameters()]
    return REGULARIZATION_WEIGHT * sum(mean_absolute_weights)


torch.manual_seed(SEED)
torch.set_num_threads(1)  # the networks are small: one thread is faster than many

training_data = FaultStabilityDataset("train")
validation_data = FaultStabilityDataset("validation")
training_batches = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

model = DLROM(training_data.inputs, training_data.cff)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[EPOCH_OF_HALVED_LEARNING_RATE], gamma=0.5
)

history = []
start = time.time()

for epoch in range(1, N_EPOCHS + 1):
    for inputs, cff in training_batches:
        reconstruction, latent = losses(model, inputs, cff)
        loss = reconstruction + latent + regularization(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    with torch.no_grad():
        training_losses = losses(model, training_data.inputs, training_data.cff)
        validation_losses = losses(model, validation_data.inputs, validation_data.cff)
    training_losses = torch.stack(training_losses).tolist()
    validation_losses = torch.stack(validation_losses).tolist()
    history.append([epoch, *training_losses, *validation_losses])

    if epoch % 50 == 1:
        print(
            f"epoch {epoch}:"
            f" training loss {sum(training_losses):.2e},"
            f" validation loss {sum(validation_losses):.2e}"
        )

print(f"training time: {time.time() - start:.0f} s")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/dlrom.pt")
np.savetxt(
    "results/dlrom_losses.txt",
    np.array(history),
    header="epoch training_reconstruction training_latent"
    " validation_reconstruction validation_latent",
)
