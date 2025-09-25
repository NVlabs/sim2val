# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.txt

# ruff: noqa

"""
This script can be run with:
    uv run --extra=nb notebooks/quadruped.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import torch
from sim2val.control_variates import control_variates_estimator
from torch import nn, optim

# Set seed for reproducibility
torch.manual_seed(2)
torch.cuda.manual_seed(2)

# ----- Step 1: Load and split data -----
real = pd.read_parquet("testdata/quadruped/real.parquet")
sim = pd.read_parquet("testdata/quadruped/paired_sim.parquet")
only_sim = pd.read_parquet("testdata/quadruped/only_sim.parquet")

# Construct X: [command_velocity (3,) + sim["avg_track_error"] (1,)]
cmd_vel = np.array([np.array(x).flatten() for x in real["command_velocity"]])  # shape (N, 3)
sim_error = sim["avg_track_error"].values.reshape(-1, 1)  # shape (N, 1)
X = np.hstack((cmd_vel, sim_error))  # shape (N, 4)

cmd_vel_only_sim = np.array(
    [np.array(x).flatten() for x in only_sim["command_velocity"]]
)  # shape (N, 3)
sim_error_only_sim = only_sim["avg_track_error"].values.reshape(-1, 1)  # shape (N, 1)
X_only_sim = np.hstack((cmd_vel_only_sim, sim_error_only_sim))  # shape (N, 4)

# Construct y: just real["avg_track_error"]
y = real["avg_track_error"].values.reshape(-1, 1)  # shape (N, 1)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
X_only_sim_tensor = torch.tensor(X_only_sim, dtype=torch.float32)

# Total number of paired samples
n = X_tensor.shape[0]
perm = torch.randperm(n)

# Split indices
n_train = 40
idx_1 = perm[:n_train]
idx_2 = perm[n_train:]

# Random splits
X_1, X_2 = X_tensor[idx_1], X_tensor[idx_2]
y_1, y_2 = y_tensor[idx_1], y_tensor[idx_2]


# ----- Step 2: Define and train Metric Correlator Function -----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4),  # input size changed from 3 to 4
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),  # output constrained to (0, 1)
        )

    def forward(self, x):
        return self.net(x)


def fit_model(X_1, y_1):
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # ----- Training loop -----
    num_epochs = 1000
    for epoch in range(num_epochs):
        # ---- Training step ----
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_1)
        loss_train = criterion(pred_train, y_1)
        loss_train.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss_train.item():.6f}")
    return model


model = fit_model(X_1, y_1)

# ----- Step 3: Control Variates -----
y_real = y_2  # real metric (not used for MCF training)
y_paired_sim = model(X_2)  # paired_sim metric (not used for MCF training)
y_only_sim = model(X_only_sim_tensor)  # sim_only metric

y_real = np.squeeze(y_real.detach().cpu().numpy())
y_paired_sim = np.squeeze(y_paired_sim.detach().cpu().numpy())
y_only_sim = np.squeeze(y_only_sim.detach().cpu().numpy())

result = control_variates_estimator(y_real, y_paired_sim, y_only_sim)
print(f"result: {result}")
