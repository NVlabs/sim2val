# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.txt

# ruff: noqa

"""The following provides a sketch of the code used for using control variates
to compute metrics from the nuPlan dataset. This is *not* runnable code.
"""

from sim2val.control_variates import control_variates_estimator

# Datasets are created that form the training, validation, and test sets.
# A MLP is trained using the training/val datasets, with the best model
# being saved for later use.
dataset_paired = ...  # Dataset with paired open and closed metrics. Note: any data that was used
# for training/validation of the model is excluded from this dataset.
dataset_unpaired = ...  # Dataset with only open metrics
model = ...  # Trained MLP model for control variates

paired_loader = DataLoader(dataset_paired, batch_size=len(dataset_paired), shuffle=False)
unpaired_loader = DataLoader(dataset_unpaired, batch_size=len(dataset_unpaired), shuffle=False)
paired_embeddings, paired_open_metrics, paired_closed_metrics = next(iter(paired_loader))
paired_closed_metrics = paired_closed_metrics.cpu().numpy().squeeze()
unpaired_embeddings, unpaired_open_metrics, _ = next(iter(unpaired_loader))
with torch.no_grad():
    paired_closed_predictions = (
        model(paired_embeddings, paired_open_metrics).cpu().numpy().squeeze()
    )
    unpaired_closed_predictions = (
        model(unpaired_embeddings, unpaired_open_metrics).cpu().numpy().squeeze()
    )

cv_result = control_variates_estimator(
    paired_closed_metrics,
    paired_closed_predictions,
    unpaired_closed_predictions,
)
