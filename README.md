# Sim2Val

This project provides `sim2val`--a Python package with utilities for validation using simulated data. For
more information, please see the [paper](https://arxiv.org/abs/2506.20553).

## Abstract
Learning-based robotic systems demand rigorous validation to assure reliable performance, but extensive real-world testing is often prohibitively expensive, and if conducted may still yield insufficient data for high-confidence guarantees. In this work we introduce Sim2Val, a general estimation framework that leverages paired data across test platforms, e.g., paired simulation and real-world observations, to achieve better estimates of real-world metrics via the method of control variates. By incorporating cheap and abundant auxiliary measurements (for example, simulator outputs) as control variates for costly real-world samples, our method provably reduces the variance of Monte Carlo estimates and thus requires significantly fewer real-world samples to attain a specified confidence bound on the mean performance. We provide theoretical analysis characterizing the variance and sample-efficiency improvement, and demonstrate empirically in autonomous driving and quadruped robotics settings that our approach achieves high-probability bounds with markedly improved sample efficiency. Our technique can lower the real-world testing burden for validating the performance of the stack, thereby enabling more efficient and cost-effective experimental evaluation of robotic systems.


## Motivation
* Traditional validation requires extensive real-world testing to achieve the confidence levels needed for safety assurances and certification
* Simulation-only validation would be much cheaper, but simulators are not yet accurate enough for standalone validation, and would shift the problem to that of validating a simulator
* Goal: Combine real-world and simulation testing to reduce real-world data requirements for validation

 <img width="128" height="84" alt="platforms" src="https://github.com/user-attachments/assets/abdae8a4-dc65-4b09-b2e5-4c50597b41e3" />

## Method
**Idea:** Use simulation as a control variate!

* With a control variate – a correlated signal whose expectation is known – we can reduce the variance of our estimator
* Simulation measurements are correlated with real-world measurements
* Because the true simulation mean is unknown, we estimate it from the sim-only data
<img width="276" height="266" alt="paired_thumbnail" src="https://github.com/user-attachments/assets/3fc8b2f0-9142-4719-8d36-67be16f60d18" />

<img width="842" height="264" alt="real_to_sim_mapping" src="https://github.com/user-attachments/assets/8cc2c4d3-31fe-430c-aaf0-0b3d6ef64217" />



### Examples of Paired Scenarios
![verf_real_sim_highway](https://github.com/user-attachments/assets/a9fde77f-1e37-499b-b86a-011601d9ab4b)
<img src="https://github.com/user-attachments/assets/a9fde77f-1e37-499b-b86a-011601d9ab4b" width="300">


## Getting started

The python package can be built with either:

```bash
# Using uv
uv run python -m build

# Using pip
python -m build .
```

And if using `pip`, installed with:

```bash
pip install .
```

### Example Notebook

To get started with `sim2val`, you can run the example notebooks--for example:

```bash
jupyter notebook notebooks/simple_run.ipynb

# or

uv run python -m jupyter notebook notebooks/simple_run.ipynb
```

## Testing

Unit tests are found in the `tests` directory and can be run with:

```bash
# Using uv
uv run --all-extras pytest

# Using pip
pytest
```

## Citation

If you use this code, please cite the following paper:

```
@inproceedings{luo2025_sim2val,
title = {Sim2Val: Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation},
author = {Rachel Luo and Heng Yang and Michael Watson and Apoorva Sharma and Sushant Veer and Edward Schmerling and Marco Pavone},
booktitle = {Proceedings of the Conference on Robot Learning (CoRL)},
year = {2025},
}
```
