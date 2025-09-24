# Sim2Val

This project provides `sim2val`--a Python package with utilities for validation using simulated data. For
more information, please see the [paper](https://arxiv.org/abs/2506.20553).

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
