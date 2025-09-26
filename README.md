# Sim2Val

This project provides `sim2val`--a Python package with utilities for validation using simulated data. For
more information, please see the [paper](https://www.arxiv.org/pdf/2506.20553) or the [poster](corl_2025_poster.pdf).

## Abstract
Learning-based robotic systems demand rigorous validation to assure reliable performance, but extensive real-world testing is often prohibitively expensive, and if conducted may still yield insufficient data for high-confidence guarantees. In this work we introduce Sim2Val, a general estimation framework that leverages paired data across test platforms, e.g., paired simulation and real-world observations, to achieve better estimates of real-world metrics via the method of control variates. By incorporating cheap and abundant auxiliary measurements (for example, simulator outputs) as control variates for costly real-world samples, our method provably reduces the variance of Monte Carlo estimates and thus requires significantly fewer real-world samples to attain a specified confidence bound on the mean performance. We provide theoretical analysis characterizing the variance and sample-efficiency improvement, and demonstrate empirically in autonomous driving and quadruped robotics settings that our approach achieves high-probability bounds with markedly improved sample efficiency. Our technique can lower the real-world testing burden for validating the performance of the stack, thereby enabling more efficient and cost-effective experimental evaluation of robotic systems.


## Motivation
<table>
  <tr>
    <td>
      <img width="384" height="252" alt="platforms" src="https://github.com/user-attachments/assets/abdae8a4-dc65-4b09-b2e5-4c50597b41e3" />
    </td>
    <td>
      Sim2Val estimator of mean:<br>
      <ul>
        <li>Note that variance reduction is a function of the scale of sim-only data (k) and the correlation (ρ²)</li>
      </ul>
      If the original paired observations have low correlation:<br>
      <ul>
        <li>Traditional validation requires extensive real-world testing to achieve the confidence levels needed for safety assurances and certification</li>
        <li>Simulation-only validation would be much cheaper, but simulators are not yet accurate enough for standalone validation, and would shift the problem to that of validating a simulator</li>
        <li>Goal: Combine real-world and simulation testing to reduce real-world data requirements for validation</li>
      </ul>
    </td>

  </tr>
</table>


## Method

<table>
  <tr>
    <td>
      **Idea:** Use simulation as a control variate!<br>
      <ul>
        <li> With a control variate – a correlated signal whose expectation is known – we can reduce the variance of our estimator </li>
        <li> Simulation measurements are correlated with real-world measurements </li>
        <li> Because the true simulation mean is unknown, we estimate it from the sim-only data </li>
      </ul>
    </td>
    <td>
        <img width="276" height="266" alt="paired_thumbnail" src="https://github.com/user-attachments/assets/3fc8b2f0-9142-4719-8d36-67be16f60d18" />
    </td>
  </tr>
</table>


<img width="842" height="264" alt="real_to_sim_mapping" src="https://github.com/user-attachments/assets/8cc2c4d3-31fe-430c-aaf0-0b3d6ef64217" />

<table>
  <tr>
    <td>
      Sim2Val estimator of mean:<br>
      <ul>
        <li>Note that variance reduction is a function of the scale of sim-only data (k) and the correlation (ρ²)</li>
      </ul>
      If the original paired observations have low correlation:<br>
      <ul>
        <li>We can learn a nonlinear metric correlator function (MCF) mapping scenario features + sim measurements → a refined surrogate metric</li>
        <li>We can then use the new surrogate as a control variate</li>
      </ul>
    </td>
    <td>
      <img width="457" height="426" alt="equations_and_corr" src="https://github.com/user-attachments/assets/a40f2b7c-5d1b-4a8f-999c-ae5fda1a0243" />
    </td>
  </tr>
</table>



### Examples of Paired Scenarios
<img src="https://github.com/user-attachments/assets/a9fde77f-1e37-499b-b86a-011601d9ab4b" width="400">
<img src="https://github.com/user-attachments/assets/20ee58be-7026-42d3-a38f-18a88849d017" width="400">

## Results

### Sim2Val Autonomous Driving Performance
<img width="1280" height="220" alt="image" src="https://github.com/user-attachments/assets/0d23df4a-b69f-4a2f-9bc7-6c87865f723f" />

By leveraging inexpensive sim samples, we can achieve equivalent confidence level as using ~6x the number of expensive real-world samples! (Variance reduction of 82.9%)

### Sim2Val Quadruped Velocity Tracking

<img width="1280" height="163" alt="image" src="https://github.com/user-attachments/assets/12ef0ac4-3456-4560-a503-ce5e73ef11e4" />

Note that even the relatively modest reduction in variance from 2.048E−5 to 1.926E−5 would have required 38% more real-world tests without sim! 

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
