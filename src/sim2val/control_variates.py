# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.txt

"""This file contains utilities to aid in the estimation of metrics using multiple platforms.

For more detailed information about the methods, see https://arxiv.org/abs/2506.20553
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class ControlVariatesResult:
    """The result of the control variates computation."""

    mu_hat_beta: float  # The estimated mean
    var_mu_hat_beta: float  # The estimated variance mu_hat_beta
    beta: float  # The estimated optimal control variate coefficient


def _covariance_matrix(a: np.ndarray, b: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Compute the covariance matrix of two arrays.

    This method assumes a is of the shape (n, m) where n is the number of samples
    and m is the number of features in a. Similarly, b is of the shape (n, p) where
    n is the number of samples and p is the number of features in b.
    """
    assert a.shape[0] == b.shape[0], "Arrays must have the same number of rows"
    n = a.shape[0]
    assert n > 1, "Arrays must have more than one row to compute covariance"
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)

    return 1 / (n - ddof) * (a - a_mean).T @ (b - b_mean)


def control_variates_estimator(
    f: np.ndarray, g: np.ndarray, g_unpaired: np.ndarray
) -> ControlVariatesResult:
    """Compute the control variates result using the given metrics.

    f: metric of interest, dimension: n
    g: paired metric of f, should be the same length as f, dimension: n or n x m
    g_unpaired: "large" set of g used to estimate population stats of g, dimension: k x m
    """
    n = len(f)
    k = g_unpaired.shape[0]

    if n <= 0:
        raise ValueError("f must have at least one element")
    elif k <= 0:
        raise ValueError("g_unpaired must have at least one element")
    elif len(g) != n:
        raise ValueError("g must have the same length as f")

    var_f = np.var(f, ddof=1)
    if len(g.shape) == 1:  # scalar version
        theta_hat = np.mean(g_unpaired)
        cov_g_f = np.cov(g, f, ddof=1)[0, 1]
        var_g = np.var(g, ddof=1)
        var_g_unpaired = np.var(g_unpaired, ddof=1)
        if np.isclose(var_g, 0):
            raise ValueError("Variance of g is zero, cannot compute control variates")

        beta_hat = k / (k + n) * cov_g_f / var_g
        mu_hat = 1 / n * np.sum(f - beta_hat * g) + beta_hat * theta_hat
        var_mu_hat = (1 / n) * (var_f + beta_hat**2 * var_g - 2 * beta_hat * cov_g_f) + (
            1 / k
        ) * beta_hat**2 * var_g_unpaired

    else:  # matrix version
        var_g = _covariance_matrix(g, g, ddof=1)
        cov_g_f = _covariance_matrix(g, f, ddof=1)
        var_g_unpaired = _covariance_matrix(g_unpaired, g_unpaired, ddof=1)

        beta_hat = k / (n + k) * np.linalg.inv(var_g) @ cov_g_f
        mu_hat = np.mean(f - beta_hat.T @ g.T) + beta_hat.T @ np.mean(g_unpaired, axis=0).T
        var_mu_hat = (1 / n) * (
            var_f + beta_hat.T @ var_g @ beta_hat - 2 * beta_hat.T @ cov_g_f
        ) + (1 / k) * beta_hat.T @ var_g_unpaired @ beta_hat

    return ControlVariatesResult(
        mu_hat_beta=mu_hat,
        var_mu_hat_beta=var_mu_hat,
        beta=beta_hat,
    )


def chebyshev_confidence_interval(
    mean: float, variance: float, confidence_level: float
) -> tuple[float, float]:
    """Returns the Chebyshev confidence interval around the mean.

    Args:
        mean (float): The mean of the distribution.
        variance (float): The variance of the distribution.
        confidence_level (float): Desired confidence level (e.g., 0.95).

    Returns:
        (float, float): Lower and upper bounds of the interval.
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")
    elif variance <= 0:
        raise ValueError("Variance must be positive.")

    k = (1 / (1 - confidence_level)) ** 0.5
    radius = k * variance**0.5
    return mean - radius, mean + radius


def normal_confidence_interval(
    mean: float, variance: float, confidence_level: float
) -> tuple[float, float]:
    """Returns the confidence interval around the mean assuming a normal distribution.

    Args:
        mean (float): The mean of the distribution.
        variance (float): The variance of the distribution.
        confidence_level (float): Desired confidence level (e.g., 0.95).

    Returns:
        (float, float): Lower and upper bounds of the interval.
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")
    elif variance <= 0:
        raise ValueError("Variance must be positive.")

    alpha = 1 - confidence_level
    radius = norm.ppf(1 - alpha / 2.0) * variance**0.5
    return mean - radius, mean + radius
