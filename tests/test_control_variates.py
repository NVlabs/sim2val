# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.txt

"""Unit tests for control_variates module."""

import numpy as np
import pytest
from sim2val.control_variates import (
    chebyshev_confidence_interval,
    control_variates_estimator,
    normal_confidence_interval,
)


def test_control_variates_estimator_scalar():
    """Nominal test for scalar control variates estimator."""
    np.random.seed(0)
    MU = 8.3
    VAR_F = 1.0
    VAR_H = 1.0
    RHO = 0.95
    N = 100
    K = 1000
    cov_f_h = RHO * np.sqrt(VAR_F * VAR_H)
    covariance = np.array([[1.0, cov_f_h], [cov_f_h, VAR_H]])

    fh = np.random.multivariate_normal(np.array([MU, 0.0]), covariance, size=N + K)
    f = fh[:N, 0]
    h = fh[:N, 1]
    hp = fh[N:, 1]

    mu_mc = np.mean(f)
    var_mc = np.var(f) / N
    result = control_variates_estimator(f, h, hp)

    # sanity check that the cv est is better than the mc est
    EXPECTED_BETA_OPT = K / (K + N) * cov_f_h / VAR_H
    assert result.var_mu_hat_beta < var_mc
    assert abs(result.mu_hat_beta - MU) < abs(mu_mc - MU)
    assert np.isclose(result.beta, EXPECTED_BETA_OPT, atol=1e-2)


def test_control_variates_estimator_fail_invalid_input():
    """Test that control variates estimator raises ValueError for invalid inputs."""
    # zero length n
    with pytest.raises(ValueError):
        control_variates_estimator(np.array([]), np.array([1, 2]), np.array([1, 2, 3]))
    # zero length k
    with pytest.raises(ValueError):
        control_variates_estimator(np.array([1, 2]), np.array([]), np.array([1, 2, 3]))
    # h not same length as f
    with pytest.raises(ValueError):
        control_variates_estimator(np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3]))
    # h has variance of zero
    with pytest.raises(ValueError):
        control_variates_estimator(np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([1, 2, 3]))


def test_control_variates_estimator_vector():
    """Nominal test for vector control variates estimator."""
    np.random.seed(0)

    MU = 8.3
    VAR_F = 1.0
    VAR_H1 = 1.0
    VAR_H2 = 2.0
    RHO_F_H1 = 0.95
    RHO_F_H2 = 0.3
    RHO_H1_H2 = 0.4
    N = 100
    K = 1000
    cov_f_h1 = RHO_F_H1 * np.sqrt(VAR_F * VAR_H1)
    cov_f_h2 = RHO_F_H2 * np.sqrt(VAR_F * VAR_H2)
    cov_h1_h2 = RHO_H1_H2 * np.sqrt(VAR_H1 * VAR_H2)
    covariance = np.array(
        [[VAR_F, cov_f_h1, cov_f_h2], [cov_f_h1, VAR_H1, cov_h1_h2], [cov_f_h2, cov_h1_h2, VAR_H2]]
    )

    fh = np.random.multivariate_normal(np.array([MU, 0.0, 0.0]), covariance, size=N + K)
    f = fh[:N, 0]
    h = fh[:N, 1:]
    hp = fh[N:, 1:]

    mu_mc = np.mean(f, axis=0)
    var_mc = np.var(f, axis=0) / N
    result = control_variates_estimator(f, h, hp)

    # sanity check that the cv est is better than the mc est
    EXPECTED_BETA_OPT = K / (K + N) * np.linalg.inv(covariance[1:, 1:]) @ covariance[1:, 0]
    assert result.var_mu_hat_beta < var_mc
    assert np.abs(result.mu_hat_beta - MU) < np.abs(mu_mc - MU)
    assert np.allclose(result.beta, EXPECTED_BETA_OPT, atol=5e-2)


def test_chebyshev_confidence_interval():
    """Test Chebyshev confidence interval with nominal values."""
    MEAN = 0.5
    VAR = 1.0
    CONFIDENCE_LEVEL = 0.95
    EXPECTED_RADIUS = 4.4721

    ci = chebyshev_confidence_interval(MEAN, VAR, CONFIDENCE_LEVEL)
    assert len(ci) == 2  # noqa: PLR2004
    assert np.isclose(ci[0], MEAN - EXPECTED_RADIUS, atol=1e-4)
    assert np.isclose(ci[1], MEAN + EXPECTED_RADIUS, atol=1e-4)


def test_chebyshev_confidence_interval_invalid():
    """Test invalid inputs for Chebyshev confidence interval."""
    # Invalid confidence level
    with pytest.raises(ValueError):
        chebyshev_confidence_interval(0, 1, 1.5)
    with pytest.raises(ValueError):
        chebyshev_confidence_interval(0, 1, -0.1)

    # Zero variance
    with pytest.raises(ValueError):
        chebyshev_confidence_interval(0, 0, 0.95)


def test_normal_confidence_interval():
    """Test normal confidence interval with nominal values."""
    MEAN = 0.5
    VAR = 1.0
    CONFIDENCE_LEVEL = 0.95
    EXPECTED_RADIUS = 1.96

    ci = normal_confidence_interval(MEAN, VAR, CONFIDENCE_LEVEL)
    assert len(ci) == 2  # noqa: PLR2004
    assert np.isclose(ci[0], MEAN - EXPECTED_RADIUS, atol=1e-4)
    assert np.isclose(ci[1], MEAN + EXPECTED_RADIUS, atol=1e-4)


def test_normal_confidence_interval_invalid():
    """Test invalid inputs for normal confidence interval."""
    # Invalid confidence level
    with pytest.raises(ValueError):
        normal_confidence_interval(0, 1, 1.5)
    with pytest.raises(ValueError):
        normal_confidence_interval(0, 1, -0.1)

    # Zero variance
    with pytest.raises(ValueError):
        normal_confidence_interval(0, 0, 0.95)
