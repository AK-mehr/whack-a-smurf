import numpy as np

from strategic_detector.detectors import roc_curve_from_score


def test_roc_curve_endpoints_and_monotonicity():
    rng = np.random.default_rng(0)
    n = 1000
    is_laundering = rng.random(n) < 0.05
    score = np.where(is_laundering, rng.normal(3, 1, n), rng.normal(0, 1, n))

    _, alpha, beta = roc_curve_from_score(score, is_laundering)

    assert alpha[-1] == 1.0
    assert beta[-1] == 1.0
    assert np.all(np.diff(alpha) >= 0)
    assert np.all(np.diff(beta) >= -1e-9)


def test_roc_curve_separates_when_scores_disjoint():
    is_laundering = np.array([True, True, False, False])
    score = np.array([10.0, 9.0, 1.0, 0.0])

    _, alpha, beta = roc_curve_from_score(score, is_laundering)

    idx = np.argmax(beta == 1.0)
    assert alpha[idx] == 0.0


def test_roc_curve_requires_both_classes():
    import pytest

    with pytest.raises(ValueError):
        roc_curve_from_score(np.array([1.0, 2.0]), np.array([True, True]))
