"""Einheitentests des Kantersieg-Stresstests."""

import numpy as np
import pytest

import mcl_margins as MG
import turnier_generator as T


def _wins(n=6, k=5, seed=3):
    return T.build_turnier(n, 0.3, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", seed).wins


def test_without_blowouts_every_win_is_one_point():
    W = _wins()
    D = MG.point_differential(W, 0.0, 10, T.SplitMix64(1))
    assert np.array_equal(D, W - W.T)


def test_with_all_blowouts_every_win_has_height_h():
    W = _wins()
    D = MG.point_differential(W, 1.0, 7, T.SplitMix64(1))
    assert np.array_equal(D, 7 * (W - W.T))


def test_differential_is_antisymmetric_and_deterministic():
    W = _wins()
    D = MG.point_differential(W, 0.3, 10, T.SplitMix64(5))
    assert np.array_equal(D, -D.T) and (np.diag(D) == 0).all()
    assert np.array_equal(D, MG.point_differential(W, 0.3, 10, T.SplitMix64(5)))
    assert not np.array_equal(D, MG.point_differential(W, 0.3, 10, T.SplitMix64(6)))


def test_blowout_share_matches_the_expected_mean_of_a_single_pair():
    """1000 Siege von 0 über 1, Anteil q = 0,2, Höhe 10: Erwartung 1000 · (1 + q·(h - 1)) = 2800; Vier-Sigma-Band um den Erwartungswert."""
    W = np.zeros((2, 2), dtype=int)
    W[0, 1] = 1000
    means = [MG.point_differential(W, 0.2, 10, T.SplitMix64(s))[0, 1] for s in range(20)]
    sigma = 9 * np.sqrt(1000 * 0.2 * 0.8)
    assert np.mean(means) == pytest.approx(2800, abs=4 * sigma / np.sqrt(20))
    assert all(abs(m - 2800) < 4 * sigma for m in means)


def test_unplayed_pairs_have_no_points():
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 2
    D = MG.point_differential(W, 0.5, 10, T.SplitMix64(1))
    assert D[0, 2] == 0 and D[1, 2] == 0 and D[2, 0] == 0
