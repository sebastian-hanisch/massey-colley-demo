"""Einheitentests der linearen Systeme gegen die von Hand gerechneten Mini-Instanzen (conftest.py)."""

import math

import numpy as np
import pytest

import mcl_linear as L
import turnier_generator as T


def test_laplacian_of_the_mini_instance(mini3):
    _, G, _ = mini3
    assert L.laplacian(G).tolist() == [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]


def test_massey_hand_values(mini3):
    _, G, D = mini3
    M, p = L.massey_system(G, D)
    assert p.tolist() == [2.0, 0.0, -2.0]
    r = L.massey_rating(G, D)
    assert r == pytest.approx([2 / 3, 0.0, -2 / 3])
    assert r.sum() == pytest.approx(0.0)


def test_colley_hand_values(mini3):
    W, G, _ = mini3
    C, b = L.colley_system(W, G)
    assert C.tolist() == [[4, -1, -1], [-1, 4, -1], [-1, -1, 4]]
    assert b.tolist() == [2.0, 1.0, 0.0]
    r = L.colley_rating(W, G)
    assert r == pytest.approx([0.7, 0.5, 0.3])
    assert r.sum() == pytest.approx(1.5)


def test_row_replaced_matches_the_pseudoinverse_on_a_connected_plan(mini3):
    _, G, D = mini3
    r, cond = L.massey_row_replaced(G, D)
    assert r == pytest.approx(L.massey_rating(G, D))
    assert 1.0 <= cond < 100


def test_row_replaced_fails_on_a_disconnected_plan_but_the_pseudoinverse_does_not(islands):
    W, G, D = islands
    r, cond = L.massey_row_replaced(G, D)
    assert r is None and math.isinf(cond)
    pinv = L.massey_rating(G, D)
    assert np.isfinite(pinv).all()
    assert pinv[0] > pinv[1] and pinv[2] > pinv[3]                 # innerhalb jedes Teils stimmt die Ordnung
    assert pinv[0] + pinv[1] == pytest.approx(0.0) and pinv[2] + pinv[3] == pytest.approx(0.0)    # je Teil Summe 0


def test_matrix_rank_deficit_counts_the_components(mini3, islands):
    assert L.matrix_rank_deficit(L.laplacian(mini3[1])) == 1       # ein Teil: M ist singulär in genau einer Richtung
    assert L.matrix_rank_deficit(L.laplacian(islands[1])) == 2     # zwei Teile
    assert T.components(islands[1]) == 2 and T.components(mini3[1]) == 1


def test_win_share_without_games_is_zero_not_nan():
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 1
    G = W + W.T
    q = L.win_share(W, G)
    assert q.tolist() == [1.0, 0.0, 0.0] and not np.isnan(q).any()


def test_rank_with_tolerance_ties_noise_but_not_real_differences():
    for s in range(30):
        noisy = L.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))
        assert noisy[2] == 2                                    # der Dritte ist klar hinten
    firsts = {L.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))[0] for s in range(40)}
    assert firsts == {0, 1}                                     # das Rauschen von 1e-13 entscheidet nicht; das Los
    real = {L.rank_with_tolerance([1.0, 1.0 + 1e-6, 0.5], T.SplitMix64(s))[1] for s in range(40)}
    assert real == {0}                                          # ein echter Unterschied schon


def test_kendall_tau_and_crossing_wrong_hand_instances():
    assert L.kendall_tau((0, 2, 1), (3, 1, 2)) == 1.0
    assert L.kendall_tau((0, 1, 2), (1, 2, 3)) == -1.0
    assert math.isnan(L.kendall_tau((0, 1), (5.0, 5.0)))
    assert L.crossing_wrong((0, 3, 1, 2), 2) == pytest.approx(0.5)
    assert L.crossing_wrong((3, 2, 1, 0), 2) == 0.0
    assert math.isnan(L.crossing_wrong((0, 1, 2), 0)) and math.isnan(L.crossing_wrong((0, 1, 2), 3))
    assert L.crossing_wrong((2, 1, 0), 2) == 0.0                # zwei Schwache, ein Starker vorn: kein nan


def test_crossing_wrong_with_an_empty_league_is_nan_without_a_numpy_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # der Mittelwert über ein leeres Feld würde eine RuntimeWarning auslösen
        assert math.isnan(L.crossing_wrong((0, 1, 2), 0))
        assert math.isnan(L.crossing_wrong((0, 1, 2), 3))


def test_crossing_wrong_counts_only_strictly_better_places():
    # absichtlich gleiche Plätze: weak = (1, 1), strong = (1, 2); nur 1 < 2 zählt (je Schwacher einmal), 1 < 1 nicht: 2 von 4
    assert L.crossing_wrong((1, 1, 1, 2), 2) == pytest.approx(0.5)


def test_residual_share_is_nan_when_every_pair_difference_is_zero():
    G = np.ones((3, 3)) - np.eye(3)
    assert math.isnan(L.residual_share(G, np.zeros((3, 3)), np.zeros(3)))
    assert math.isnan(L.residual_share(G, np.zeros((3, 3)), np.array([1.0, 0.0, -1.0])))     # Nenner 0: kein Anteil definiert


def test_residual_is_zero_for_exactly_transitive_differences():
    s = np.array([3.0, 1.0, 0.0, -1.0])
    G = np.ones((4, 4)) - np.eye(4)
    D = s[:, None] - s[None, :]
    r = L.massey_rating(G, D)
    assert L.residual_share(G, D, r) == pytest.approx(0.0, abs=1e-12)
    # bei einem Zyklus bleibt etwas übrig: A schlägt B, B schlägt C, C schlägt A (je 1)
    G3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    D3 = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
    assert L.residual_share(G3, D3, L.massey_rating(G3, D3)) == pytest.approx(1.0)       # r = 0: nichts wird erklärt
    assert math.isnan(L.residual_share(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3)))
