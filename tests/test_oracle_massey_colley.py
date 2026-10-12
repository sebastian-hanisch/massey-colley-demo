"""Orakel-Tests massey-colley-demo: lineare Systeme und Kennzahlen gegen einen ANDEREN Rechenweg (DEMO-PLAYBOOK §12).

1. Massey gegen die Normalgleichungen der Kleinsten Quadrate aus der Partienliste und gegen numpy.linalg.pinv.
2. Colley gegen die geschlossene Formel r_i = (1 + w_i + k/2)/(2 + k n) im vollständigen, ausgeglichenen Plan; Summe n/2; positiv definit.
3. Rang der Laplace-Matrix gegen die Zahl der Komponenten (networkx).
4. Vollständig und ausgeglichen: Massey = Colley = Siegtabelle (mit und ohne Gleichstände).
5. Residuum 0 bei exakt transitiven Daten; Massey linear in der Skala der Punktedifferenz.
"""

import itertools
import random

import numpy as np
import pytest

import mcl_linear as L
import mcl_margins as MG
import turnier_generator as T


def random_plan(n, rnd, p=0.6, k_max=3):
    """Zufälliger Spielplan mit Ausgängen: je Paar mit Wahrscheinlichkeit p bis zu k_max Partien."""
    G = np.zeros((n, n), dtype=int)
    W = np.zeros((n, n), dtype=int)
    games = []
    for i, j in itertools.combinations(range(n), 2):
        if rnd.random() < p:
            g = rnd.randint(1, k_max)
            w = sum(rnd.random() < 0.5 for _ in range(g))
            G[i, j] = G[j, i] = g
            W[i, j], W[j, i] = w, g - w
            games += [(i, j, 1)] * w + [(j, i, 1)] * (g - w)
    return W, G, games


# --- 1. Massey -----------------------------------------------------------------------------------------------------------------------

def test_massey_solves_the_least_squares_normal_equations_from_the_game_list():
    rnd = random.Random(2026)
    for _ in range(60):
        n = rnd.randint(3, 8)
        W, G, games = random_plan(n, rnd)
        if not games:
            continue
        D = W - W.T
        # Kleinste Quadrate über die Partienliste: min Σ (r_i - r_j - 1)² für jede Partie "i schlägt j"
        X = np.zeros((len(games), n))
        y = np.zeros(len(games))
        for row, (a, b, margin) in enumerate(games):
            X[row, a], X[row, b], y[row] = 1.0, -1.0, margin
        ref, *_ = np.linalg.lstsq(X, y, rcond=None)           # Minimum-Norm
        ours = L.massey_rating(G, D)
        assert np.allclose(ours, ref, atol=1e-8)
        assert np.allclose(ours, np.linalg.pinv(L.laplacian(G)) @ D.sum(axis=1), atol=1e-8)


def test_massey_scales_linearly_with_the_point_differential():
    rnd = random.Random(7)
    for _ in range(30):
        W, G, _ = random_plan(6, rnd, p=0.8)
        D = W - W.T
        assert np.allclose(L.massey_rating(G, 7 * D), 7 * L.massey_rating(G, D))


# --- 2. Colley -----------------------------------------------------------------------------------------------------------------------

def test_colley_closed_form_in_a_complete_balanced_plan_and_its_properties():
    for s in range(30):
        t = T.build_turnier(8, 0.3, 3, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 700 + s)
        r = L.colley_rating(t.wins, t.games)
        wins = t.wins.sum(axis=1)
        games = t.games.sum(axis=1)
        k = 3                                                          # Partien je Paar in diesem Turnier
        # Herleitung: (2 + g) r_i - k (n/2 - r_i) = 1 + (w_i - l_i)/2 mit g = k (n-1) und Summe der Wertungen n/2
        assert np.allclose(r, (1 + wins + k / 2) / (2 + k * 8))
        assert np.allclose(games, k * 7)
        assert r.sum() == pytest.approx(8 / 2)


def test_colley_matrix_is_positive_definite_with_smallest_eigenvalue_two():
    rnd = random.Random(3)
    for _ in range(40):
        W, G, _ = random_plan(rnd.randint(3, 9), rnd)
        C, _ = L.colley_system(W, G)
        assert np.linalg.eigvalsh(C).min() >= 2.0 - 1e-9


# --- 3. Rang und Komponenten ----------------------------------------------------------------------------------------------------------

def test_rank_of_the_laplacian_is_n_minus_the_number_of_components():
    nx = pytest.importorskip("networkx")
    rnd = random.Random(11)
    for _ in range(80):
        n = rnd.randint(2, 10)
        _, G, _ = random_plan(n, rnd, p=0.25)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if G[i, j] > 0)
        comps = nx.number_connected_components(g)
        assert L.matrix_rank_deficit(L.laplacian(G)) == comps == T.components(G)


# --- 4. Vollständig und ausgeglichen ---------------------------------------------------------------------------------------------------

def test_in_a_complete_balanced_plan_massey_colley_and_the_table_give_the_same_scores_up_to_order():
    for s in range(60):
        t = T.build_turnier(8, 0.3, 2, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 900 + s)
        wins = t.wins.sum(axis=1).astype(float)
        D = t.wins - t.wins.T
        massey = L.massey_rating(t.games, D)
        colley = L.colley_rating(t.wins, t.games)
        # Massey r_i = p_i / n; Colley r_i = (1 + w_i)/(2 + g_i): beide streng monoton in den Siegen (auch bei Gleichständen: gleiche Wertung)
        def sign(x):
            return 0.0 if abs(x) < 1e-9 else float(np.sign(x))      # Gleitkomma-Rauschen von 1e-16 ist kein Unterschied

        for i, j in itertools.combinations(range(8), 2):
            assert sign(massey[i] - massey[j]) == sign(wins[i] - wins[j])
            assert sign(colley[i] - colley[j]) == sign(wins[i] - wins[j])


# --- 5. Residuum ------------------------------------------------------------------------------------------------------------------------

def test_residual_is_zero_for_transitive_differences_on_random_plans():
    rnd = random.Random(19)
    for _ in range(40):
        n = rnd.randint(4, 8)
        s = np.array([rnd.uniform(-3, 3) for _ in range(n)])
        _, G, _ = random_plan(n, rnd, p=0.9)
        if T.components(G) > 1:
            continue
        D = (s[:, None] - s[None, :]) * (G > 0)
        r = L.massey_rating(G, D * G)       # Differenzen je Partie mal Zahl der Partien: Summe der Margins je Paar
        assert L.residual_share(G, D * G, r) == pytest.approx(0.0, abs=1e-9)


def test_blowout_height_scales_the_massey_ratings_through_the_point_differential():
    W = T.build_turnier(6, 0.3, 4, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 5).wins
    G = W + W.T
    D1 = MG.point_differential(W, 1.0, 1, T.SplitMix64(1))
    D5 = MG.point_differential(W, 1.0, 5, T.SplitMix64(1))
    assert np.allclose(L.massey_rating(G, D5), 5 * L.massey_rating(G, D1))
