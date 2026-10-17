"""Eigenschaftstests massey-colley-demo (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für Massey und Colley gelten müssen, ohne dass das richtige
Ergebnis bekannt ist. Je Eigenschaft 30 Zufallsinstanzen mit festem Seed (SplitMix64), plus Randfälle (kleine n, zerfallender Plan, Teilnehmer ohne
Partie, ungeschlagene Mannschaft, nur Remis). Alle Aussagen folgen aus den Definitionen M r = p (Laplace-Matrix, p_i = Summe der Punktedifferenzen) und
C r = b (C = 2I + M, b_i = 1 + (Siege - Niederlagen)/2); nichts hier beschreibt nur das heutige Verhalten des Codes.

1. Umnummerieren der Teilnehmer permutiert beide Wertungen.
2. Spiegeln aller Ergebnisse (Sieger und Verlierer vertauschen, D -> -D): Massey r -> -r, Colley r -> 1 - r (C · 1 = 2 · 1 und b' = 2 - b).
3. Summen: Massey summiert sich je Teil des Spielplans zu 0 (Minimum-Norm-Lösung, README), Colley mittelt je Teil auf 1/2 (C ist blockdiagonal, jede Zeile von
   C summiert sich zu 2).
4. Verdoppeln aller Partien (G, D -> 2G, 2D) ändert die Massey-Wertung nicht (M und p verdoppeln sich); die Skalierung der Punktedifferenz mit c > 0 skaliert Massey.
5. Ergebnis einer Partie umdrehen (j schlug i, jetzt schlägt i j): Colley hebt i und senkt j (C ist eine streng diagonaldominante M-Matrix, die Spalte der
   Inversen hat ihr striktes Maximum auf der Diagonalen); bei Massey wächst der Abstand r_i - r_j strikt (Änderung = 2 * (e_i - e_j)' M^+ (e_i - e_j) > 0, wenn i und j im selben Teil liegen).
6. Colley-Wertungen liegen strikt in (0, 1); ein Teilnehmer ohne Partie hat Colley 1/2 und Massey 0, ohne dass er die anderen verändert.
7. Rangordnung (`rank_with_tolerance`) ohne Gleichstände = absteigende Reihenfolge der Wertung und kommt beim Umnummerieren mit; Wertungen nahe 1e-12 sind gleich.
"""

import itertools

import numpy as np
import pytest

import mcl_linear as L
import turnier_generator as T

N_INSTANCES = 30


def instance(seed, n=None, p=0.6, kmax=3, draws=False):
    """Zufälliger Spielplan mit Ausgängen und Punktedifferenzen (SplitMix64, deterministisch in seed). Rückgabe G, W, D (float).
    Mit draws=True gibt es Remis (Differenz 0, kein Sieg): nur für Massey, Colley kennt sie nicht."""
    rng = T.SplitMix64(seed)
    if n is None:
        n = 2 + int(rng.uniform(1)[0] * 7)             # 2 bis 8
    G = np.zeros((n, n))
    W = np.zeros((n, n))
    D = np.zeros((n, n))
    pairs = list(itertools.combinations(range(n), 2))
    play = rng.uniform(len(pairs)) < p
    for (i, j), on in zip(pairs, play):
        if not on:
            continue
        k = 1 + int(rng.uniform(1)[0] * kmax)
        for u1, u2 in zip(rng.uniform(k), rng.uniform(k)):
            margin = 1 + int(u2 * 6)
            if draws and u2 < 0.15:
                margin = 0
            G[i, j] += 1
            G[j, i] += 1
            if margin == 0:
                continue
            w, l = (i, j) if u1 < 0.5 else (j, i)
            W[w, l] += 1
            D[w, l] += margin
            D[l, w] -= margin
    return G, W, D


def instances(draws=False):
    return [instance(1000 + s, draws=draws) for s in range(N_INSTANCES)]


def component_labels(G):
    n = len(G)
    lab = [-1] * n
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        while stack:
            u = stack.pop()
            if lab[u] >= 0:
                continue
            lab[u] = c
            stack.extend(v for v in range(n) if G[u][v] > 0 and lab[v] < 0)
        c += 1
    return np.array(lab)


def permute(perm, *mats):
    """Teilnehmer neu nummerieren: neuer Index a ist der alte Index perm[a]."""
    return [m[np.ix_(perm, perm)] for m in mats]


# --- 1. Umnummerieren -------------------------------------------------------------------------------------------------------------------

def test_relabelling_the_participants_permutes_both_ratings():
    rng = T.SplitMix64(11)
    for G, W, D in instances(draws=True) + [instance(5, n=2), instance(6, n=3, p=1.0)]:
        perm = rng.permutation(len(G))
        G2, W2, D2 = permute(perm, G, W, D)
        assert np.allclose(L.massey_rating(G2, D2), L.massey_rating(G, D)[perm], atol=1e-9)
    for G, W, D in instances() + [instance(5, n=2), instance(6, n=3, p=1.0)]:
        perm = rng.permutation(len(G))
        G2, W2, D2 = permute(perm, G, W, D)
        assert np.allclose(L.colley_rating(W2, G2), L.colley_rating(W, G)[perm], atol=1e-9)


# --- 2. Spiegeln ------------------------------------------------------------------------------------------------------------------------

def test_mirroring_all_results_negates_massey_and_maps_colley_to_one_minus_r():
    for G, W, D in instances(draws=True):
        assert np.allclose(L.massey_rating(G, -D), -L.massey_rating(G, D), atol=1e-9)
    for G, W, D in instances():
        assert np.allclose(L.colley_rating(W.T, G), 1.0 - L.colley_rating(W, G), atol=1e-9)
        assert np.allclose(L.massey_rating(G, -D), -L.massey_rating(G, D), atol=1e-9)


def test_mirroring_reverses_the_ranking_when_there_are_no_ties():
    rng = T.SplitMix64(3)
    checked = 0
    for G, W, D in instances():
        for rate, mirrored in ((L.massey_rating(G, D), L.massey_rating(G, -D)), (L.colley_rating(W, G), L.colley_rating(W.T, G))):
            if np.min(np.abs(np.diff(np.sort(rate)))) < 1e-6:        # Gleichstand: Umkehr ist dann nur bis aufs Los belegt
                continue
            a = L.rank_with_tolerance(rate, rng)
            b = L.rank_with_tolerance(mirrored, rng)
            assert [len(G) - 1 - x for x in a] == list(b)
            checked += 1
    assert checked >= 20


# --- 3. Summen und Mittel ----------------------------------------------------------------------------------------------------------------

def test_massey_sums_to_zero_and_colley_averages_one_half_in_every_component():
    for G, W, D in instances(draws=True):
        lab = component_labels(G)
        massey = L.massey_rating(G, D)
        for c in set(lab):
            assert massey[lab == c].sum() == pytest.approx(0.0, abs=1e-9)
        assert massey.sum() == pytest.approx(0.0, abs=1e-9)
    for G, W, D in instances():
        lab = component_labels(G)
        colley = L.colley_rating(W, G)
        for c in set(lab):
            assert colley[lab == c].mean() == pytest.approx(0.5, abs=1e-9)
        assert colley.mean() == pytest.approx(0.5, abs=1e-9)


def test_with_only_draws_massey_is_zero_and_a_colley_chain_is_strictly_ordered():
    G = np.ones((4, 4)) - np.eye(4)
    assert np.allclose(L.massey_rating(G, np.zeros((4, 4))), 0.0, atol=1e-12)
    W = np.triu(np.ones((4, 4)), 1)                      # 0 schlägt alle, 1 schlägt 2 und 3, 2 schlägt 3
    r = L.colley_rating(W, G)
    assert r[0] > r[1] > r[2] > r[3] and r.mean() == pytest.approx(0.5)


# --- 4. Verdoppeln und Skalieren -----------------------------------------------------------------------------------------------------------

def test_doubling_all_games_keeps_massey_and_scaling_the_margins_scales_it():
    for s, (G, W, D) in enumerate(instances(draws=True)):
        base = L.massey_rating(G, D)
        assert np.allclose(L.massey_rating(2 * G, 2 * D), base, atol=1e-9)
        c = 0.5 + s * 0.37
        assert np.allclose(L.massey_rating(G, c * D), c * base, atol=1e-8)


def test_a_positive_scale_keeps_the_massey_order_and_a_negative_one_reverses_it():
    for G, W, D in instances():
        base = L.massey_rating(G, D)
        diff = base[:, None] - base[None, :]
        clear = np.abs(diff) > 1e-6                      # nur eindeutige Paare: Gleichstände entscheidet sonst das Rauschen
        scaled = L.massey_rating(G, 7.5 * D)
        assert (np.sign(scaled[:, None] - scaled[None, :])[clear] == np.sign(diff)[clear]).all()
        neg = L.massey_rating(G, -3.0 * D)
        assert (np.sign(neg[:, None] - neg[None, :])[clear] == -np.sign(diff)[clear]).all()


# --- 5. Eine Partie umdrehen ---------------------------------------------------------------------------------------------------------------

def test_turning_a_loss_into_a_win_lifts_the_winner_and_lowers_the_loser():
    flips = 0
    for G, W, D in instances():
        n = len(G)
        pairs = [(i, j) for i in range(n) for j in range(n) if W[j, i] > 0]        # j hat i mindestens einmal geschlagen
        if not pairs:
            continue
        i, j = pairs[(flips * 7) % len(pairs)]
        W2 = W.copy()
        W2[j, i] -= 1
        W2[i, j] += 1
        r, r2 = L.colley_rating(W, G), L.colley_rating(W2, G)
        assert r2[i] > r[i] and r2[j] < r[j]
        # Massey: gleiche Umkehr, die Punktedifferenz dieser Partie wechselt das Vorzeichen (Betrag m -> -m, also Änderung 2m)
        D2 = D.copy()
        m = 3.0
        D2[i, j] += 2 * m
        D2[j, i] -= 2 * m
        if component_labels(G)[i] == component_labels(G)[j]:
            m1, m2 = L.massey_rating(G, D), L.massey_rating(G, D2)
            assert (m2[i] - m2[j]) > (m1[i] - m1[j]) + 1e-9
        flips += 1
    assert flips >= 20


# --- 6. Wertebereich und unbeteiligte Teilnehmer -------------------------------------------------------------------------------------------

def test_colley_stays_strictly_between_zero_and_one_even_for_an_unbeaten_team():
    for G, W, D in instances():
        r = L.colley_rating(W, G)
        assert (r > 0).all() and (r < 1).all()
    # ungeschlagen über viele Partien, schlechtester Gegner: kommt nahe an 1, erreicht sie nie
    G = np.ones((3, 3)) * 20 - np.eye(3) * 20
    W = np.zeros((3, 3))
    W[0, 1] = W[0, 2] = W[1, 2] = 20
    r = L.colley_rating(W, G)
    assert 0.5 < r[0] < 1.0 and 0.0 < r[2] < 0.5 and r[0] > r[1] > r[2]


def test_a_participant_without_games_is_neutral_and_leaves_the_others_unchanged():
    for G, W, D in instances()[:20]:
        n = len(G)
        pad = lambda m: np.pad(m, ((0, 1), (0, 1)))
        Gp, Wp, Dp = pad(G), pad(W), pad(D)
        assert L.massey_rating(Gp, Dp)[-1] == pytest.approx(0.0, abs=1e-9)
        assert L.colley_rating(Wp, Gp)[-1] == pytest.approx(0.5, abs=1e-9)
        assert np.allclose(L.massey_rating(Gp, Dp)[:n], L.massey_rating(G, D), atol=1e-9)
        assert np.allclose(L.colley_rating(Wp, Gp)[:n], L.colley_rating(W, G), atol=1e-9)


# --- 7. Rangordnung -----------------------------------------------------------------------------------------------------------------------

def test_ranking_without_ties_is_the_descending_order_and_follows_a_relabelling():
    rng = T.SplitMix64(21)
    for k in range(N_INSTANCES):
        n = 2 + k % 9
        score = rng.uniform(n) * 10 - 5
        rank = L.rank_with_tolerance(score, rng)
        assert sorted(rank) == list(range(n))
        assert [int(x) for x in np.argsort(-score)] == [i for _, i in sorted((p, i) for i, p in enumerate(rank))]
        perm = rng.permutation(n)
        rank2 = L.rank_with_tolerance(score[perm], rng)
        assert list(rank2) == [rank[x] for x in perm]
