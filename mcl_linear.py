"""Lineare Bewertungssysteme auf dem Spielplan (numpy/Standardbibliothek): Massey (Kleinste Quadrate auf der Punktedifferenz) und Colley (geglättete
Siegquote mit Gegnerstärke). Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); Zufall nur über SplitMix64.

Notation: G[i, j] = Zahl der Partien von i gegen j (symmetrisch), W[i, j] = Siege von i gegen j, D[i, j] = Punkte von i gegen j minus Punkte von j
gegen i (antisymmetrisch)."""

import numpy as np

import turnier_generator as T


def laplacian(G) -> np.ndarray:
    """Laplace-Matrix des Spielgraphen: Diagonale = Zahl der Partien, außerhalb = minus Zahl der Partien zwischen den beiden."""
    A = np.asarray(G, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def massey_system(G, D) -> tuple:
    """Massey: M r = p mit M = Laplace-Matrix und p_i = Summe der Punktedifferenzen von i über alle Gegner."""
    return laplacian(G), np.asarray(D, dtype=float).sum(axis=1)


def massey_rating(G, D) -> np.ndarray:
    """Massey-Wertung per Minimum-Norm-Lösung (Pseudoinverse): auch bei zerfallendem Plan definiert, je Teil mit Summe 0."""
    M, p = massey_system(G, D)
    r, *_ = np.linalg.lstsq(M, p, rcond=None)
    return r


def massey_row_replaced(G, D):
    """Originalweg: die letzte Zeile durch lauter Einsen ersetzen (Summe der Wertungen 0). Rückgabe (Wertung oder None, Kondition der Matrix).
    Bei zerfallendem Plan ist die Matrix singulär: Wertung None, Kondition unendlich."""
    M, p = massey_system(G, D)
    M = M.copy()
    p = p.copy()
    M[-1, :] = 1.0
    p[-1] = 0.0
    cond = float(np.linalg.cond(M))
    if not np.isfinite(cond) or cond > 1e12:
        return None, float("inf")
    return np.linalg.solve(M, p), cond


def colley_system(W, G) -> tuple:
    """Colley: C r = b mit C = 2I + Laplace-Matrix und b_i = 1 + (Siege_i - Niederlagen_i) / 2."""
    W = np.asarray(W, dtype=float)
    G = np.asarray(G, dtype=float)
    wins = W.sum(axis=1)
    losses = G.sum(axis=1) - wins
    return 2.0 * np.eye(len(G)) + laplacian(G), 1.0 + (wins - losses) / 2.0


def colley_rating(W, G) -> np.ndarray:
    """Colley-Wertung: die Lösung des Systems; Mittel aller Wertungen ist 1/2."""
    C, b = colley_system(W, G)
    return np.linalg.solve(C, b)


def win_share(W, G) -> np.ndarray:
    """Siegquote je Teilnehmer (Siege je Partie; ohne Partien 0)."""
    wins = np.asarray(W, dtype=float).sum(axis=1)
    games = np.asarray(G, dtype=float).sum(axis=1)
    return np.divide(wins, games, out=np.zeros_like(wins), where=games > 0)


def rank_with_tolerance(score, rng: T.SplitMix64, tol: float = 1e-9) -> tuple:
    """Platz je Teilnehmer (0 = bester) nach absteigender Wertung. Wertungen, die sich um weniger als `tol` unterscheiden, sind gleich und werden
    durch das Los getrennt: Sonst entschiede Gleitkomma-Rauschen (1e-16) in vollständigen Plänen, wo Massey exakte Gleichstände hat."""
    score = np.asarray(score, dtype=float)
    key = np.round(score / tol).astype(np.int64)
    lots = rng.uniform(len(score))
    order = sorted(range(len(score)), key=lambda i: (-key[i], lots[i]))
    pos = [0] * len(score)
    for place, i in enumerate(order):
        pos[i] = place
    return tuple(pos)


def kendall_tau(rank, values) -> float:
    """τ zwischen den Plätzen (kleiner = besser) und einer Größe `values` (größer = besser); Paare mit gleichem Wert zählen nicht."""
    r = np.asarray(rank, dtype=float)
    v = np.asarray(values, dtype=float)
    iu = np.triu_indices(len(r), 1)
    sv = np.sign(v[iu[0]] - v[iu[1]])
    sr = np.sign(r[iu[1]] - r[iu[0]])
    counted = sv != 0
    if not counted.any():
        return float("nan")
    return float((sv[counted] * sr[counted]).mean())


def crossing_wrong(rank, n_weak: int) -> float:
    """Anteil der Paare (stark, schwach) aus zwei Staffeln, in denen der Schwache in der Tabelle vor dem Starken steht (Indexhälften)."""
    r = np.asarray(rank)
    weak, strong = r[:n_weak], r[n_weak:]
    if len(weak) == 0 or len(strong) == 0:
        return float("nan")
    return float((weak[:, None] < strong[None, :]).mean())


def residual_share(G, D, r) -> float:
    """Anteil der Paar-Differenzen (Punktedifferenz je Partie), den r_i - r_j nicht erklärt: Σ(y - fit)² / Σ y² über alle gespielten Paare.
    Ohne gespielte Paare oder bei y = 0 überall: nan."""
    G = np.asarray(G, dtype=float)
    D = np.asarray(D, dtype=float)
    r = np.asarray(r, dtype=float)
    iu = np.triu_indices(len(G), 1)
    played = G[iu] > 0
    if not played.any():
        return float("nan")
    y = D[iu][played] / G[iu][played]
    fit = (r[:, None] - r[None, :])[iu][played]
    denom = float((y ** 2).sum())
    return float(((y - fit) ** 2).sum() / denom) if denom > 0 else float("nan")


def matrix_rank_deficit(M) -> int:
    """n minus Rang: bei einer Laplace-Matrix die Zahl der Teile des Spielplans."""
    return len(M) - int(np.linalg.matrix_rank(M))
