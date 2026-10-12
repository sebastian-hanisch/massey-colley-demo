"""Punktedifferenz aus den Ausgängen des Generators (Version 1 liefert nur Sieg oder Niederlage). Lokaler Stresstest, keine Aussage über reale Spiele:
Jeder Sieg zählt 1 Punkt Vorsprung; mit Wahrscheinlichkeit q ist er ein Kantersieg der Höhe h, unabhängig von der Stärke."""

import numpy as np

import turnier_generator as T


def point_differential(W, q: float, h: int, rng: T.SplitMix64) -> np.ndarray:
    """D[i, j] = Punkte von i gegen j minus Punkte von j gegen i über alle Partien des Paares. Deterministisch in (W, q, h, rng)."""
    W = np.asarray(W, dtype=int)
    n = len(W)
    points = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins = int(W[i, j])
            blow = int((rng.uniform(wins) < q).sum()) if (q > 0 and wins > 0) else 0
            points[i, j] = (wins - blow) + blow * h
    return points - points.T
