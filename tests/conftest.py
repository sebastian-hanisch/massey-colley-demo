"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini3():
    """3 Teilnehmer A=0, B=1, C=2, je eine Partie gegen jeden: A schlägt B, B schlägt C, A schlägt C (Punktedifferenz je 1).
    Handrechnung: M = 3I - J (Diagonale 2, außerhalb -1), p = (2, 0, -2), Massey r = p / 3 = (2/3, 0, -2/3).
    Colley: C = 5I - J (Diagonale 4, außerhalb -1), b = (1+1, 1, 1-1) = (2, 1, 0); Summe der Wertungen = n/2 = 1,5, r = (b + 1,5)/5 = (0,7; 0,5; 0,3)."""
    G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    W = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    return W, G, W - W.T


@pytest.fixture
def islands():
    """4 Teilnehmer, zwei getrennte Paare: 0 schlägt 1 (zweimal), 2 schlägt 3 (einmal). Der Plan hat 2 Teile; die Laplace-Matrix hat Rang 4 - 2 = 2."""
    G = np.zeros((4, 4), dtype=int)
    G[0, 1] = G[1, 0] = 2
    G[2, 3] = G[3, 2] = 1
    W = np.zeros((4, 4), dtype=int)
    W[0, 1] = 2
    W[2, 3] = 1
    return W, G, W - W.T
