"""Literatur-Referenzwerte massey-colley-demo (ORACLE-PLAYBOOK §5.1): Massey und Colley gegen Zahlen, die in eingesehenen Quellen stehen.

Alle Quellen am 2026-10-17 eingesehen (WebFetch; die beiden PDFs lokal mit pdftotext gelesen). Die Werte stammen aus den Quellen, nicht aus dem
Code der Demo.

1. Colley, W. N. (2002): Colley's Bias Free College Football Ranking Method: The Colley Matrix Explained. https://www.colleyrankings.com/matrate.pdf,
   Abschnitt 6.3 (Gl. 21-25, S. 13-14 der PDF-Fassung).
   a) Zwei Mannschaften, eine Partie (1-0 und 0-1): 3 r_W - r_L = 3/2, -r_W + 3 r_L = 1/2, Lösung r_W = 5/8, r_L = 3/8 (Gl. 21-23).
   b) Fünf Mannschaften a-e: gedruckte Matrix C (Gl. 24), rechte Seite (1/2, 1, 1, 1, 3/2), Lösung r = {19, 24, 23, 22, 27}/46 (Gl. 25), Mittel 1/2,
      Mannschaft c genau 1/2. Die Paarungen sind aus der gedruckten Ergebnistabelle (Zeilen a-e) gelesen; sie stimmen mit der gedruckten Matrix
      und den gedruckten Bilanzen überein (a 1-2, b 1-1, c 2-2, d 1-1, e 2-1). Colley hängt nur an Bilanz und Zahl der Partien.
2. Bozzo, E., Franceschet, M. (2017): The Massey's method for sport rating: a network science perspective. arXiv:1701.03363v1,
   Abschnitt 2.1 (S. 5-6): vier fiktive Mannschaften A-D, vier Partien (A-C 2:0, A-D 3:0, B-C 1:1, B-D 2:1), gedruckte Matrix M, Torvektor
   p = (5, 1, -2, -4) und Massey-Wertung r = (1,75; -0,25; -0,25; -1,25). Die Partie B-C ist ein Remis (Differenz 0): trägt Massey mit seiner
   Normierung "Summe 0" und zählt als Partie in M.
3. Dokumentation der Referenzsoftware comperank (R-Paket), Vignette "Methods Overview",
   https://archive.linux.duke.edu/cran/web/packages/comperank/vignettes/methods-overview.html: Datensatz ncaa2005 (die zehn Partien von Duke,
   Miami, UNC, UVA, VT, nach dem Beispiel in Langville/Meyer, "Who's #1?"; die Vignette nennt keine Seitenzahl) mit den gedruckten Ergebnissen
   von rate_massey (Duke -24,8; Miami 18,2; UNC -8; UVA -3,40; VT 18) und rank_colley(keep_rating = TRUE) (Duke 0,214; Miami 0,786; UNC 0,5;
   UVA 0,357; VT 0,643). Paketversion in der Vignette nicht genannt. Die Torvektor-Summen je Mannschaft (-124, 91, -40, -17, 90) rechnet der Test
   aus den Partien nach.

Nicht eingesehen (kein Test, siehe Bericht "offen"): Massey (1997) Abschlussarbeit; Langville/Meyer (2012) selbst (nur über 3. gestützt).
"""

import numpy as np
import pytest

import mcl_linear as L


def _tables(n, games):
    """G, D, W aus (i, j, Punkte_i, Punkte_j)-Listen; Remis zählt als Partie ohne Sieg."""
    G = np.zeros((n, n))
    D = np.zeros((n, n))
    W = np.zeros((n, n))
    for a, b, s1, s2 in games:
        G[a, b] += 1
        G[b, a] += 1
        D[a, b] += s1 - s2
        D[b, a] += s2 - s1
        if s1 > s2:
            W[a, b] += 1
        elif s2 > s1:
            W[b, a] += 1
    return G, D, W


# --- Colley 2002 ---------------------------------------------------------------------------------------------------------------------

def test_colley_two_team_example_gives_five_eighths_and_three_eighths():
    W = np.array([[0, 1], [0, 0]])
    G = W + W.T
    C, b = L.colley_system(W, G)
    assert C.tolist() == [[3, -1], [-1, 3]]
    assert b.tolist() == [1.5, 0.5]
    assert L.colley_rating(W, G) == pytest.approx([5 / 8, 3 / 8], abs=1e-12)


def test_colley_five_team_example_matches_the_printed_matrix_and_ratings():
    # Gewinner, Verlierer (a=0 ... e=4), gelesen aus der gedruckten Tabelle: a schlägt c; d und e schlagen a; c schlägt b und d; b schlägt e; e schlägt c
    results = [(0, 2), (3, 0), (4, 0), (2, 1), (1, 4), (2, 3), (4, 2)]
    W = np.zeros((5, 5))
    G = np.zeros((5, 5))
    for win, lose in results:
        W[win, lose] += 1
        G[win, lose] += 1
        G[lose, win] += 1
    assert W.sum(axis=1).tolist() == [1, 1, 2, 1, 2] and (G.sum(axis=1) - W.sum(axis=1)).tolist() == [2, 1, 2, 1, 1]   # gedruckte Bilanzen
    C, b = L.colley_system(W, G)
    printed_C = [[5, 0, -1, -1, -1], [0, 4, -1, 0, -1], [-1, -1, 6, -1, -1], [-1, 0, -1, 4, 0], [-1, -1, -1, 0, 5]]    # Gl. 24
    assert C.tolist() == printed_C
    assert b.tolist() == [0.5, 1.0, 1.0, 1.0, 1.5]
    r = L.colley_rating(W, G)
    assert r == pytest.approx(np.array([19, 24, 23, 22, 27]) / 46, abs=1e-12)    # Gl. 25
    assert r.mean() == pytest.approx(0.5, abs=1e-12)
    assert r[2] == pytest.approx(0.5, abs=1e-12)                                  # c ist 2-2 gegen Gegner mit Mittel 1/2
    assert r[1] > r[3]                                                            # b vor d bei gleicher Bilanz: Gegnerstärke zählt


# --- Bozzo/Franceschet 2017 ----------------------------------------------------------------------------------------------------------

def test_massey_four_team_example_with_a_draw_matches_the_paper():
    # A=0, B=1, C=2, D=3: A-C 2:0, A-D 3:0, B-C 1:1, B-D 2:1
    G, D, _ = _tables(4, [(0, 2, 2, 0), (0, 3, 3, 0), (1, 2, 1, 1), (1, 3, 2, 1)])
    M, p = L.massey_system(G, D)
    assert M.tolist() == [[2, 0, -1, -1], [0, 2, -1, -1], [-1, -1, 2, 0], [-1, -1, 0, 2]]      # gedruckte Matrix M
    assert p.tolist() == [5, 1, -2, -4]                                                       # gedruckter Torvektor
    r = L.massey_rating(G, D)
    assert r == pytest.approx([1.75, -0.25, -0.25, -1.25], abs=1e-12)
    assert r.sum() == pytest.approx(0.0, abs=1e-12)
    # Originalweg (letzte Zeile durch Einsen) gibt dasselbe, wie im Papier
    r2, _ = L.massey_row_replaced(G, D)
    assert r2 == pytest.approx(r, abs=1e-10)


# --- comperank (Referenzsoftware-Dokumentation), Datensatz ncaa2005 ---------------------------------------------------------------------

NCAA2005 = [(0, 1, 7, 52), (0, 2, 21, 24), (0, 3, 7, 38), (0, 4, 0, 45), (1, 2, 34, 16), (1, 3, 25, 17), (1, 4, 27, 7), (2, 3, 7, 5),
            (2, 4, 3, 30), (3, 4, 14, 52)]            # Duke, Miami, UNC, UVA, VT


def test_massey_and_colley_on_ncaa2005_match_the_comperank_documentation():
    G, D, W = _tables(5, NCAA2005)
    assert D.sum(axis=1).tolist() == [-124, 91, -40, -17, 90]
    massey = L.massey_rating(G, D)
    assert massey == pytest.approx([-24.8, 18.2, -8.0, -3.4, 18.0], abs=1e-9)
    colley = L.colley_rating(W, G)
    assert colley == pytest.approx([0.214, 0.786, 0.5, 0.357, 0.643], abs=5.1e-4)      # gedruckt auf drei Stellen
    # gedruckte Plätze: Massey Miami, VT, UVA, UNC, Duke; Colley Miami, VT, UNC, UVA, Duke
    assert list(np.argsort(-massey)) == [1, 4, 3, 2, 0]
    assert list(np.argsort(-colley)) == [1, 4, 2, 3, 0]
