"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 16, 12
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
K_MIN, K_MAX, DEFAULT_K = 1, 20, 1
X_MIN, X_MAX, DEFAULT_X = 0, 8, 1                 # Kreuzpartien je Spieler der starken Staffel (nur Plan "staffeln")
P_MIN, P_MAX, P_STEP, DEFAULT_P = 0.1, 1.0, 0.05, 0.5   # Spielwahrscheinlichkeit je Paar (nur Plan "luecken")
CYCLE_MIN, CYCLE_MAX, CYCLE_STEP, DEFAULT_CYCLE = 0.0, 0.5, 0.05, 0.0
Q_MIN, Q_MAX, Q_STEP, DEFAULT_Q = 0.0, 0.5, 0.05, 0.0    # Anteil der Siege, die Kantersiege sind (Stresstest)
H_MIN, H_MAX, DEFAULT_H = 2, 20, 10                       # Höhe eines Kantersiegs in Punkten (nur bei q > 0)
SEED_MAX, DEFAULT_SEED = 999999, 35

PLAN_LABELS = {"vollstaendig": "Vollständig: jeder gegen jeden", "staffeln": "Zwei Staffeln", "luecken": "Zufällige Lücken"}
STIL_LABELS = {
    "ausgewogen": "Ausgewogen (reihum)",
    "konter": "Konter-Stil dominiert (der Stärkste wird gekontert)",
    "beute": "Beute-Stil dominiert (der Stärkste kontert)",
}
METHODS = ("Siege", "Siegquote", "Massey", "Colley")
TOL = 1e-9                                  # Wertungen, die sich um weniger unterscheiden, sind gleich (Gleitkomma-Rauschen)

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100300))
EXP_N = 12                  # Experimente 1, 2, 4, 5: feste Instanz mit 12 Teilnehmern
EXP_N_COMPLETE = 8          # Experiment 2, Zeile "vollständig": 8 Teilnehmer
EXP_DELTA = 0.3
X_SWEEP = (0, 1, 2, 3, 6)
P_SWEEP = (1.0, 0.75, 0.5, 0.33, 0.2)
K_COMPLETE_SWEEP = (1, 2, 4, 10)
Q_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.5)
EXP4_K = 4
EXP5_K = 20
C_SWEEP = (0.0, 0.2, 0.4)

COLOR_SIEGE = "rgba(150,150,150,0.9)"
COLOR_QUOTE = "#8e6bb0"
COLOR_MASSEY = "#c0392b"
COLOR_COLLEY = "#2F6B65"
COLOR_MAIN = "#2F6B65"
METHOD_COLORS = {"Siege": COLOR_SIEGE, "Siegquote": COLOR_QUOTE, "Massey": COLOR_MASSEY, "Colley": COLOR_COLLEY}

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "k": DEFAULT_K, "plan": "staffeln", "x": DEFAULT_X, "p": DEFAULT_P, "c": 0.0, "stil": "ausgewogen",
         "q": 0.0, "h": DEFAULT_H, "seed": DEFAULT_SEED}
PRESETS = {
    "Zwei Staffeln, eine Kreuzpartie": dict(_BASE),
    "Zerfallender Plan": {**_BASE, "x": 0},
    "Kantersiege": {**_BASE, "plan": "vollstaendig", "k": 4, "q": 0.2},
    "Gekonterter Stärkster": {**_BASE, "plan": "vollstaendig", "k": 20, "c": 0.3, "stil": "konter"},
}
PRESET_HELP = {
    "Zwei Staffeln, eine Kreuzpartie": "12 Teilnehmer in einer starken und einer schwachen Staffel, jeder Starke spielt nur einmal gegen einen Schwachen: die Siegtabelle ordnet die Staffeln schlecht, das lineare System deutlich besser.",
    "Zerfallender Plan": "Dieselben Staffeln ohne eine einzige Kreuzpartie: Der Plan zerfällt in zwei Teile, die Matrix wird singulär, kein Verfahren kann die Staffeln ordnen.",
    "Kantersiege": "Vollständiges Turnier, 4 Partien je Paar, jeder fünfte Sieg ein Kantersieg: Massey folgt den Ausreißern, die Siegtabelle und Colley nicht.",
    "Gekonterter Stärkster": "Vollständiges Turnier, 20 Partien je Paar, drei Spielstile mit Konter: das Residuum bleibt klein, obwohl der Stärkste abrutscht.",
}
