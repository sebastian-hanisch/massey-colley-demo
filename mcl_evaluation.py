"""Kennzahlen, Verdikt-Logik und Experimente (über die festen Instanzen SWEEP_SEEDS)."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import mcl_constants as C
import mcl_linear as L
import turnier_generator as T
from mcl_scenario import Run, Settings, make_run

_LOT_SALT = 0xC0FFEE


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    run: Run
    scores: dict           # Verfahren -> Wertung je Teilnehmer
    rank: dict             # Verfahren -> Platz je Teilnehmer (0 = bester)
    tau_s: dict            # Verfahren -> τ gegen die Generator-Stärke
    tau_e: dict            # Verfahren -> τ gegen die erwartete Siegquote
    crossing: dict         # Verfahren -> falsch geordnete Staffel-Paare (nan außer Plan "zwei Staffeln")
    components: int        # Zahl der Teile des Spielplans
    rank_deficit: int      # n minus Rang der Laplace-Matrix (gleich der Zahl der Teile)
    residual: float        # Anteil der Paar-Differenzen, den das Massey-Modell nicht erklärt
    best_place: dict       # Verfahren -> Platz des Stärksten (1 = Spitze)


def _scores(run: Run) -> dict:
    t = run.turnier
    return {
        "Siege": t.wins.sum(axis=1).astype(float),
        "Siegquote": L.win_share(t.wins, t.games),
        "Massey": L.massey_rating(t.games, run.D),
        "Colley": L.colley_rating(t.wins, t.games),
    }


@functools.lru_cache(maxsize=16384)
def analyse(settings: Settings) -> Analysis:
    """Gleiche Reglerwerte, gleiche Analyse; der Zwischenspeicher lässt die Experimente Turniere teilen (Settings ist unveränderlich)."""
    run = make_run(settings)
    t = run.turnier
    n = settings.n
    scores = _scores(run)
    rank = {m: L.rank_with_tolerance(scores[m], T.SplitMix64(settings.seed ^ _LOT_SALT), C.TOL) for m in C.METHODS}
    n_weak = n // 2
    components = T.components(t.games)
    return Analysis(
        settings=settings, run=run, scores=scores, rank=rank,
        tau_s={m: L.kendall_tau(rank[m], t.strengths) for m in C.METHODS},
        tau_e={m: L.kendall_tau(rank[m], t.expected) for m in C.METHODS},
        crossing={m: (L.crossing_wrong(rank[m], n_weak) if settings.plan == "staffeln" else float("nan")) for m in C.METHODS},
        components=components,
        rank_deficit=L.matrix_rank_deficit(L.laplacian(t.games)),
        residual=L.residual_share(t.games, run.D, scores["Massey"]),
        best_place={m: rank[m][n - 1] + 1 for m in C.METHODS},
    )


def row_replaced_info(a: Analysis) -> tuple:
    """Originalweg von Massey (letzte Zeile durch Einsen): (Wertung oder None, Kondition). Nur für die Anzeige der Singularität."""
    return L.massey_row_replaced(a.run.turnier.games, a.run.D)


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


def _runs(make_settings, seeds):
    return [analyse(make_settings(s)) for s in seeds]


# --- Experiment 1: ungleiche Gegner (zwei Staffeln) ---------------------------------------------------------------------------------------

def sweep_staffeln(xs=C.X_SWEEP, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for x in xs:
        runs = _runs(lambda s: Settings(n=n, delta=delta, k=1, plan="staffeln", x=x, seed=s), seeds)
        row = {"x": x, "components": _mean([float(r.components) for r in runs]), "disconnected": _mean([float(r.components > 1) for r in runs])}
        for m in C.METHODS:
            row[m] = _mean([r.crossing[m] for r in runs])
        rows.append(row)
    return rows


# --- Experiment 2: Lücken, und das Vollständige --------------------------------------------------------------------------------------------

def sweep_luecken(ps=C.P_SWEEP, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for p in ps:
        runs = _runs(lambda s: Settings(n=n, delta=delta, k=1, plan="luecken", p=p, seed=s), seeds)
        row = {"p": p, "disconnected": _mean([float(r.components > 1) for r in runs])}
        for m in C.METHODS:
            row[m] = _mean([r.tau_s[m] for r in runs])
        rows.append(row)
    return rows


def sweep_komplett(ks=C.K_COMPLETE_SWEEP, n=C.EXP_N_COMPLETE, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Vollständig und ausgeglichen: Wie oft ist die Ordnung von Massey und Colley die der Siegtabelle (nur Turniere ohne Gleichstand)?"""
    rows = []
    for k in ks:
        runs = _runs(lambda s: Settings(n=n, delta=delta, k=k, plan="vollstaendig", seed=s), seeds)
        free = [r for r in runs if len(set(r.scores["Siege"])) == n]
        rows.append({"k": k, "tie_free": len(free), "runs": len(runs),
                     "same_massey": sum(r.rank["Massey"] == r.rank["Siege"] for r in free),
                     "same_colley": sum(r.rank["Colley"] == r.rank["Siege"] for r in free),
                     "tau_siege": _mean([r.tau_s["Siege"] for r in runs]), "tau_massey": _mean([r.tau_s["Massey"] for r in runs]),
                     "tau_colley": _mean([r.tau_s["Colley"] for r in runs])})
    return rows


# --- Experiment 3: Kantersiege --------------------------------------------------------------------------------------------------------------

def sweep_kantersiege(qs=C.Q_SWEEP, n=C.EXP_N, k=C.EXP4_K, h=C.DEFAULT_H, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for q in qs:
        runs = _runs(lambda s: Settings(n=n, delta=delta, k=k, plan="vollstaendig", q=q, h=h, seed=s), seeds)
        rows.append({"q": q, **{m: _mean([r.tau_s[m] for r in runs]) for m in C.METHODS}})
    return rows


# --- Experiment 4: was das Residuum sieht ---------------------------------------------------------------------------------------------------

def sweep_residual(cs=C.C_SWEEP, stile=("ausgewogen", "konter"), n=C.EXP_N, k=C.EXP5_K, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for stil in stile:
        for c in cs:
            runs = _runs(lambda s: Settings(n=n, delta=delta, k=k, plan="vollstaendig", c=c, stil=stil, seed=s), seeds)
            rows.append({"stil": stil, "c": c, "residual": _mean([r.residual for r in runs]),
                         "tau_siege": _mean([r.tau_s["Siege"] for r in runs]), "tau_massey": _mean([r.tau_s["Massey"] for r in runs]),
                         "tau_e_massey": _mean([r.tau_e["Massey"] for r in runs]), "best_place": _mean([float(r.best_place["Massey"]) for r in runs])})
    return rows


# --- Verdikte (live aus den Messwerten) --------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Hilft das lineare System, und wo ist es gleich der Tabelle oder blind?"""
    s = a.settings
    n = s.n
    if a.components > 1:
        return "warning", (f"Der Spielplan zerfällt in **{a.components} Teile**, die nie gegeneinander gespielt haben: Die Laplace-Matrix hat den Rang "
                           f"{n - a.rank_deficit} statt {n}, das Massey-System ist singulär und hat keine eindeutige Lösung. Alle Verfahren ordnen die Teile "
                           "untereinander wie einen Münzwurf, ohne es zu merken.")
    if s.plan == "staffeln":
        c = a.crossing
        return "success" if c["Massey"] < c["Siege"] else "info", (
            f"Falsch geordnete Paare über die beiden Staffeln: Siegtabelle **{pct(c['Siege'])} %**, Massey **{pct(c['Massey'])} %**, "
            f"Colley **{pct(c['Colley'])} %**. Die Gegnerstärke fließt ins System ein; die Tabelle zählt jeden Sieg gleich.")
    t = a.tau_s
    text = (f"τ gegen die Generator-Stärke: Siegtabelle {de(t['Siege'])}, Siegquote {de(t['Siegquote'])}, Massey {de(t['Massey'])}, "
            f"Colley {de(t['Colley'])}.")
    if s.plan == "vollstaendig" and s.q == 0 and s.c == 0:
        text += " Bei vollständigem, ausgeglichenem Plan ändern Massey und Colley die Ordnung der Tabelle nicht."
    if s.q > 0:
        text += f" Mit {pct(s.q)} % Kantersiegen folgt Massey den Ausreißern; Colley und die Tabelle nutzen nur Sieg oder Niederlage."
    return "info", text


def verdict_staffeln(rows) -> tuple[str, str]:
    r0, r1 = next(r for r in rows if r["x"] == 0), next(r for r in rows if r["x"] == 1)
    return "warning", (f"Ohne Kreuzpartie ordnen alle Verfahren die Staffeln wie einen Münzwurf (Siegtabelle {pct(r0['Siege'])} %, Massey {pct(r0['Massey'])} % "
                       f"falsch): Der Plan zerfällt, die Matrix ist singulär. Mit **einer** Kreuzpartie je Starkem sinkt der Anteil falsch geordneter Paare "
                       f"von {pct(r1['Siege'])} % (Siegtabelle) auf {pct(r1['Massey'])} % (Massey) und {pct(r1['Colley'])} % (Colley).")


def verdict_luecken(rows, komplett) -> tuple[str, str]:
    half = next(r for r in rows if r["p"] == 0.5)
    k4 = next((r for r in komplett if r["k"] == 4), komplett[-1])
    return "info", (f"Bei zufälligen Lücken (p = 0,5) liegt τ gegen die Stärke für die Siegtabelle bei {de(half['Siege'])}, für die Siegquote bei "
                    f"{de(half['Siegquote'])}, für Massey bei {de(half['Massey'])} und für Colley bei {de(half['Colley'])}: ein Gewinn gegenüber den Siegen, "
                    f"gegenüber der Siegquote kaum. Im vollständigen, ausgeglichenen Turnier ist die Ordnung von Massey und Colley in {k4['same_massey']} bzw. "
                    f"{k4['same_colley']} von {k4['tie_free']} gleichstandsfreien Turnieren (4 Partien je Paar) die der Siegtabelle: dort bringen sie nichts.")


def verdict_kantersiege(rows) -> tuple[str, str]:
    r0 = next(r for r in rows if r["q"] == 0.0)
    r1 = next(r for r in rows if r["q"] == 0.1)
    return "warning", (f"Ohne Kantersiege liegt Massey bei τ = {de(r0['Massey'], 3)} (Siegtabelle {de(r0['Siege'], 3)}). Mit 10 % Kantersiegen fällt Massey auf "
                       f"{de(r1['Massey'], 3)}, **unter** die Siegtabelle ({de(r1['Siege'], 3)}); Colley bleibt bei {de(r1['Colley'], 3)}, weil es nur Sieg "
                       "oder Niederlage nutzt. Das Kantersieg-Modell ist ein Stresstest (unabhängig von der Stärke), keine Aussage über reale Spiele.")


def verdict_residual(rows) -> tuple[str, str]:
    bal = next(r for r in rows if r["stil"] == "ausgewogen" and r["c"] == 0.4)
    bal0 = next(r for r in rows if r["stil"] == "ausgewogen" and r["c"] == 0.0)
    cnt = next(r for r in rows if r["stil"] == "konter" and r["c"] == 0.4)
    return "warning", (f"Bei ausgewogenen Stilen steigt das Residuum mit der Zyklenstärke von {de(bal0['residual'], 2)} auf {de(bal['residual'], 2)}: Das "
                       f"lineare Modell erklärt die Zyklen nicht. Ist der Stärkste aber gekontert, bleibt es bei {de(cnt['residual'], 2)}, obwohl τ gegen die "
                       f"Stärke auf {de(cnt['tau_massey'], 2)} einbricht und der Stärkste im Mittel auf Platz {de(cnt['best_place'], 1)} steht: Der Konter "
                       "verschwindet im transitiven Anteil.")
