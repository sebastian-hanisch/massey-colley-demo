"""Analyse, Szenario und Verdikte; Randfälle der Regler."""

import itertools
import math

import numpy as np
import pytest

import mcl_constants as C
import mcl_evaluation as ev
import turnier_generator as T
from mcl_scenario import Settings, make_run

CASES = [Settings(n=n, k=k, plan=plan, x=x, p=0.5, c=c, stil="konter", q=q, h=10, seed=seed)
         for n, k, plan, x, c, q, seed in itertools.product((4, 8, 12), (1, 3), ("vollstaendig", "staffeln", "luecken"), (0, 1), (0.0, 0.3), (0.0, 0.2), (3, 4))]


def test_analysis_invariants_over_many_settings():
    for s in CASES:
        a = ev.analyse(s)
        n = s.n
        for m in C.METHODS:
            assert sorted(a.rank[m]) == list(range(n)), (s, m)
            assert 1 <= a.best_place[m] <= n
        # Colley: Mittel der Wertungen genau 1/2; Massey: Summe je Teil 0, also insgesamt 0
        assert a.scores["Colley"].mean() == pytest.approx(0.5, abs=1e-9)
        assert a.scores["Massey"].sum() == pytest.approx(0.0, abs=1e-9)
        assert a.components == T.components(a.run.turnier.games) == a.rank_deficit
        assert (s.plan == "staffeln") == (not math.isnan(a.crossing["Siege"]))


def test_without_a_connected_plan_no_method_beats_a_coin_flip_across_the_leagues():
    wrongs = [ev.analyse(Settings(n=12, plan="staffeln", x=0, seed=s)).crossing for s in range(100000, 100120)]
    for m in C.METHODS:
        assert abs(np.mean([w[m] for w in wrongs]) - 0.5) < 0.05, m


def test_complete_balanced_plan_gives_the_same_order_for_all_methods_without_ties():
    seen = 0
    for s in range(100000, 100200):
        a = ev.analyse(Settings(n=8, k=4, plan="vollstaendig", seed=s))
        if len(set(a.scores["Siege"])) == 8:
            assert a.rank["Massey"] == a.rank["Siege"] == a.rank["Colley"]
            seen += 1
    assert seen > 30


def test_analysis_is_cached_and_deterministic():
    s = Settings(n=8, seed=11)
    assert ev.analyse(s) is ev.analyse(s)
    r1, r2 = make_run(s), make_run(Settings(n=8, seed=11))
    assert np.array_equal(r1.turnier.wins, r2.turnier.wins) and np.array_equal(r1.D, r2.D)


def test_row_replaced_info_is_none_for_disconnected_plans_and_finite_otherwise():
    split = ev.analyse(Settings(n=8, plan="staffeln", x=0, seed=5))
    w, cond = ev.row_replaced_info(split)
    assert w is None and math.isinf(cond)
    connected = ev.analyse(Settings(n=8, plan="vollstaendig", seed=5))
    w, cond = ev.row_replaced_info(connected)
    assert w is not None and np.isfinite(cond)


def test_verdict_cases():
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=12, plan="staffeln", x=0, seed=3)))
    assert level == "warning" and "zerfällt" in text and "singulär" in text
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=12, plan="staffeln", x=1, seed=3)))
    assert "Falsch geordnete Paare über die beiden Staffeln" in text
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=8, plan="vollstaendig", k=4, seed=3)))
    assert "ändern Massey und Colley die Ordnung der Tabelle nicht" in text
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=8, plan="vollstaendig", k=4, q=0.2, seed=3)))
    assert "folgt Massey den Ausreißern" in text and "ändern Massey und Colley" not in text


def test_sweeps_have_consistent_shapes():
    rows = ev.sweep_staffeln(xs=(0, 1), seeds=C.SWEEP_SEEDS[:20])
    assert [r["x"] for r in rows] == [0, 1] and rows[0]["disconnected"] == 1.0
    rows = ev.sweep_luecken(ps=(1.0, 0.5), seeds=C.SWEEP_SEEDS[:20])
    assert rows[0]["disconnected"] == 0.0
    komp = ev.sweep_komplett(ks=(4,), seeds=C.SWEEP_SEEDS[:40])
    assert komp[0]["same_massey"] == komp[0]["same_colley"] == komp[0]["tie_free"]
    rows = ev.sweep_kantersiege(qs=(0.0, 0.2), seeds=C.SWEEP_SEEDS[:20])
    assert rows[0]["Massey"] == pytest.approx(rows[0]["Siege"], abs=0.01)
    rows = ev.sweep_residual(cs=(0.0,), stile=("ausgewogen",), seeds=C.SWEEP_SEEDS[:20])
    assert 0.0 <= rows[0]["residual"] <= 1.0


def test_decimal_comma_helpers():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 3) == "3,142" and ev.pct1(0.0526) == "5,3" and ev.pct(0.405) in ("40", "41")
