"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import mcl_evaluation as ev


@pytest.fixture(scope="module")
def staffeln():
    return {r["x"]: r for r in ev.sweep_staffeln()}


@pytest.fixture(scope="module")
def luecken():
    return {r["p"]: r for r in ev.sweep_luecken()}


@pytest.fixture(scope="module")
def komplett():
    return {r["k"]: r for r in ev.sweep_komplett()}


@pytest.fixture(scope="module")
def kanter():
    return {r["q"]: r for r in ev.sweep_kantersiege()}


@pytest.fixture(scope="module")
def residual():
    return {(r["stil"], r["c"]): r for r in ev.sweep_residual()}


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_without_crossing_games_all_methods_are_a_coin_flip_and_the_plan_falls_apart(staffeln):
    r = staffeln[0]
    assert r["disconnected"] == 1.0 and r["components"] == 2.0
    for m in ("Siege", "Siegquote", "Massey", "Colley"):
        assert r[m] == pytest.approx(0.50, abs=0.02), m


def test_one_crossing_game_per_strong_player_halves_the_error_for_massey(staffeln):
    r = staffeln[1]
    assert r["Siege"] == pytest.approx(0.37, abs=0.006) and r["Massey"] == pytest.approx(0.155, abs=0.006) and r["Colley"] == pytest.approx(0.24, abs=0.006)
    assert r["Massey"] < r["Colley"] < r["Siege"]
    assert staffeln[2]["Siege"] == pytest.approx(0.27, abs=0.006) and staffeln[2]["Massey"] == pytest.approx(0.12, abs=0.006)
    assert staffeln[3]["Siege"] == pytest.approx(0.18, abs=0.006) and staffeln[3]["Massey"] == pytest.approx(0.095, abs=0.006)


def test_with_all_crossing_games_the_plan_is_complete_and_all_methods_agree(staffeln):
    r = staffeln[6]
    assert r["Siege"] == pytest.approx(0.06, abs=0.006)
    assert r["Massey"] == r["Colley"] == r["Siege"] == r["Siegquote"]


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_with_random_gaps_massey_and_colley_beat_wins_but_hardly_the_win_share(luecken):
    r = luecken[0.5]
    assert r["Siege"] == pytest.approx(0.50, abs=0.006) and r["Siegquote"] == pytest.approx(0.58, abs=0.006)
    assert r["Massey"] == pytest.approx(0.585, abs=0.006) and r["Colley"] == pytest.approx(0.59, abs=0.006)
    assert r["Massey"] - r["Siegquote"] < 0.02
    assert luecken[0.2]["disconnected"] == pytest.approx(0.71, abs=0.006) and luecken[0.33]["disconnected"] == pytest.approx(0.16, abs=0.006)


def test_complete_plan_no_gain_and_identical_order(luecken, komplett):
    r = luecken[1.0]
    assert r["Siege"] == r["Siegquote"] == r["Massey"] == r["Colley"]
    for k in (2, 4, 10):
        assert komplett[k]["same_massey"] == komplett[k]["tie_free"] and komplett[k]["same_colley"] == komplett[k]["tie_free"]
    assert komplett[4]["tie_free"] == 86 and komplett[10]["tie_free"] == 188
    assert komplett[4]["tau_siege"] == pytest.approx(0.776, abs=0.006)


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_massey_falls_below_the_win_table_with_blowouts_colley_does_not(kanter):
    assert kanter[0.0]["Massey"] == kanter[0.0]["Siege"] == kanter[0.0]["Colley"]
    assert kanter[0.1]["Siege"] == pytest.approx(0.868, abs=0.006) and kanter[0.1]["Colley"] == pytest.approx(0.868, abs=0.006)
    assert kanter[0.1]["Massey"] == pytest.approx(0.72, abs=0.006)
    assert all(kanter[q]["Massey"] < kanter[q]["Siege"] for q in (0.1, 0.2, 0.3, 0.5))


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_the_residual_sees_balanced_cycles(residual):
    assert residual[("ausgewogen", 0.0)]["residual"] == pytest.approx(0.10, abs=0.006)
    assert residual[("ausgewogen", 0.2)]["residual"] == pytest.approx(0.33, abs=0.006)
    assert residual[("ausgewogen", 0.4)]["residual"] == pytest.approx(0.66, abs=0.006)
    assert residual[("ausgewogen", 0.4)]["tau_massey"] > 0.95


def test_the_residual_does_not_see_a_countered_strongest(residual):
    for c in (0.2, 0.4):
        assert residual[("konter", c)]["residual"] < 0.15
    assert residual[("konter", 0.2)]["tau_massey"] == pytest.approx(0.73, abs=0.006)
    assert residual[("konter", 0.4)]["tau_massey"] == pytest.approx(0.515, abs=0.006)
    assert residual[("konter", 0.4)]["best_place"] == pytest.approx(5.8, abs=0.06)
    assert residual[("konter", 0.4)]["tau_e_massey"] > 0.9


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(staffeln, luecken, komplett, kanter, residual):
    level, text = ev.verdict_staffeln(list(staffeln.values()))
    assert level == "warning" and ev.pct(staffeln[1]["Siege"]) in text and ev.pct(staffeln[1]["Massey"]) in text
    level, text = ev.verdict_luecken(list(luecken.values()), list(komplett.values()))
    assert ev.de(luecken[0.5]["Siegquote"]) in text and "86 bzw. 86 von 86" in text
    level, text = ev.verdict_kantersiege(list(kanter.values()))
    assert ev.de(kanter[0.1]["Massey"], 3) in text and ev.de(kanter[0.1]["Siege"], 3) in text
    level, text = ev.verdict_residual(list(residual.values()))
    assert ev.de(residual[("ausgewogen", 0.4)]["residual"]) in text and ev.de(residual[("konter", 0.4)]["residual"]) in text
