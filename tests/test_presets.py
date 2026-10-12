"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt, was seine Beschreibung verspricht."""

import dataclasses

import mcl_constants as C
import mcl_evaluation as ev
import mcl_presets as P
from mcl_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX, name
        assert C.K_MIN <= p["k"] <= C.K_MAX and C.X_MIN <= p["x"] <= C.X_MAX and C.P_MIN <= p["p"] <= C.P_MAX, name
        assert p["plan"] in C.PLAN_LABELS and p["stil"] in C.STIL_LABELS, name
        assert C.CYCLE_MIN <= p["c"] <= C.CYCLE_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX and C.H_MIN <= p["h"] <= C.H_MAX, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Zwei Staffeln, eine Kreuzpartie"]) == Settings()


def test_setting_specs_cover_every_preset_key_and_settings_fields_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_widgets():
    assert set(P.KEPT_KEYS) == {"x_slider", "p_slider", "stil_select", "h_slider"}


def _mean(preset, attr, seeds=range(100000, 100080)):
    vals = [attr(ev.analyse(Settings(**{**C.PRESETS[preset], "seed": s}))) for s in seeds]
    return sum(vals) / len(vals)


def test_each_preset_tells_its_story():
    staffeln = "Zwei Staffeln, eine Kreuzpartie"
    assert _mean(staffeln, lambda a: a.crossing["Massey"]) < _mean(staffeln, lambda a: a.crossing["Siege"]) - 0.1
    assert _mean("Zerfallender Plan", lambda a: float(a.components)) == 2.0
    assert _mean("Zerfallender Plan", lambda a: a.crossing["Massey"]) > 0.4          # Münzwurf
    blow = "Kantersiege"
    assert _mean(blow, lambda a: a.tau_s["Massey"]) < _mean(blow, lambda a: a.tau_s["Siege"]) - 0.05
    assert abs(_mean(blow, lambda a: a.tau_s["Colley"]) - _mean(blow, lambda a: a.tau_s["Siege"])) < 0.02
    counter = "Gekonterter Stärkster"
    assert _mean(counter, lambda a: a.residual) < 0.2
    assert _mean(counter, lambda a: a.tau_s["Massey"]) < 0.85
    assert _mean(counter, lambda a: float(a.best_place["Massey"])) > 2.0
