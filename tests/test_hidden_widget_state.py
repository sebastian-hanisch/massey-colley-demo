"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider} | {s.key for s in at.selectbox}


def test_widgets_are_only_drawn_where_they_act():
    at = fresh()
    assert at.selectbox(key="plan_select").value == "staffeln"
    keys = widget_keys(at)
    assert "x_slider" in keys and not {"p_slider", "stil_select", "h_slider"} & keys
    at.selectbox(key="plan_select").select("luecken").run(timeout=TIMEOUT)
    keys = widget_keys(at)
    assert "p_slider" in keys and "x_slider" not in keys
    at.slider(key="c_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert "stil_select" in widget_keys(at)
    at.slider(key="q_slider").set_value(0.2).run(timeout=TIMEOUT)
    assert "h_slider" in widget_keys(at)
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.slider(key="x_slider").set_value(4).run(timeout=TIMEOUT)
    at.selectbox(key="plan_select").select("vollstaendig").run(timeout=TIMEOUT)
    assert "x_slider" not in at.session_state
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    assert at.slider(key="x_slider").value == 4
    at.slider(key="c_slider").set_value(0.2).run(timeout=TIMEOUT)
    at.selectbox(key="stil_select").select("konter").run(timeout=TIMEOUT)
    at.slider(key="c_slider").set_value(0.0).run(timeout=TIMEOUT)
    assert "stil_select" not in at.session_state
    at.slider(key="c_slider").set_value(0.2).run(timeout=TIMEOUT)
    assert at.selectbox(key="stil_select").value == "konter"
    assert not at.exception


def test_preset_sets_hidden_widgets_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Gekonterter Stärkster"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="plan_select").value == "vollstaendig"
    assert at.selectbox(key="stil_select").value == "konter" and abs(at.slider(key="c_slider").value - 0.3) < 1e-9
    assert "x_slider" not in at.session_state
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Kantersiege"]].click().run(timeout=TIMEOUT)
    assert abs(at.slider(key="q_slider").value - 0.2) < 1e-9 and at.slider(key="h_slider").value == 10
    assert "stil_select" not in at.session_state
    assert not at.exception
