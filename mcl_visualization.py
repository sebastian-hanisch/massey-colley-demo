"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mcl_constants as C
import mcl_linear as L


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_places(a):
    """Platz je Verfahren für jeden Teilnehmer, in der wahren Reihenfolge (Stärkster links): auf der Diagonale stimmt das Verfahren."""
    t = a.run.turnier
    n = t.n
    order = list(range(n - 1, -1, -1))
    names = [t.names[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names, y=list(range(1, n + 1)), mode="lines", line=dict(color="rgba(150,150,150,0.5)", dash="dash"), name="wahre Ordnung",
                             hoverinfo="skip"))
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=names, y=[a.rank[m][i] + 1 for i in order], mode="markers", name=m,
                                 marker=dict(size=10 if m in ("Massey", "Colley") else 7, color=C.METHOD_COLORS[m],
                                             symbol={"Siege": "circle-open", "Siegquote": "square-open", "Massey": "diamond", "Colley": "circle"}[m]),
                                 hovertemplate="%{x}: Platz %{y} (" + m + ")<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Teilnehmer (wahre Reihenfolge, Stärkster links)")
    fig.update_yaxes(title_text="Platz", autorange="reversed", dtick=1)
    return _base(fig, height=380)


def build_matrices(a):
    """Die beiden Systemmatrizen nebeneinander (nur für kleine n): Massey M = Laplace-Matrix, Colley C = 2I + M."""
    t = a.run.turnier
    n = t.n
    order = list(range(n - 1, -1, -1))
    names = [t.names[i] for i in order]
    M = L.laplacian(t.games)[np.ix_(order, order)]
    Cm = (2.0 * np.eye(n) + L.laplacian(t.games))[np.ix_(order, order)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Massey: M (Laplace-Matrix)", "Colley: C = 2I + M"), horizontal_spacing=0.12)
    for col, mat in ((1, M), (2, Cm)):
        fig.add_trace(go.Heatmap(z=mat, x=names, y=names, text=[[f"{v:g}" for v in row] for row in mat], texttemplate="%{text}", colorscale="RdBu_r",
                                 zmid=0, showscale=False, hovertemplate="%{y}, %{x}: %{text}<extra></extra>", xgap=1, ygap=1), row=1, col=col)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top")
    return _base(fig, height=max(300, 36 * n + 120))


def build_staffeln(rows):
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    for m in ("Siege", "Massey", "Colley"):
        fig.add_trace(go.Bar(x=xs, y=[r[m] for r in rows], name=m, marker_color=C.METHOD_COLORS[m],
                             hovertemplate=m + ", %{x} Kreuzpartien: %{y:.0%} falsch<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#c0392b", annotation_text="Münzwurf", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Kreuzpartien je Spieler der starken Staffel", type="category")
    fig.update_yaxes(title_text="falsch geordnete Paare über die Staffeln", tickformat=".0%", range=[0, 0.6])
    return _base(fig)


def build_luecken(rows):
    xs = [f"{r['p']:.2f}" for r in rows]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], mode="lines+markers", name=m, line=dict(color=C.METHOD_COLORS[m], width=3 if m in ("Massey", "Colley") else 2,
                                 dash="solid" if m in ("Massey", "Colley") else "dot"), hovertemplate=m + ", p = %{x}: τ = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Wahrscheinlichkeit, dass ein Paar spielt", type="category")
    fig.update_yaxes(title_text="τ gegen die Generator-Stärke", range=[0, 0.8])
    return _base(fig)


def build_kantersiege(rows):
    xs = [f"{r['q']:.1f}" for r in rows]
    fig = go.Figure()
    for m in ("Siege", "Massey", "Colley"):
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], mode="lines+markers", name=m, line=dict(color=C.METHOD_COLORS[m], width=3),
                                 hovertemplate=m + ", q = %{x}: τ = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Siege, die Kantersiege sind (Höhe 10)", type="category")
    fig.update_yaxes(title_text="τ gegen die Generator-Stärke", range=[0.6, 0.95])
    return _base(fig)


def build_residual(rows):
    labels = [f"{r['stil']}, c = {r['c']:.1f}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["residual"] for r in rows], name="Residuum des Massey-Modells", marker_color=C.COLOR_MASSEY,
                         hovertemplate="%{x}: Residuum %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[r["tau_massey"] for r in rows], mode="lines+markers", name="τ gegen die Stärke", line=dict(color="#222222", width=3),
                             hovertemplate="%{x}: τ = %{y:.2f}<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Stilverteilung und Zyklenstärke")
    fig.update_yaxes(title_text="Residuum bzw. τ", range=[0, 1.05])
    return _base(fig, height=400)
