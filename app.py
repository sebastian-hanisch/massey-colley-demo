"""Massey und Colley: ein lineares System auf dem Spielplan - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Ranking-Linie: Zwei lineare Bewertungssysteme auf dem Spielgraphen beheben die ungleichen Gegner der Siegtabelle; vier Experimente messen
wo, und wo nicht. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import mcl_constants as C
import mcl_evaluation as ev
from mcl_pdf_export import build_pdf
from mcl_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from mcl_scenario import Settings
from mcl_visualization import build_kantersiege, build_luecken, build_matrices, build_places, build_residual, build_staffeln

st.set_page_config(page_title="Massey und Colley – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _staffeln():
    return ev.sweep_staffeln()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _luecken():
    return ev.sweep_luecken(), ev.sweep_komplett()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _kantersiege():
    return ev.sweep_kantersiege()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _residual():
    return ev.sweep_residual()


def verdict(level, text):
    SHOW[level](text)


st.title("🧮 Massey und Colley: ein lineares System auf dem Spielplan")
st.markdown(
    """
Die Siegtabelle zählt jeden Sieg gleich, gleich gegen wen. Diese Demo ist das **vierte Stück der Ranking-Linie** und zeigt, was ein **lineares
Gleichungssystem auf dem Spielgraphen** ändert: **Massey** legt die Kleinsten Quadrate auf die Punktedifferenzen, **Colley** glättet die Siegquote
und rechnet die Stärke der Gegner ein. Beide werden **in einem Schritt gelöst**, ohne Iteration. Das Ergebnis ist ungleichmäßig: Bei ungleichen Gegnern
helfen sie deutlich, bei vollständigem, ausgeglichenem Plan ändern sie ohne Kantersiege **nichts**, und Massey **kippt bei Kantersiegen**.

Weiter unten: vier **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So rechnet man mit einem linearen System", expanded=True):
    st.markdown(
        """
1. **Der Spielplan ist ein Graph:** Knoten sind die Teilnehmer, eine Kante steht für gespielte Partien. Seine **Laplace-Matrix** M hat die Zahl der
   Partien auf der Diagonale und minus die Zahl der Partien zwischen zwei Teilnehmern daneben.
2. **Massey:** M · r = p, wobei p die Summe der Punktedifferenzen je Teilnehmer ist. Gesucht ist r, sodass r_i − r_j die Punktedifferenz jedes Paares
   im Sinne der Kleinsten Quadrate trifft. M ist singulär (nur Differenzen zählen), deshalb wird die Summe auf 0 festgelegt.
3. **Colley:** C · r = b mit C = 2I + M und b_i = 1 + (Siege − Niederlagen)/2, das ist die Siegquote nach Laplace (+1 und +2) mit Gegnerstärke.
   C ist immer lösbar; die Wertungen liegen um 1/2.
4. **Was beide gemeinsam haben:** eine Gegnerstärke, die über die Kanten des Graphen weitergegeben wird. Ohne Verbindung zwischen zwei Teilen des Plans
   gibt es nichts weiterzugeben.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Teilnehmer**")
    n = st.slider("Teilnehmer n", *bounds("n_slider"), key="n_slider", help="Wie viele Teilnehmer im Turnier antreten.")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Teilnehmer (logistische Skala).")
    st.markdown("**Spielplan**")
    plan = st.selectbox("Spielplan", list(C.PLAN_LABELS), format_func=C.PLAN_LABELS.get, key="plan_select")
    k = st.slider("Partien je Paar", *bounds("k_slider"), key="k_slider", help="Wie oft ein Paar gegeneinander spielt, wenn es überhaupt spielt.")
    if plan == "staffeln":
        seed_widget("x_slider")
        x = st.slider("Kreuzpartien je Spieler der starken Staffel", *bounds("x_slider"), key="x_slider",
                      help="Gegen wie viele zufällige Spieler der schwachen Staffel jeder Starke zusätzlich antritt. 0 = die Staffeln begegnen sich nie.")
        remember("x_slider", x)
    if plan == "luecken":
        seed_widget("p_slider")
        p = st.slider("Wahrscheinlichkeit, dass ein Paar spielt", *bounds("p_slider"), step=C.P_STEP, key="p_slider",
                      help="Jedes Paar spielt mit dieser Wahrscheinlichkeit (dann k-mal), sonst gar nicht.")
        remember("p_slider", p)
    st.markdown("**Zyklen**")
    c = st.slider("Zyklenstärke c", *bounds("c_slider"), step=C.CYCLE_STEP, key="c_slider",
                  help="Wie stark drei Spielstile einander kontern. 0 = keine Zyklen.")
    if c > 0:
        seed_widget("stil_select")
        stil = st.selectbox("Stilverteilung", list(C.STIL_LABELS), format_func=C.STIL_LABELS.get, key="stil_select",
                            help="Wie die drei Spielstile verteilt sind. Bei Konter und Beute hat der Stärkste Stil 0; bei „ausgewogen“ sind die Stile reihum vergeben.")
        remember("stil_select", stil)
    st.markdown("**Kantersiege (Stresstest)**")
    q = st.slider("Anteil der Siege, die Kantersiege sind", *bounds("q_slider"), step=C.Q_STEP, key="q_slider",
                  help="Unabhängig von der Stärke: ein Ausreißermodell, um Massey zu belasten. 0 = jeder Sieg zählt einen Punkt Vorsprung.")
    if q > 0:
        seed_widget("h_slider")
        h = st.slider("Höhe eines Kantersiegs (Punkte)", *bounds("h_slider"), key="h_slider")
        remember("h_slider", h)
    seed = st.number_input("Zufalls-Seed des Turniers", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Turnier würfeln", width="stretch", on_click=randomize_seed)

x_value = int(current_value("x_slider"))
p_value = round(float(current_value("p_slider")), 2)
stil_value = str(current_value("stil_select"))
h_value = int(current_value("h_slider"))
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "k_slider": int(k), "plan_select": plan, "x_slider": x_value,
                   "p_slider": p_value, "c_slider": round(float(c), 2), "stil_select": stil_value, "q_slider": round(float(q), 2), "h_slider": h_value,
                   "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(k), plan, x_value, p_value, round(float(c), 2), stil_value, round(float(q), 2), h_value, int(seed))
a = _analysis(settings)
t = a.run.turnier

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Wertungen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Teile des Spielplans", a.components, delta=f"Rang von M: {t.n - a.rank_deficit} von {t.n}", delta_color="off",
          help="Zahl der Zusammenhangskomponenten; die Laplace-Matrix hat den Rang n minus diese Zahl.")
m2.metric("τ gegen die Stärke: Siegtabelle", ev.de(a.tau_s["Siege"]), delta=f"Massey {ev.de(a.tau_s['Massey'])}, Colley {ev.de(a.tau_s['Colley'])}",
          delta_color="off")
m3.metric("Falsch geordnete Staffel-Paare (Tabelle)", "–" if math.isnan(a.crossing["Siege"]) else f"{ev.pct(a.crossing['Siege'])} %",
          delta=("nur bei zwei Staffeln" if plan != "staffeln" else f"Massey {ev.pct(a.crossing['Massey'])} %, Colley {ev.pct(a.crossing['Colley'])} %"),
          delta_color="off")
m4.metric("Residuum des Massey-Modells", "–" if math.isnan(a.residual) else ev.de(a.residual),
          delta="Anteil, den das lineare Modell nicht erklärt", delta_color="off")

order = list(range(t.n - 1, -1, -1))
df = pd.DataFrame({
    "Teilnehmer": [t.names[i] for i in order],
    "Platz nach Stärke": list(range(1, t.n + 1)),
    "Siege": [int(a.scores["Siege"][i]) for i in order],
    "Partien": [int(t.games.sum(axis=1)[i]) for i in order],
    "Siegquote": [round(float(a.scores["Siegquote"][i]), 2) for i in order],
    "Massey r": [round(float(a.scores["Massey"][i]), 3) for i in order],
    "Colley r": [round(float(a.scores["Colley"][i]), 3) for i in order],
    "Platz Siegtabelle": [a.rank["Siege"][i] + 1 for i in order],
    "Platz Massey": [a.rank["Massey"][i] + 1 for i in order],
    "Platz Colley": [a.rank["Colley"][i] + 1 for i in order],
    "Stärke s": [round(float(t.strengths[i]), 2) for i in order],
})
st.dataframe(df, hide_index=True, width="stretch")
st.plotly_chart(build_places(a), width="stretch", key="places_chart")
st.caption("Je Teilnehmer der Platz nach jedem Verfahren, in der wahren Reihenfolge; auf der gestrichelten Linie stimmt der Platz. Die wahre Stärke ist hier "
           "bekannt, weil das Turnier simuliert ist.")

if t.n <= 8:
    st.markdown("**Die beiden Systemmatrizen** (nur für n ≤ 8)")
    st.plotly_chart(build_matrices(a), width="stretch", key="matrix_chart")
    w, cond = ev.row_replaced_info(a)
    if w is None:
        st.warning("Der Originalweg von Massey (letzte Zeile durch Einsen ersetzen) scheitert hier: Die Matrix ist singulär, weil der Plan in Teile zerfällt. "
                   "Die Demo rechnet mit der Pseudoinverse (Minimum-Norm-Lösung).")
    else:
        st.caption(f"Originalweg von Massey (letzte Zeile durch Einsen ersetzen): lösbar, Kondition {cond:.1f}. Die Demo rechnet mit der Pseudoinverse, "
                   "die hier dasselbe liefert.")
st.download_button("📄 PDF der aktuellen Wertungen", data=build_pdf(a), file_name="massey_colley.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hilft das lineare System, und wo ändert es nichts?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ungleiche Gegner, zwei Staffeln")
st.caption("12 Teilnehmer (feste Instanz, unabhängig vom Regler), eine starke und eine schwache Staffel, innerhalb jede gegen jeden, dazu x Kreuzpartien je "
           "Starkem; Mittel über 300 feste Turniere.")
rows1 = _staffeln()
st.plotly_chart(build_staffeln(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_staffeln(rows1))

st.subheader("🔬 Experiment 2: Zufällige Lücken, und das Vollständige")
st.caption("12 Teilnehmer (feste Instanz), jedes Paar spielt mit Wahrscheinlichkeit p einmal. Darunter: bei vollständigem, ausgeglichenem Plan (8 Teilnehmer) "
           "die Ordnung von Massey und Colley gegen die der Siegtabelle.")
rows2, komplett = _luecken()
st.plotly_chart(build_luecken(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_luecken(rows2, komplett))
st.dataframe(pd.DataFrame([{"Partien je Paar": r["k"], "Turniere ohne Gleichstand": f"{r['tie_free']} von {r['runs']}",
                            "Massey = Siegtabelle": f"{r['same_massey']} von {r['tie_free']}", "Colley = Siegtabelle": f"{r['same_colley']} von {r['tie_free']}",
                            "τ Siegtabelle": round(r["tau_siege"], 3), "τ Massey": round(r["tau_massey"], 3), "τ Colley": round(r["tau_colley"], 3)}
                           for r in komplett]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Kantersiege")
st.caption("12 Teilnehmer (feste Instanz), vollständig, 4 Partien je Paar, Kantersiege der Höhe 10 mit Anteil q, **unabhängig von der Stärke** (Stresstest).")
rows3 = _kantersiege()
st.plotly_chart(build_kantersiege(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_kantersiege(rows3))

st.subheader("🔬 Experiment 4: Was das Residuum sieht")
st.caption("12 Teilnehmer (feste Instanz), vollständig, 20 Partien je Paar, damit das Rauschen kleiner wird. Das Residuum ist der Anteil der Paar-Differenzen, "
           "den das Massey-Modell nicht erklärt.")
rows4 = _residual()
st.plotly_chart(build_residual(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_residual(rows4))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Plan ist zusammenhängend** | Zerfällt er, ist die Matrix singulär und keine Reihenfolge über die Teile hinweg belegt; beide Verfahren raten (Experiment 1). | – |
| **Punktedifferenzen sind zuverlässig** | Massey folgt den Ausreißern; schon 10 % Kantersiege der Höhe 10 setzen es unter die Siegtabelle (Experiment 3). | Bradley-Terry |
| **Der Plan ist ungleich** | Im vollständigen, ausgeglichenen Plan ändern beide Verfahren ohne Kantersiege die Ordnung der Tabelle nicht (Experiment 2). | – |
| **Die Leistung hängt nur von der Stärke ab** | Bei Konter-Stilen wird der Stärkste fehlgeordnet, und das Residuum merkt es nicht (Experiment 4). | Nicht-Transitivität |
| **Ein Spiel ist Sieg oder Niederlage mit Punkten** | Tore, Remis und Heimvorteil haben ein eigenes Modell; hier steckt nur Sieg oder Niederlage hinter den Punkten. | Tore, Remis, Heimvorteil |
| **Alles ist simuliert** | Die Stärke ist eine Konstruktion des Generators; die Punktedifferenz entsteht aus den Ausgängen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Spielgraph.** $G_{ij}$ ist die Zahl der Partien von $i$ gegen $j$, $W_{ij}$ die Siege von $i$ gegen $j$, $D_{ij}$ die Punkte von $i$ gegen $j$ minus
die von $j$ gegen $i$. Laplace-Matrix $M = \mathrm{diag}(G\mathbf{1}) - G$.

**Massey.** $M r = p$ mit $p_i = \sum_j D_{ij}$. Das ist die Normalgleichung von $\min_r \sum_{\text{Partien}} (r_i - r_j - d_{ij})^2$. $M$ hat den Rang
$n - c$ ($c$ = Zahl der Teile des Plans); die Lösung mit kleinster Norm (Pseudoinverse) ist je Teil bis auf eine Konstante bestimmt.

**Colley.** $C r = b$ mit $C = 2I + M$ und $b_i = 1 + (w_i - l_i)/2$; $C$ ist positiv definit, die Summe der Wertungen ist $n/2$. Im vollständigen,
ausgeglichenen Plan mit $k$ Partien je Paar gilt $r_i = (1 + w_i + k/2)/(2 + k n)$: streng monoton in den Siegen.

**Vollständig und ausgeglichen.** $M = k(nI - J)$ bei $k$ Partien je Paar, also $r_i = p_i / (k n)$: dieselbe Ordnung wie die Siegtabelle, wenn jeder Sieg einen Punkt Vorsprung hat.

**Residuum.** $\rho = \sum_{i<j}(y_{ij} - (r_i - r_j))^2 / \sum_{i<j} y_{ij}^2$ mit der mittleren Punktedifferenz $y_{ij}$ je Partie über die gespielten Paare.

**Rangordnung.** Wertungen, die sich um weniger als $10^{-9}$ unterscheiden, gelten als gleich (Gleitkomma-Rauschen); Gleichstand entscheidet das Los.

**Literatur.**
- Massey, K. (1997): Statistical models applied to the rating of sports teams. Abschlussarbeit (Honors Project), Bluefield College. Eingesehen 2026-10-17 (Volltext,
  masseyratings.com/theory/massey97.pdf): Kleinste Quadrate auf den Punktedifferenzen, Normalgleichungen mit Diagonale = Partien und Nebendiagonale = minus
  Partien zwischen den beiden (S. 38), Ersatz einer Gleichung durch $\sum r = 0$ (S. 35), Beispiel 4.2 (S. 36–37).
- Colley, W. N. (2002): Colley's bias free college football ranking method: The Colley matrix explained. Online-Dokument, colleyrankings.com.
- Langville, A. N., Meyer, C. D. (2012): Who's #1? The Science of Rating and Ranking. Princeton University Press (Massey: Kapitel 2, Colley: Kapitel 3).
- Jiang, X., Lim, L.-H., Yao, Y., Ye, Y. (2011): Statistical ranking and combinatorial Hodge theory. *Mathematical Programming* 127, 203–244 (zerlegt die
  Vergleiche in einen Gradientenanteil und einen zyklischen Anteil).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
