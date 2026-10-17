# Massey und Colley: ein lineares System auf dem Spielplan – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-massey-colley-demo.streamlit.app/)**

Viertes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `siegtabelle-demo` und `rangaggregation-demo`: Die Siegtabelle zählt jeden Sieg gleich,
gleich gegen wen. Hier wird der Spielplan als **Graph** gelesen und die Wertung als Lösung eines **linearen Gleichungssystems** berechnet, das die Stärke der
Gegner über die Kanten weitergibt: **Massey** (Kleinste Quadrate auf der Punktedifferenz) und **Colley** (geglättete Siegquote mit Gegnerstärke). Das Vehikel
ist das neutrale, simulierte Turnier der Wurzel (Generator Version 1, wortgleich); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Was ändert ein lineares System gegenüber der Siegtabelle, und wo ändert es **nichts**? Ein System auf dem Spielgraphen mildert den Fehler durch ungleiche Gegner (behebt ihn aber nicht); wo der
Plan vollständig und ausgeglichen ist, gibt es ohne Kantersiege dieselbe Ordnung wie die Tabelle; Massey kippt bei Kantersiegen; und das Residuum sieht Zyklen nur, wenn sie
sich aufheben.

## Modell

- **Spielgraph:** Knoten sind die Teilnehmer, Kanten die gespielten Paare (mit Zahl der Partien). Laplace-Matrix **M** = Diagonale (Zahl der Partien) minus
  Zahl der Partien zwischen zwei Teilnehmern.
- **Massey:** M · r = p, p_i = Summe der Punktedifferenzen von i. Gelöst per Pseudoinverse (Minimum-Norm), damit auch zerfallende Pläne eine Lösung haben;
  der Originalweg (letzte Zeile durch Einsen ersetzen) wird für n ≤ 8 mit seiner Kondition mitgezeigt und scheitert dort, wo der Plan zerfällt.
- **Colley:** C · r = b mit C = 2I + M, b_i = 1 + (Siege − Niederlagen)/2; immer lösbar, Mittel der Wertungen 1/2.
- **Punktedifferenz:** Generator Version 1 liefert nur Sieg oder Niederlage. Jeder Sieg zählt 1 Punkt Vorsprung; ein einstellbarer Anteil q der Siege sind
  Kantersiege der Höhe h, **unabhängig von der Stärke**: ein Stresstest für Massey, keine Aussage über reale Spiele.
- **Rangordnung:** Wertungen, die sich um weniger als 10⁻⁹ unterscheiden, gelten als gleich; Gleichstand entscheidet das Los (Gleitkomma-Rauschen von
  10⁻¹⁶ entschiede sonst über Gleichstände, die in vollständigen Plänen exakt vorkommen).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 300 feste Turniere (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Teilnehmern
  (Experiment 2, Zeile „vollständig“: 8), unabhängig vom Regler.
- **Maße:** falsch geordnete Paare über die Staffeln, Kendall-τ gegen die Generator-Stärke, Zahl der Teile des Plans, Rang und Kondition der Matrix,
  Residuum des Massey-Modells.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit `siegtabelle-demo`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 300 feste Turniere, 12 Teilnehmer, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Zwei Staffeln ohne Kreuzpartie | Alle Verfahren ordnen die Staffeln wie einen Münzwurf (50 % der Paare falsch); der Plan zerfällt in 100 % der Turniere in zwei Teile, die Matrix ist singulär. | `test_without_crossing_games_all_methods_are_a_coin_flip_and_the_plan_falls_apart` |
| **Eine** Kreuzpartie je Starkem | Falsch geordnete Paare: Siegtabelle **37 %**, Massey **15,5 %**, Colley **24 %**. Mit 2 Kreuzpartien 27 / 12 / 16 %, mit 3 Kreuzpartien 18 / 9,5 / 11 %. | `test_one_crossing_game_per_strong_player_halves_the_error_for_massey` |
| Alle Kreuzpartien (vollständiger Plan) | Alle vier Verfahren gleich, 6 % falsch: kein Gewinn. | `test_with_all_crossing_games_the_plan_is_complete_and_all_methods_agree` |
| Zufällige Lücken (p = 0,5, 1 Partie) | τ gegen die Stärke: Siegtabelle 0,49, Siegquote 0,58, Massey 0,585, Colley 0,59. **Gegenüber den Siegen ein Gewinn (etwa +0,07 bis +0,10), gegenüber der Siegquote höchstens etwa +0,01.** Bei p = 0,33 zerfällt der Plan in rund 14 bis 16 %, bei p = 0,2 in rund 68 bis 71 % der Turniere (Stichprobe: 16 % und 71 %; die Rechnung für zufällige Lücken bei 12 Teilnehmern ergibt 14,0 % und 67,7 %). | `test_with_random_gaps_massey_and_colley_beat_wins_but_hardly_the_win_share` |
| Vollständig und ausgeglichen (8 Teilnehmer) | Die Ordnung von Massey und Colley ist in **allen** gleichstandsfreien Turnieren die der Siegtabelle (11 von 11, 86 von 86, 188 von 188 bei 2, 4, 10 Partien je Paar); τ gleich (0,67, 0,78, 0,88). | `test_complete_plan_no_gain_and_identical_order` |
| Kantersiege (4 Partien je Paar, Höhe 10, Stresstest) | Siegtabelle und Colley τ = 0,868 unabhängig von q. **Massey 0,868 bei q = 0, 0,724 bei q = 0,1**, 0,737 (0,2), 0,767 (0,3), 0,809 (0,5): unter der Tabelle. Das ist der ungünstigste Fall (Höhe 10, unabhängig von der Stärke); bei Höhe 2 liegt Massey bei q = 0,1 kaum darunter (0,865), bei Höhe 5 bei 0,804. | `test_massey_falls_below_the_win_table_with_blowouts_colley_does_not` |
| Residuum bei ausgewogenen Zyklen (20 Partien je Paar) | 0,10 (c = 0) → 0,33 (c = 0,2) → **0,66** (c = 0,4); τ gegen die Stärke bleibt über 0,95. | `test_the_residual_sees_balanced_cycles` |
| Residuum bei gekontertem Stärkstem | **0,10 bis 0,12**, obwohl τ gegen die Stärke auf 0,73 (c = 0,2) und 0,515 (c = 0,4) einbricht und der Stärkste im Mittel auf Platz 5,8 steht: Der Konter geht vermutlich im transitiven Anteil auf (Erklärung, nicht eigens gemessen). | `test_the_residual_does_not_see_a_countered_strongest` |

## Befunde und Korrekturen gegenüber dem Plan

- **Colley-Formel.** Der Plan nannte für den vollständigen, ausgeglichenen Plan `r_i = (1 + w_i)/(2 + g_i)`. Das ist die Laplace-Regel ohne Gegnerstärke und
  stimmt nicht. Nachgerechnet gilt `r_i = (1 + w_i + k/2)/(2 + k·n)` mit k Partien je Paar (Herleitung im 📐-Expander, geprüft im Orakel-Test); beide Formeln
  sind streng monoton in den Siegen.
- **Gleitkomma-Rauschen** hat die Vorab-Messreihe verzerrt: Bei x = 6 erschien Massey mit 8,2 % statt 6,3 % falsch, weil exakte Gleichstände durch Rauschen von
  10⁻¹⁶ getrennt wurden statt durch das Los. Die Demo rangiert mit Toleranz 10⁻⁹.
- **Zahlen der Vorab-Messreihe** wichen im Rauschen ab (z. B. Siegtabelle bei x = 1: 36,3 % statt 36,8 %): andere feste Turniere.
- **Regler „Partien je Paar“** geht bis 20 (der AppTest fand, dass ein Preset 20 setzte, der Regler aber bei 10 endete).

## Abgrenzung

- [Zentralität: PageRank](https://sebastianhanisch-centrality-demo.streamlit.app/) (Graphen und Netzwerke) berechnet eine Rangfolge als **Eigenvektor** eines
  Graphen; hier liegen **Kleinste Quadrate** auf demselben Graphen. Kontrast, kein Nachbau.
- [Spectral Clustering](https://sebastianhanisch-spectral-demo.streamlit.app/) (Clustering) benutzt die Laplace-Matrix eines Ähnlichkeitsgraphen für die
  Eigenzerlegung; hier steht dieselbe Matrix als Systemmatrix des Spielplans.
- [Siegtabelle](https://sebastianhanisch-siegtabelle-demo.streamlit.app/) (Wurzel) liefert die Spielpläne (zwei Staffeln, Lücken) und die Zyklen; dieses
  Stück baut direkt darauf auf.

## Ehrliche Grenzen

- **Der Gewinn hängt am Plan:** nur bei ungleichen Gegnern oder Lücken; im ausgeglichenen vollständigen Plan keiner.
- **Kantersiege sind ein Stresstest** (unabhängig von der Stärke); in echten Ligen steigt die Punktedifferenz vermutlich mit dem Stärkeabstand und belastet Massey dann weniger (nicht gemessen in dieser Demo; in einem einfachen Gegenmodell liegt Massey weiter unter der Tabelle, aber weniger weit als bei unabhängigen Kantersiegen).
- **Das Residuum sieht keinen gekonterten Stärksten**, weil der Konter im transitiven Anteil aufgeht.
- **Zerfallender Plan:** keine Reihenfolge über die Teile hinweg belegt; die Verfahren raten, die Matrix verrät es.
- **Alles simuliert**, ein Generator (Version 1), n ≤ 16; die Stärke ist eine Konstruktion, keine Messung.

## Quellen

- Massey, K. (1997): Statistical models applied to the rating of sports teams. Abschlussarbeit (Honors Project), Bluefield College. Eingesehen 2026-10-17
  (Volltext, masseyratings.com/theory/massey97.pdf): Kleinste Quadrate auf den Punktedifferenzen, Normalgleichungen mit Diagonale = Partien und
  Nebendiagonale = minus Partien zwischen den beiden (S. 38), Ersatz einer Gleichung durch Σ r = 0 (S. 35), Beispiel 4.2 (S. 36–37).
- Colley, W. N. (2002): Colley's bias free college football ranking method: The Colley matrix explained. Online-Dokument, colleyrankings.com.
- Langville, A. N., Meyer, C. D. (2012): Who's #1? The Science of Rating and Ranking. Princeton University Press (Massey: Kapitel 2, Colley: Kapitel 3,
  Zusammenhang beider: S. 24).
- Jiang, X., Lim, L.-H., Yao, Y., Ye, Y. (2011): Statistical ranking and combinatorial Hodge theory. *Mathematical Programming* 127, 203–244
  (DOI 10.1007/s10107-010-0419-x).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472.

## Tests

90 Tests, rund 100 Sekunden: Einheiten gegen die von Hand gerechneten Mini-Instanzen (`test_linear.py`, `test_margins.py`), Generator (`test_generator.py`,
wortgleich mit der Wurzel), **Orakel** (`test_oracle_massey_colley.py`: Massey gegen die Normalgleichungen aus der Partienliste und `pinv`, Colley gegen die
geschlossene Formel, Rang gegen die Komponenten per `networkx`, Positiv-Definitheit, Skalierung, Residuum bei exakt transitiven Daten), Presets, ausblendbare
Regler (`test_hidden_widget_state.py`), Analyse und Verdikte (`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`) und AppTest-Rauchtests.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `mcl_linear.py` | Laplace-Matrix, Massey, Colley, Rangordnung mit Toleranz, Residuum |
| `mcl_margins.py` | Punktedifferenz aus den Ausgängen, Kantersieg-Stresstest |
| `mcl_scenario.py` | Reglerwerte (`Settings`) und das Turnier mit Punktedifferenz |
| `mcl_evaluation.py` | Analyse, Experimente über die festen Turniere, Verdikte |
| `mcl_presets.py`, `mcl_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `mcl_visualization.py`, `mcl_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Wahrscheinlichkeiten statt Punktedifferenz** (Bradley-Terry), **Online-Fortschreibung** (Elo): eigene Stücke der Linie.
- **Echte Punktedifferenzen** mit Heimvorteil und Remis (Tore, Remis, Heimvorteil): eigenes Stück; hier nur der Kantersieg-Stresstest.
- **Volle Hodge-Zerlegung** (Gradient- und Zyklenanteil): gehört in „Nicht-Transitivität“; hier nur das Residuum.
- **Eigenvektor-Wertung** (Keener, PageRank): Kontrast, nicht nachgebaut.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
