# Auswertungsprotokoll

## Daten und Zielvariablen

Die Dateien aus `berenslab/mini-atlas` werden am Commit
`3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9` fixiert. Die Hauptkohorte verwendet
Raumtemperatur-Aufnahmen; zusätzliche `phys_temp`-Aufnahmen werden nicht gepoolt.
Aufnahme-IDs werden durchgängig als Zell-IDs beibehalten.

Kohortenfilter: zugewiesener `RNA type`, keine `Exclusion reasons` und endliche
Werte in allen 17 Targets. Erwartet werden 1.219 von 1.329 Zellen. Die
Implementierung bricht bei abweichenden Dimensionen ab, damit eine geänderte
Quelldatei nicht unbemerkt einen anderen Benchmark erzeugt.

Die Targets sind in kanonischer Reihenfolge:

1. AP amplitude (mV)
2. AP amplitude adaptation index
3. AP coefficient of variation
4. AP threshold (mV)
5. AP width (ms)
6. Afterhyperpolarization (mV)
7. ISI adaptation index
8. ISI coefficient of variation
9. Input resistance (MOhm)
10. Latency (ms)
11. Max number of APs
12. Membrane time constant (ms)
13. Rebound (mV)
14. Resting membrane potential (mV)
15. Rheobase (pA)
16. Sag ratio
17. Upstroke-to-downstroke ratio

Wie im Autorencode werden AP coefficient of variation, ISI coefficient of
variation und ISI adaptation index vor der Modellierung natürlich logarithmiert.
Die anschließende Targetstandardisierung verwendet nur das jeweilige Training.
Vor der Hauptauswertung werden die Transformationen rückgängig gemacht.

## Repräsentationen

Die gespeicherten Counts bestehen aus Exon plus Intron. Beide Matrizen bleiben
zusätzlich als eigene Layer erhalten. Für die drei Expressionsbenchmarks wird
die Autorentransformation verwendet:

```text
length_corrected = exon / exon_length + intron / (intron_length + 0.001)
expression = log2(1 + 1e6 * length_corrected / sum_over_all_genes(length_corrected))
```

Das ist eine Transformation pro Zelle, ohne Anpassung an andere Testzellen.
Genfilter, Genskalierung und PCA werden ausschließlich im jeweiligen Training
angepasst. Ridge stimmt seine Regularisierung in drei inneren Folds ab; auch dort
werden die angepassten Expressionsschritte neu geschätzt. Bei Mouse-held-out CV
bleiben die inneren Folds ebenfalls nach Maus getrennt.

Das feste 25-Gen-Panel lautet:

```text
Arpp21 Bcan Bdnf Cacna1e Cacna2d1 Cacna2d3 Coro6 Erbb4 Gabrd Gad1
Galnt14 Htr3a Kcnip2 Lamp5 Mybpc1 Nxph1 Pde1a Ptk2b Pvalb Slc6a1
Sst Synpr Tac1 Tafa1 Vip
```

Dieses Panel wurde ursprünglich mit Kenntnis der Gesamtkohorte ausgewählt. Sein
Ergebnis ist ein Vergleich auf einer publizierten festen Auswahl, keine
unverzerrte Bewertung einer innerhalb der neuen Trainingsfolds ausgewählten
25-Gen-Signatur. In einem Trainingsfold konstante Gene werden entfernt.

GenePT verwendet log1p(CPM10k), mit Bibliotheksgröße aus allen Genen, feste
eindeutige Maus-Mensch-Orthologe und die originalen 3.072-dimensionalen
Genvektoren. Die gewichtete Summe wird durch die Anzahl berücksichtigter Gene
geteilt (`gene_count_mean`). UCE und Nicheformer verwenden ihre native
Vorverarbeitung mit vorab gespeicherten Modellressourcen. Nicheformer bevorzugt
native Maus-IDs im Modellvokabular und verwendet eindeutige humane Orthologe als
explizit dokumentierten Ersatz. Nicht eindeutige Mappings werden verworfen.

Alle Encoder sind eingefroren. Welcher Encoder für elektrophysiologische Targets
am besten funktioniert, ist die empirische Frage des Benchmarks. Eine mögliche
Überlappung öffentlicher Expressionsdaten mit den Vortrainingskorpora ist noch
nicht abschließend geprüft; sie ist von einer Label-Leakage im neuen CV zu
unterscheiden und muss bei Ergebnisinterpretation berücksichtigt werden.

## Prädiktoren und Kontrollen

TabPFN 3.5 verwendet ausdrücklich das konfigurierte lokale 3.5-Checkpoint und
acht Ensemblemitglieder. LimiX-2 verwendet das 400M-Modell, die native V2-Rezeptur
ohne Retrieval, acht Pipelines und kein Adaptive-SVD. Kumo verwendet das lokale
Large-Modell, acht Ensemblemitglieder und das native q500 als Punktvorhersage.
Die native Kumo-Rezeptur kann pro Ensemblemitglied 500 Feature-Spalten auswählen;
das begrenzt nicht die Dimension des gespeicherten Embeddings. Alle
Kontextzeilen werden beibehalten. Native Transformationen und effektive
Dimensionen werden in den Ergebnis-Audits gespeichert.

Die faire Sparse-RRR-Kontrolle wählt 1.000 Gene mit der Nullraten-/Expressionsregel
der Autoren neu im Training aus. Rang ist 5, die Lambda-Auswahl aus
`[0.1, 0.3, 0.45, 0.5, 1.0]` erfolgt in drei inneren Folds. Sie modelliert die
17 Targets gemeinsam und verwendet einen dokumentierten sklearn-kompatiblen
Solver. Die Konvergenz wird protokolliert.

## Metriken und historischer Vergleich

Die primäre Metrik ist das ungewichtete Mittel der 17 einzelnen R²-Werte auf den
zusammengeführten Out-of-fold-Vorhersagen in Originaleinheiten. Zusätzlich werden
MAE und Pearson-Korrelation je Target berichtet. Mausweises Bootstrap mit 1.000
Wiederholungen liefert Intervalle und gepaarte Unterschiede auf denselben
Vorhersagen. Diese Intervalle erfassen die Stichprobenunsicherheit der gespeicherten
OOF-Vorhersagen, nicht die zusätzliche Variation einer erneuten Modellanpassung
mit anderen Fold-Zuordnungen. Berichte innerhalb von RNA-Familien verlangen
mindestens 20 Zellen.

Der historische Modus reproduziert die globale Gen-/Targetskalierung, die
ursprüngliche Genwahl und den Seed-42-Permutationssplit mit ganzzahliger
Fold-Länge 121. Dadurch sind 1.210 Zellen im Test und neun immer im Training.
Seine Gesamtmetrik ist pro Fold `1 - SSE / sum(Y_test_centered_by_train_mean^2)`,
anschließend über Folds gemittelt. Sie ist nicht mit dem Mittel der 17 modernen
OOF-R²-Werte gleichzusetzen. Für breite Gene gilt Lambda 0.45, für das
Ionenkanal-Panel 0.303; jeweils Rang 5 und Alpha 1. Unvollständige historische
Durchläufe erhalten keinen Gesamtwert über weniger Folds.

Im historischen Modus sind Solver-Rekonstruktion und Originalbackend explizit
getrennt. Die publizierten Werte (ungefähr 0.38 bzw. 0.33) sind Referenzen, keine
in diesem Repository bereits erzielten Resultate.
