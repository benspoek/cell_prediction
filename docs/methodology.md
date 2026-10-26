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
zusätzlich als eigene Layer erhalten. Für die Expressionsbenchmarks wird
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

Die zusätzlichen Eingaben `hvg` und `hvg_pca128` verwenden die Genwahl aus dem
[fixierten Autoren-Notebook](https://github.com/berenslab/mini-atlas/blob/3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9/code/reduced-rank-regression.ipynb).
Die Funktion `rnaseqTools.geneSelection` betrachtet den Anteil der Zellen mit
höchstens 32 Exon-Counts und die mittlere log2-Expression der übrigen Zellen.
Sie wählt Gene oberhalb einer exponentiellen Nullraten-Grenze aus. Der
horizontale Offset dieser Grenze wird per binärer Suche auf etwa 1.000 Gene
angepasst (`n=1000`, `threshold=32`, `atleast=10`, `decay=1.5`,
`yoffset=0.02`, anfänglicher `xoffset=5`). Bei Bindungen muss die Auswahl nicht
exakt 1.000 Gene ergeben; die tatsächliche Anzahl wird protokolliert. Dies ist
die Autorenregel und keine einfache Rangfolge nach Varianz.

Im ursprünglichen Notebook erfolgt diese Genwahl auf der gesamten Kohorte vor
der Kreuzvalidierung. Hier wird sie für jeden äußeren Trainingsfold auf dessen
rohen **Exon-Counts** neu angepasst, ohne die Testzellen einzubeziehen. Die
ausgewählten Symbole, Parameter und Eingabe-Hashes stehen im Fold-Manifest.
Die Genlisten können zwischen Folds variieren, sind innerhalb eines Folds aber
für alle drei Prädiktoren und für beide HVG-Eingaben gleich. Konstante Gene im
Training werden entfernt. Die Feature-Werte bleiben die oben definierte
log2-normalisierte Expression aus Exon und Intron, mit Bibliotheksgröße über
alle Gene; sie werden anhand des Trainings skaliert.

`hvg` reicht diese etwa 1.000 Gen-Features direkt an den jeweiligen Prädiktor
weiter. `hvg_pca128` passt zusätzlich eine PCA nur an diese skalierten
Trainingsdaten an und transformiert die Testzellen mit derselben PCA. Die
Komponentenzahl ist auf `min(128, n_train - 1, n_selected_genes)` begrenzt.
Die bestehende Eingabe `pca128` verwendet dagegen alle ausreichend detektierten,
im Training variierenden Gene, ohne diese HVG-Auswahl. Die HVG-Experimente nutzen
dieselben gespeicherten `cell_kfold`- und `mouse_group_kfold`-Splits wie die
übrigen Hauptbenchmarks; sie verwenden nicht den separaten historischen Split.

Die zusätzlichen Encoder-Eingaben `hvg_genept`, `hvg_uce` und
`hvg_nicheformer` verwenden für jeden Fold dieselbe HVG-Liste wie die direkten
HVG-Eingaben. Die Encoder erhalten ausschließlich die rohen Exon-plus-Intron-
Counts dieser Gene und behalten ihre native Vorverarbeitung bei. GenePT
normalisiert und mittelt damit auf dem ausgewählten Panel; UCE verwendet seine
native countbasierte Stichprobe; Nicheformer seine festen Referenzmittel und
Genrangfolge. Die Encoder bleiben eingefroren und behalten ihre nativen
Dimensionen 3.072, 1.280 bzw. 512.

Trainingszellen stehen beim Kodieren vor den Testzellen, um insbesondere bei UCE
keine von vorherigen Testzellen abhängige Zufallsfolge im Training zu erzeugen.
Gespeichert werden Embeddings in kanonischer Zell-ID-Reihenfolge. Die zwanzig
HVG-Panels und sechzig Encoder-Ausgaben werden über alle drei Prädiktoren
wiederverwendet. Jeder Fold wird gegen sein eigenes Auswahl- und Split-Audit
geprüft; die gemeinsame Experimentdefinition muss über Folds übereinstimmen.
Dies ist ein Vergleich nativer HVG-Panel-Pipelines, keine Ablation mit über alle
Genpanels festgehaltener Normalisierung.

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

Jeder moderne Bericht berechnet zusätzlich `authors_joint_r2` nach der
Scoring-Definition der Autoren. Beobachtungen und Vorhersagen werden für AP
coefficient of variation, ISI coefficient of variation und ISI adaptation index
natürlich logarithmiert. Für jedes transformierte Target wird die
Standardabweichung der beobachteten Werte über die gesamte Kohorte mit
`ddof=0` berechnet. Pro gespeichertem Testfold gilt:

```text
SSE_fold = sum_over_test_cells_and_targets(((transformed_true - transformed_pred) / cohort_std)^2)
SST_fold = sum_over_test_cells_and_targets(((transformed_true - mean_transformed_training_target) / cohort_std)^2)
authors_joint_r2 = mean_over_all_expected_folds(1 - SSE_fold / SST_fold)
```

Alle Folds werden gleich gewichtet, auch bei unterschiedlichen Testgrößen.
Die Mittelwerte im Nenner stammen aus den tatsächlich gespeicherten
Trainings-IDs des jeweiligen Folds. Die globale Standardabweichung wird nur
zur Bewertung verwendet und nicht an die Prädiktoren weitergegeben.

Dieser zusätzliche Score wendet die Autorenmetrik auf die vorhandenen
Zell- bzw. Mouse-held-out-Splits an; er ist keine exakte Reproduktion der
historischen Evaluation. Der historische Integer-Divisions-Effekt wird dabei
nicht eingeführt. Primär bleiben die 17 R²-Werte in Originaleinheiten und ihr
Mittelwert; Rangfolge, Bootstrap-Intervalle und gepaarte Vergleiche bleiben auf
dieser primären Skala.

`summary.csv` enthält den zusätzlichen Score sowie Status, Grund und Anzahl
verwendbarer Folds. `authors_joint_r2_folds.csv` speichert je Methode/Fold den
Score, SSE, SST und den Validitätsstatus. Nur bei vollständigen Vorhersagen und
allen gültigen Folds wird ein Gesamtwert ausgegeben. Nichtpositive Werte bei
einer Log-Transformation, fehlende oder nicht endliche Werte, konstante
Kohorten-Targets und ein nichtpositiver Fold-Nenner ergeben einen fehlenden
Joint-Score mit Begründung. Es werden keine Werte beschnitten und keine
Zellen, Targets oder Folds stillschweigend ausgeschlossen; die vorhandenen
Originalskalen-Metriken bleiben verfügbar.

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
