# Cell property prediction

For clone/pull, environment setup and the first three TabPFN experiments, follow
the [cluster quickstart](docs/cluster-quickstart.md). The committed entry point is
`zsh cluster/start.sh` with `setup`, `pilot`, `submit` and `report` stages.

After completing those three benchmarks, follow the
[pending-experiment guide](docs/pending-experiments.md) to run the remaining
15 input/model combinations with `zsh cluster/pending.sh`. It preserves the
completed folds and submits a separate 300-task continuation array.

For the authors' approximately 1,000 highly variable genes and PCA128 of those
genes, follow the [HVG experiment guide](docs/hvg-experiments.md). The additional
`zsh cluster/hvg.sh` workflow evaluates both inputs with all three foundation
models on the same saved cell and mouse-held-out folds (120 additional tasks).

Reproduzierbarer Benchmark für die Vorhersage von 17 elektrophysiologischen
Zelleigenschaften aus Einzelzell-Genexpression. Ausgangspunkt ist Scala et al.,
*Phenotypic variation of transcriptomic cell types in mouse motor cortex*
([Paper](https://doi.org/10.1038/s41586-020-2907-3),
[Autorencode und Daten](https://github.com/berenslab/mini-atlas)).

Die Pipeline ist implementiert und mit synthetischen Daten geprüft.
Datenaufbereitung und Modellläufe erfolgen auf dem Cluster; wissenschaftliche
Ergebnisse werden dort erzeugt und nicht in Git gespeichert. Die lokalen Tests
laden keine Datensätze oder Modellgewichte herunter.

## Benchmark

Eine Zeile entspricht einer Zelle. Aus den vollständigen, unnormalisierten Exon-
und Intron-Gen-Counts werden nach den Kriterien des Autorencodes 1.219 Zellen
erwartet. Die Ausgangsmatrizen enthalten 1.329 Zellen und 42.466 Gene. Die
Count-Dateien sind bereits aligniert und quantifiziert; FASTQ-Verarbeitung ist
nicht Teil dieser Pipeline. Weitere Rohdaten sind unter
[GSE163764](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163764) verfügbar.

Die drei einfachen TabPFN-3.5-Benchmarks verwenden:

1. Breite Genexpression mit ausschließlich im Trainingsfold angepasster PCA
   auf maximal 128 Komponenten.
2. Das vollständige kuratierte Ionenkanal-Panel aus dem Autorenrepository.
3. Die festen 25 Gene aus der Hauptanalyse der Autoren.

Im vollständigen Vergleich kommen drei eingefrorene Zellrepräsentationen hinzu:
GenePT v2 (3.072 Dimensionen), UCE mit 33 Schichten (1.280) und Nicheformer (512).
Alle sechs Repräsentationen werden mit TabPFN 3.5, LimiX-2 und Kumo Tabular Large
verglichen. Zwei zusätzliche Expressionsrepräsentationen, `hvg` und
`hvg_pca128`, verwenden die Nullraten-/Expressionsregel der Autoren für etwa
1.000 Gene; die zweite reduziert diese Gene anschließend auf maximal 128
PCA-Komponenten. Genwahl, Skalierung und PCA werden in jedem Trainingsfold neu
angepasst. Beide werden ebenfalls mit allen drei Foundation-Modellen geprüft.
Ridge und ein Trainingsmittelwert bilden Kontrollmodelle. Eine
zusätzliche Sparse-RRR-Baseline wählt 1.000 Gene und ihre Hyperparameter innerhalb
der Trainingsfolds aus. Es findet kein Encoder-Finetuning statt.

| Protokoll | Zweck | Aufteilung |
|---|---|---|
| `cell_kfold` | Neue Zellen aus derselben Stichprobe | 10 Folds, Shuffle, Seed 42; jede Zelle genau einmal im Test |
| `mouse_group_kfold` | Übertragung auf andere Tiere | 10 GroupKFold-Folds nach Maus; kein Tier gleichzeitig in Training und Test |
| `author_historical` | Separater Vergleich mit dem historischen Autorencode | 10 × 121 Testzellen; neun Zellen immer im Training |

Bei korrigierter Zell-CV haben neun Folds 1.097 Trainings- und 122 Testzellen,
einer 1.098 und 121. Exakt 1.098/121 in allen zehn Folds lässt sich bei 1.219
Zellen nicht mit vollständiger Testabdeckung vereinbaren. Der historische
Integer-Divisions-Effekt stammt aus dem veröffentlichten Code; er wird hier
getrennt vom korrigierten Benchmark behandelt.

Die 17 Zielvariablen, Normalisierung, Panel-Gene und Auswertungsregeln stehen in
[docs/methodology.md](docs/methodology.md). Alle Modelle verwenden dieselben
gespeicherten äußeren Folds und alle verfügbaren Trainingszeilen. Native
Feature-Auswahl und Ensemble-Verfahren der Prädiktoren werden protokolliert.

## Cluster vorbereiten

Das Repository auf den Cluster übertragen. Die folgenden Befehle dort im
Projektverzeichnis ausführen. Sie sind eine Anleitung, keine bereits gestarteten
Jobs. Account und Partitionen sind in den Skripten auf `thes2301`, `c23mm` und
`c23g` eingestellt; bei anderer Clusterumgebung die `#SBATCH`-Zeilen anpassen.

```zsh
export CPP_ROOT="$PWD"
mkdir -p logs
# Vorher passende Python-Module der Clusterinstallation aktivieren.
zsh cluster/bootstrap.sh core
```

Standard ist Python 3.12; UCE benötigt wegen seiner Torch-Version Python 3.11.
Abweichende ausführbare Dateien lassen sich mit `CPP_PYTHON` bzw.
`CPP_PYTHON_UCE` angeben. Für einen vorgegebenen CUDA-Wheel-Index kann bei der
Installation `PIP_EXTRA_INDEX_URL` gesetzt werden. Die Modellumgebungen bleiben
getrennt. `logs/environment-*.lock.txt` dokumentiert die tatsächlich installierten
Versionen.

Auf einem Clusterknoten mit erlaubtem Internetzugang einmalig herunterladen:

```zsh
CPP_ENV=core zsh cluster/run.sh prefetch --data \
  --models tabpfn limix kumo genept uce nicheformer
for model in tabpfn limix kumo uce nicheformer; do
  zsh cluster/bootstrap.sh "$model"
done
```

Vor einem gegebenenfalls geschützten Hugging-Face-Download die jeweiligen
Modellbedingungen akzeptieren und `HF_TOKEN` über die eigene Clusterumgebung
bereitstellen. Die Pipeline verwendet lokale Gewichte für die Inferenz. Sie
zeichnet Download-Hashes und konkrete HF-/Git-Revisionen unter `data/raw/` und
`cache/model_manifest.json` auf. Folgeläufe behalten diese Revisionen bei.
GenePT- und Nicheformer-Mappings werden vorab eingefroren. LimiX-2 ist mit den
jeweiligen nichtkommerziellen Modellbedingungen zu verwenden; die Links zu den
Original-Lizenzen stehen unten.

Danach Datenaufbereitung und Splits als CPU-Jobs ausführen; jeweils erst nach
erfolgreichem Abschluss des vorherigen Schritts fortfahren:

```zsh
sbatch --export=ALL cluster/stage.sbatch prepare
# Nach Abschluss:
sbatch --export=ALL cluster/stage.sbatch splits --historical
```

`prepare` prüft Ausgangsdimensionen und Kohortengröße, richtet alle Tabellen nach
Zell-IDs aus und speichert vollständige Counts, Autorennormalisierung, Targets,
Metadaten und ein Filter-Audit. Für Encoder enthält `counts.h5ad` keine
elektrophysiologischen Labels, Maus- oder Zelltypinformationen.

## Embeddings und Pilot

Für die drei einfachen Benchmarks sind noch keine Embeddings erforderlich.
Für die vollständige Matrix diese drei Schritte vorher abschließen:

```zsh
CPP_ENV=core sbatch --export=ALL cluster/stage.sbatch \
  embed --encoder genept --device cpu
sbatch --export=ALL cluster/encoder.sbatch uce
sbatch --export=ALL cluster/encoder.sbatch nicheformer
```

Die GPU-Encoder-Jobs fordern jeweils eine GPU, 16 CPUs, 64 GB RAM und vier Stunden
an. Die native Encoder-Vorverarbeitung wird verwendet; eine veränderte
Embedding-Dimension löst einen Fehler aus. `embeddings/*.audit.json` dokumentiert
Mapping, Abdeckung, Modellressourcen und Ausgabe-Hash.

Piloten messen zwei Targets im ersten Zellfold. Jeder Prädiktor läuft in seiner
eigenen Umgebung. Die Pilot-Jobs **nacheinander** starten, da sie dieselbe
Timing-Datei ergänzen. Beispiel für die vollständige Matrix:

```zsh
p1=$(sbatch --export=ALL --parsable cluster/pilot.sbatch tabpfn)
p2=$(sbatch --export=ALL --parsable --dependency=afterok:"$p1" cluster/pilot.sbatch limix)
sbatch --export=ALL --dependency=afterok:"$p2" cluster/pilot.sbatch kumo
```

Für den ersten einfachen Benchmark genügt stattdessen:

```zsh
sbatch --export=ALL cluster/pilot.sbatch tabpfn \
  --representations pca128 ion_channels authors25
```

Wenn alle Embeddings schon vor der ersten Pilotmessung erzeugt wurden, kann
dieser TabPFN-Pilot später um die übrigen Repräsentationen und Modelle ergänzt
werden. Falls neue Eingabedateien, Modellkonfiguration oder Quellcode hinzukommen,
eine neue Pilotmessung in einem frischen Ergebnisverzeichnis vornehmen.
Gespeicherte Target-Laufzeiten werden bei einem Neustart berücksichtigt.

## Jobmatrix erstellen und ausführen

Nach den Piloten das **aktuell verbleibende** GPU-Stundenbudget in
`configs/benchmark.yaml`, `cluster.remaining_gpu_hours`, eintragen. Die gesamte
Projektzuteilung ist kein Nachweis des Restbudgets. Diese eine Änderung
invalidiert den Piloten nicht.

```zsh
zsh cluster/run.sh jobs --stage simple --output jobs/simple.tsv
zsh cluster/run.sh jobs --stage full --output jobs/full.tsv
zsh cluster/run.sh jobs --stage baselines --output jobs/baselines.tsv
zsh cluster/run.sh jobs --stage rrr --output jobs/rrr.tsv
zsh cluster/run.sh jobs --stage hvg --output jobs/hvg.tsv
zsh cluster/run.sh budget --jobs jobs/full.tsv
```

`simple` erzeugt 60 Jobs bzw. 1.020 Target/Fold-Kombinationen. `full` erzeugt
360 Jobs bzw. 6.120 Kombinationen und enthält die einfachen Benchmarks bereits.
Für die Fortsetzung eines abgeschlossenen `simple`-Laufs den
[Pending-Workflow](docs/pending-experiments.md) verwenden: `pending` erzeugt nur
die übrigen 300 Jobs bzw. 5.100 Fits. Seine separaten Piloten und Budgetprüfung
verhindern, dass alte Pilotmessungen nach neuen Downloads weiterverwendet werden.
`full` ist die vollständige 360-Job-Matrix, insbesondere für frische Läufe. Die Budgetschätzung ist konservativ
(Pilotzeit × Targetanzahl × Faktor 2), ersetzt aber keine Messung der ganzen
Laufzeit und des GPU-Speichers.

`hvg` ergänzt die ursprüngliche Matrix um sechs weitere Eingabe/Modell-Paare,
120 Jobs und 2.040 Target/Fold-Fits. Es ist ein eigener Zusatz zum unveränderten
`full`- bzw. `pending`-Manifest. Die ursprünglichen sechs Eingaben plus beide
HVG-Eingaben ergeben insgesamt acht Eingaben × drei Modelle × zwei Protokolle ×
zehn Folds, also 480 Jobs. Für den zusätzlichen HVG-Lauf werden separate Piloten
und ein eigenes Budget verwendet; die Anleitung steht im
[HVG experiment guide](docs/hvg-experiments.md).

```zsh
# Standard: nur Kommando anzeigen.
zsh cluster/submit.sh jobs/simple.tsv
zsh cluster/submit.sh jobs/baselines.tsv --cpu

# Tatsächlich einreichen, wenn Pilot und Budget passen:
zsh cluster/submit.sh jobs/simple.tsv --execute
# Nach dem Einstieg; vollständige Matrix, fertige Targets werden übersprungen:
zsh cluster/submit.sh jobs/full.tsv --execute
zsh cluster/submit.sh jobs/baselines.tsv --cpu --execute
zsh cluster/submit.sh jobs/rrr.tsv --cpu --execute
```

GPU-Arrayjobs laufen standardmäßig mit höchstens vier gleichzeitigen Jobs, je
einer GPU, acht CPUs, 32 GB RAM und sechs Stunden. Ein Job umfasst einen äußeren
Fold und eine Repräsentation/Modell-Kombination, mit 17 separaten Zielmodellen.
Sparse-RRR modelliert die Targets gemeinsam; seine Laufzahl ist deshalb nicht
mit 17 unabhängigen Fits gleichzusetzen. Mit `CPP_ARRAY_CONCURRENCY` lässt sich
die Array-Parallelität ändern. Beim Fortsetzen dieselbe Konfiguration verwenden;
abweichende Daten-, Modell- oder Code-Hashes werden zurückgewiesen.

Die Clusterparameter in YAML dienen der Dokumentation und Budgetierung.
Ressourcenanforderungen stehen in den `#SBATCH`-Zeilen und werden nicht automatisch
aus YAML übernommen. Encoder und Prädiktoren können je nach GPU zusätzliche
Ressourcen benötigen; das ist im Pilot zu prüfen.

## Historischer Vergleich und Bericht

```zsh
# Separater CPU-Job mit historischen Splits und historischer Gesamtmetrik:
sbatch --export=ALL cluster/stage.sbatch historical --backend sklearn_compatible
# Nach Abschluss der gewünschten korrigierten Benchmark-Jobs:
sbatch --export=ALL cluster/stage.sbatch report
```

`sklearn_compatible` rekonstruiert Sparse-RRR mit einem kompatiblen Optimierer.
Es ist keine bitgenaue Reproduktion des ursprünglichen glmnet-Solvers. Für den
optionalen `--backend original` müssen die historischen glmnet-Abhängigkeiten
zusätzlich in einer kompatiblen Umgebung installiert sein; der Quellcode-Hash
wird geprüft und es gibt keinen automatischen Solverwechsel. Die im Paper
berichteten Zahlen dürfen erst nach einem echten Lauf als reproduziert gelten.

Berichte erscheinen unter `results/reports/`: Vorhersagen pro Zelle und Target,
R²/MAE/Pearson je Target, mittleres R² über die Targets, mausweise
Bootstrap-Konfidenzintervalle, gepaarte Modellunterschiede und Ergebnisse innerhalb
von RNA-Familien. Unvollständige OOF-Methoden werden markiert und nicht gerankt.
Historische Ergebnisse und Pilotdaten werden getrennt gespeichert.

## Prüfung und Aufbau

```zsh
# Auf dem Cluster in der Core-Umgebung; nur kleine synthetische Fixtures:
.venvs/core/bin/python -m pytest -q
for script in cluster/*.sh cluster/*.sbatch; do zsh -n "$script"; done
```

`src/cell_property_prediction/` enthält Daten-, Split-, Embedding-, Modell-,
Evaluations- und Reportingmodule. `cluster/` enthält Bootstrap- und Slurm-Skripte,
`envs/` die modellabhängigen Pakete und `configs/benchmark.yaml` die Konfiguration.
Große Daten, Gewichte, Umgebungen und Ergebnisse sind in Git ausgeschlossen.

Primärquellen: [TabPFN 3.5](https://huggingface.co/Prior-Labs/tabpfn_3_5),
[LimiX-Code](https://github.com/limix-ldm-ai/LimiX) und
[Gewichte/Lizenz](https://huggingface.co/stable-ai/LimiX-2),
[Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular),
[GenePT](https://github.com/yiqunchen/GenePT),
[UCE](https://github.com/snap-stanford/UCE),
[Nicheformer](https://github.com/theislab/nicheformer).
