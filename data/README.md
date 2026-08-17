# Data for Spiral Inquiry Geometry (SIG)

This directory contains the third-party benchmark data required to reproduce the empirical experiments reported for **Spiral Inquiry Geometry (SIG)**.

The benchmark datasets themselves are **not distributed in this public SIG repository**. Users should obtain them from their official upstream sources and place the required files in the directory structure documented below.

The SIG software uses two benchmark families:

1. **QuestBench**
2. **AskBench**

The expected raw-data structure is:

```text
data/
├── README.md
├── questbench/
│   ├── GSM-Q.csv
│   ├── GSME-Q.csv
│   ├── Logic-Q.csv
│   └── Planning-Q.csv
└── askbench/
    ├── train/
    │   ├── mind.jsonl
    │   └── overconfidence.jsonl
    └── eval/
        ├── ask_mind.jsonl
        ├── ask_mind_bbhde.jsonl
        ├── ask_mind_gpqade.jsonl
        ├── ask_mind_math500de.jsonl
        ├── ask_mind_medqade.jsonl
        └── ask_overconfidence.jsonl
```

Preserve these file and directory names unless you also update the corresponding paths in the SIG source code.

---

## 1. QuestBench

**QuestBench: Can LLMs Ask the Right Question to Acquire Information in Reasoning Tasks?**

QuestBench is used in SIG to evaluate sufficient-question identification across mathematical reasoning, logic, and planning tasks.

### Official sources

Official Google DeepMind repository:

https://github.com/google-deepmind/questbench

Dataset page:

https://huggingface.co/datasets/belindazli/QuestBench

Users should consult the official repository and dataset documentation for the current dataset description, citation information, licensing terms, and usage requirements.

### Required QuestBench files

SIG uses these four CSV files:

```text
GSM-Q.csv
GSME-Q.csv
Logic-Q.csv
Planning-Q.csv
```

Place them in:

```text
data/questbench/
```

The final directory must be:

```text
data/
└── questbench/
    ├── GSM-Q.csv
    ├── GSME-Q.csv
    ├── Logic-Q.csv
    └── Planning-Q.csv
```

### GSM-Q

Required location:

```text
data/questbench/GSM-Q.csv
```

Source:

https://huggingface.co/datasets/belindazli/QuestBench

This file is used for the GSM-Q sufficient-question experiments.

### GSME-Q

Required location:

```text
data/questbench/GSME-Q.csv
```

Source:

https://huggingface.co/datasets/belindazli/QuestBench

This file is used for the GSME-Q sufficient-question experiments.

### Logic-Q

Required location:

```text
data/questbench/Logic-Q.csv
```

Source:

https://huggingface.co/datasets/belindazli/QuestBench

This file is used for the Logic-Q experiments.

### Planning-Q

Required location:

```text
data/questbench/Planning-Q.csv
```

Source:

https://huggingface.co/datasets/belindazli/QuestBench

This file is used for the Planning-Q experiments.

### How SIG reads QuestBench

The SIG QuestBench analysis expects:

```text
data/questbench/GSM-Q.csv
data/questbench/GSME-Q.csv
data/questbench/Logic-Q.csv
data/questbench/Planning-Q.csv
```

The QuestBench processing code is located under:

```text
src/sig/
```

and the complete workflow is launched through:

```text
run_all.py
```

Do not rename the benchmark files without updating the corresponding paths in the source code.

---

## 2. AskBench

AskBench provides the interactive and multi-turn data used to evaluate trajectory-level predictions of Spiral Inquiry Geometry.

SIG uses two parts of AskBench:

1. **training / trajectory data**
2. **evaluation data**

The training data are used to reconstruct multi-turn inquiry trajectories. The evaluation data are used for the AskBench evaluation analyses.

### Official sources

Official AskBench GitHub repository:

https://github.com/jialeuuz/askbench

AskBench evaluation dataset:

https://huggingface.co/datasets/jialeuuz/askbench_bench

AskBench training / trajectory dataset:

https://huggingface.co/datasets/jialeuuz/askbench_train

Users should consult these official sources for the latest documentation, citation requirements, licensing information, and dataset updates.

---

## 3. AskBench Training / Multi-turn Trajectory Data

The SIG trajectory and representation-robustness experiments require:

```text
mind.jsonl
overconfidence.jsonl
```

Download them from:

https://huggingface.co/datasets/jialeuuz/askbench_train

Place them in:

```text
data/askbench/train/
```

The directory should therefore contain:

```text
data/
└── askbench/
    └── train/
        ├── mind.jsonl
        └── overconfidence.jsonl
```

### mind.jsonl

Required location:

```text
data/askbench/train/mind.jsonl
```

Official source:

https://huggingface.co/datasets/jialeuuz/askbench_train

This file supplies the AskMind records used for multi-turn inquiry-trajectory analysis.

SIG uses records containing usable:

```text
conversation_history
```

to reconstruct inquiry trajectories.

### overconfidence.jsonl

Required location:

```text
data/askbench/train/overconfidence.jsonl
```

Official source:

https://huggingface.co/datasets/jialeuuz/askbench_train

This file supplies the AskOverconfidence records used in trajectory-level analysis.

As with AskMind, only records containing usable conversation histories enter the corresponding trajectory analysis.

### Important trajectory-count note

The raw AskBench training files include records with and without usable `conversation_history`.

SIG does **not** treat every JSONL row as a multi-turn trajectory.

For the dataset version used in the reported SIG experiments, the two files contained:

```text
13,094 total JSONL rows
```

of which:

```text
6,547
```

contained usable multi-turn trajectories:

```text
3,226 AskMind trajectories
3,321 AskOverconfidence trajectories
```

The software determines usable trajectories from the data rather than assuming every JSONL row represents an observed trajectory.

---

## 4. AskBench Evaluation Data

SIG also uses the official AskBench evaluation data.

Download the evaluation dataset from:

https://huggingface.co/datasets/jialeuuz/askbench_bench

The software expects the evaluation files in:

```text
data/askbench/eval/
```

For the dataset version used by the SIG analyses, the expected files are:

```text
ask_mind.jsonl
ask_mind_bbhde.jsonl
ask_mind_gpqade.jsonl
ask_mind_math500de.jsonl
ask_mind_medqade.jsonl
ask_overconfidence.jsonl
```

The complete evaluation directory should therefore be:

```text
data/
└── askbench/
    └── eval/
        ├── ask_mind.jsonl
        ├── ask_mind_bbhde.jsonl
        ├── ask_mind_gpqade.jsonl
        ├── ask_mind_math500de.jsonl
        ├── ask_mind_medqade.jsonl
        └── ask_overconfidence.jsonl
```

### ask_mind.jsonl

Required location:

```text
data/askbench/eval/ask_mind.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used in the main AskMind evaluation analysis.

### ask_mind_bbhde.jsonl

Required location:

```text
data/askbench/eval/ask_mind_bbhde.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used as an AskMind evaluation subset.

### ask_mind_gpqade.jsonl

Required location:

```text
data/askbench/eval/ask_mind_gpqade.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used as an AskMind evaluation subset.

### ask_mind_math500de.jsonl

Required location:

```text
data/askbench/eval/ask_mind_math500de.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used as an AskMind evaluation subset.

### ask_mind_medqade.jsonl

Required location:

```text
data/askbench/eval/ask_mind_medqade.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used as an AskMind evaluation subset.

### ask_overconfidence.jsonl

Required location:

```text
data/askbench/eval/ask_overconfidence.jsonl
```

Source:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Used for the AskOverconfidence evaluation analysis.

---

## 5. Complete Required Raw-Data Checklist

Before attempting a complete from-raw reproduction, verify that the following files exist.

### QuestBench

```text
[ ] data/questbench/GSM-Q.csv
[ ] data/questbench/GSME-Q.csv
[ ] data/questbench/Logic-Q.csv
[ ] data/questbench/Planning-Q.csv
```

### AskBench training trajectories

```text
[ ] data/askbench/train/mind.jsonl
[ ] data/askbench/train/overconfidence.jsonl
```

### AskBench evaluation data

```text
[ ] data/askbench/eval/ask_mind.jsonl
[ ] data/askbench/eval/ask_mind_bbhde.jsonl
[ ] data/askbench/eval/ask_mind_gpqade.jsonl
[ ] data/askbench/eval/ask_mind_math500de.jsonl
[ ] data/askbench/eval/ask_mind_medqade.jsonl
[ ] data/askbench/eval/ask_overconfidence.jsonl
```

A complete raw-data installation therefore contains:

```text
4 QuestBench CSV files
+
2 AskBench training JSONL files
+
6 AskBench evaluation JSONL files
=
12 raw data files
```

---

## 6. Final Expected Directory Tree

After downloading the datasets, the complete data directory should look like:

```text
data/
├── README.md
├── questbench/
│   ├── GSM-Q.csv
│   ├── GSME-Q.csv
│   ├── Logic-Q.csv
│   └── Planning-Q.csv
└── askbench/
    ├── train/
    │   ├── mind.jsonl
    │   └── overconfidence.jsonl
    └── eval/
        ├── ask_mind.jsonl
        ├── ask_mind_bbhde.jsonl
        ├── ask_mind_gpqade.jsonl
        ├── ask_mind_math500de.jsonl
        ├── ask_mind_medqade.jsonl
        └── ask_overconfidence.jsonl
```

---

## 7. Installing the Software Dependencies

From the repository root, install the required Python dependencies:

```bash
python -m pip install -r requirements.txt
```

For an editable package installation, where supported:

```bash
python -m pip install -e .
```

See the root-level `README.md` and `REPRODUCIBILITY.md` for the supported reproduction workflow.

---

## 8. Running SIG After Installing the Data

### Windows

Run:

```bat
run_all.bat
```

### Linux / macOS

Run:

```bash
./run_all.sh
```

Alternatively:

```bash
python run_all.py
```

The main workflow reads the benchmark data from:

```text
data/questbench/
data/askbench/train/
data/askbench/eval/
```

and writes generated analysis outputs under:

```text
results/
```

---

## 9. Representation-Robustness Experiment

The representation-robustness analysis uses the same AskBench multi-turn trajectories rather than a separate dataset.

It primarily reads:

```text
data/askbench/train/mind.jsonl
data/askbench/train/overconfidence.jsonl
```

Run:

```bash
python run_representation_robustness.py
```

The robustness analysis compares multiple representations/metrics used in the SIG study and generates corresponding tables and figures under:

```text
results/
```

No additional benchmark dataset is required for this experiment.

---

## 10. Generated Files Are Not Raw Datasets

Files under:

```text
results/
```

are generated or derived research outputs.

Depending on the workflow/version, these may include derived feature tables, statistical summaries, robustness results, and publication figures.

Generated outputs may include files such as:

```text
results/candidate_features.csv
results/askbench_trajectory_features.csv
results/askbench_eval_geometry.csv
results/representation_trajectory_features.csv
```

and publication outputs under:

```text
results/tables/
results/figures/
```

These are **not** substitutes for the original QuestBench or AskBench files when performing a complete from-raw reproduction.

---

## 11. Data Provenance

QuestBench and AskBench are independent third-party research resources.

They were not created by the SIG project.

### QuestBench

Official repository:

https://github.com/google-deepmind/questbench

Dataset:

https://huggingface.co/datasets/belindazli/QuestBench

### AskBench

Official repository:

https://github.com/jialeuuz/askbench

Evaluation dataset:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Training / trajectory dataset:

https://huggingface.co/datasets/jialeuuz/askbench_train

Users should cite the original benchmark publications when using these resources in derived research.

---

## 12. Licensing

The Apache License 2.0 included in the SIG repository applies to the original SIG software except where otherwise stated.

It does **not** automatically relicense QuestBench, AskBench, or other third-party resources.

The benchmark data are intentionally not bundled with the public SIG software distribution.

Users are responsible for:

- downloading the datasets from their official upstream sources;
- checking the current upstream licenses and terms;
- satisfying applicable attribution requirements;
- satisfying applicable citation requirements; and
- ensuring that their intended use is permitted.

Always rely on the current upstream dataset documentation for authoritative licensing and usage information.

---

## 13. Why Raw Benchmark Data Are Not Included

The public SIG repository deliberately separates original SIG software from third-party benchmark resources.

The intended reproduction model is:

```text
SIG source code
        +
QuestBench obtained from its official source
        +
AskBench obtained from its official source
        ↓
Complete SIG reproduction
        ↓
Derived analyses, tables, and figures
```

This design:

1. preserves benchmark provenance;
2. avoids unnecessary redistribution of third-party data;
3. keeps licensing responsibilities explicit;
4. directs users to authoritative upstream versions; and
5. separates original SIG software from external research resources.

---

## 14. Quick Setup

### Step 1 — QuestBench

Download QuestBench from:

https://huggingface.co/datasets/belindazli/QuestBench

Place:

```text
GSM-Q.csv
GSME-Q.csv
Logic-Q.csv
Planning-Q.csv
```

in:

```text
data/questbench/
```

### Step 2 — AskBench training trajectories

Download from:

https://huggingface.co/datasets/jialeuuz/askbench_train

Place:

```text
mind.jsonl
overconfidence.jsonl
```

in:

```text
data/askbench/train/
```

### Step 3 — AskBench evaluation data

Download from:

https://huggingface.co/datasets/jialeuuz/askbench_bench

Place:

```text
ask_mind.jsonl
ask_mind_bbhde.jsonl
ask_mind_gpqade.jsonl
ask_mind_math500de.jsonl
ask_mind_medqade.jsonl
ask_overconfidence.jsonl
```

in:

```text
data/askbench/eval/
```

### Step 4 — Verify

Your final data structure should be:

```text
data/
├── README.md
├── questbench/
│   ├── GSM-Q.csv
│   ├── GSME-Q.csv
│   ├── Logic-Q.csv
│   └── Planning-Q.csv
└── askbench/
    ├── train/
    │   ├── mind.jsonl
    │   └── overconfidence.jsonl
    └── eval/
        ├── ask_mind.jsonl
        ├── ask_mind_bbhde.jsonl
        ├── ask_mind_gpqade.jsonl
        ├── ask_mind_math500de.jsonl
        ├── ask_mind_medqade.jsonl
        └── ask_overconfidence.jsonl
```

### Step 5 — Run

Windows:

```bat
run_all.bat
```

Linux/macOS:

```bash
./run_all.sh
```

or:

```bash
python run_all.py
```

For the representation-robustness experiment:

```bash
python run_representation_robustness.py
```

---

## 15. Reproducibility

For the complete experimental protocol, dependency information, generated outputs, and reproduction details, see:

```text
README.md
REPRODUCIBILITY.md
QA_REPORT.md
```

at the repository root.

If a raw-data file is missing, download it from the corresponding official source listed above and place it in the exact expected directory before rerunning the full workflow.
