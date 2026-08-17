# SIG Quality-Assurance Report

## Project

**Spiral Inquiry Geometry (SIG)**

Associated research work:

> Akhtar, M. A. K. (2026). *Inquiry Has Geometry: A Mathematical Theory of Recursive Question Transformation* (Version V1). Zenodo.  
> https://doi.org/10.5281/zenodo.21977561

Repository:

https://github.com/Arithmetic-Power-Geometry/SIG

## Purpose

This document records the repository-level checks expected before a
release or manuscript submission. It is intended as a transparent QA
checklist, not as a substitute for the run manifests or the outputs of
GitHub Actions.

## 1. Repository structure

Expected core components include:

```text
.github/
data/
results/
src/
tests/
LICENSE
NOTICE
README.md
REPRODUCIBILITY.md
QA_REPORT.md
requirements.txt
run_all.py
run_all.bat
run_all.sh
run_representation_robustness.py
```

Status: **PASS** when these required components are present for the
release being audited.

## 2. Data separation

Expected design:

- raw QuestBench data are not required to be committed;
- raw AskBench data are not required to be committed;
- `data/README.md` documents upstream sources and required paths;
- third-party datasets retain upstream terms.

Status: **PASS** for the intended public-repository design.

## 3. Full-reproduction workflow

The full workflow should be able to reconstruct results from documented
raw benchmark inputs using:

```bash
python run_all.py --from-raw
```

The GitHub Actions workflow should perform the corresponding automated
reproduction and upload `results/` as an artifact.

Status should be determined from the latest successful GitHub Actions
run and its run manifest.

## 4. Test suite

Run:

```bash
pytest -q
```

A release should not be labeled QA-passing unless the test suite
completes successfully.

The exact number of tests can change as the repository evolves; the
current GitHub Actions log is authoritative for a particular release.

## 5. AskBench trajectory integrity

The dataset version used for the associated study contains:

```text
13,094 total training JSONL rows
6,547 usable multi-turn trajectories
3,226 AskMind trajectories
3,321 AskOverconfidence trajectories
```

Only records with usable `conversation_history` should enter the
multi-turn trajectory analysis.

Expected status: **PASS** if the run manifest reproduces these counts for
the documented dataset version.

## 6. QuestBench reconstruction

The full-from-raw workflow should rebuild QuestBench candidate-level
features for the documented benchmark files and regenerate the
candidate-question evaluation outputs.

For the associated V1 results, the run should reconstruct the dataset
counts recorded in the authoritative run manifest.

## 7. Manuscript/result consistency

The manuscript, repository results, and latest full-reproduction
artifact should agree.

Key V1 rounded values include:

### Logic-Q, nonlinear SIG model

```text
ROC-AUC            0.839
Average precision  0.480
Top-1 accuracy     0.552
```

### Planning-Q, nonlinear SIG model

```text
ROC-AUC            0.945
Average precision  0.902
Top-1 accuracy     0.365
```

Expected status: **PASS** only when the authoritative repository outputs
and manuscript report the same values after rounding.

## 8. Representation robustness

The robustness workflow should evaluate the same AskMind trajectories
under multiple substantive representations and a random-text negative
control.

Expected substantive conditions include:

```text
character n-gram hashing
word TF-IDF
LSA-256
binary Jaccard geometry
```

plus:

```text
random-text negative control
```

The analysis should preserve the distinction between raw correlations
and stricter incremental tests controlling for clarification-turn count
and question-token volume.

## 9. Negative-control interpretation

The random-text condition is intentionally retained.

A positive raw path correlation under a random representation is not
treated as evidence for semantic geometry. Instead, it demonstrates a
mechanical trajectory-length component that motivates adjusted analyses.

Expected status: **PASS** when the generated results and documentation
retain this falsification logic rather than suppressing the null/control
finding.

## 10. Generated tables and figures

The reproduction workflow should regenerate the analysis tables and
figures under:

```text
results/tables/
results/figures/
```

The artifact may contain more diagnostic figures than the manuscript.
This is expected: only figures selected for the paper need to appear in
the manuscript.

For manuscript QA, every table and figure that appears in the manuscript
should be explicitly cited in the text before it appears.

## 11. Run manifests

A release should have one clearly authoritative run manifest for the
current full reproduction.

Legacy/intermediate manifests should not be presented in a way that can
be mistaken for the current authoritative run.

Recommended release practice:

```text
results/run_manifest.json
```

should identify the current full reproduction, including dataset/run
metadata sufficient to distinguish it from older runs.

## 12. Licensing QA

Expected:

- original SIG code: Apache License 2.0;
- third-party benchmark data: upstream terms;
- third-party packages/models: upstream terms;
- no claim that Apache-2.0 relicenses QuestBench or AskBench.

Status: **PASS** when `LICENSE`, `NOTICE`, and `data/README.md` preserve
this distinction.

## 13. Release hygiene

Before a public release, verify that the repository contains no:

```text
TODO
FIXME
TBD
placeholder
note to author
remove before submission
insert here
```

unless intentionally included in developer documentation.

Also verify that no credentials, API keys, private tokens, or local
environment files are committed.

## 14. Manuscript hygiene

For the associated final manuscript package, verify:

- all tables are cited;
- all figures are cited;
- citations occur before the corresponding float where required;
- no unresolved LaTeX references remain;
- no unresolved bibliography citations remain;
- no author instructions or drafting placeholders remain;
- manuscript values match the authoritative artifact;
- bibliography metadata is checked against authoritative publication
  sources where practical.

## 15. Final release criterion

A SIG release is considered repository-QA-ready when:

```text
tests pass
+
full reproduction succeeds
+
authoritative manifest is generated
+
tables/figures are regenerated
+
artifact uploads successfully
+
manuscript values match the artifact
+
licensing/provenance documentation is present
```

The GitHub Actions log and generated run manifest remain the authoritative
evidence for a specific computational run.

## Citation

> Akhtar, M. A. K. (2026). *Inquiry Has Geometry: A Mathematical Theory of Recursive Question Transformation* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.21977561
