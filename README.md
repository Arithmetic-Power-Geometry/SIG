# Spiral Inquiry Geometry (SIG) - Reproducibility Repository

This repository reproduces the experiments for **Inquiry Has Geometry: A Mathematical Theory of Recursive Question Transformation** using the supplied QuestBench and AskBench files.

## One-click / one-command reproduction

### Windows
Double-click `run_all.bat` (or run it from Command Prompt).

### Linux/macOS
```bash
./run_all.sh
```

Both commands install the pinned dependency ranges, run the test suite, rebuild features from the raw benchmark files, regenerate tables/figures, and write manifests under `results/`.

### GitHub Actions
After uploading this folder to GitHub, open **Actions -> Reproduce SIG results -> Run workflow**. The included `workflow_dispatch` action performs the full `--from-raw` reproduction and uploads `results/` as an artifact.

## Fast verification
If you only want to verify the checked-in feature tables and regenerate AskBench diagnostics:
```bash
python run_all.py
```
For a complete reconstruction from raw data:
```bash
python run_all.py --from-raw
```

## What is implemented
- Formal SIG metric utilities and theorem sanity tests.
- QuestBench structural feature extraction for GSM-Q, GSME-Q, Logic-Q, and Planning-Q.
- Grouped held-out candidate-question evaluation with leakage controls.
- AskBench trajectory extraction from `conversation_history` only.
- Deterministic 8,192-dimensional character hashing representation (3-5 grams) with cosine distance.
- Representation/metric robustness on the same 3,226 AskMind trajectories using word TF-IDF, LSA-256, binary Jaccard geometry, and a deterministic random-text negative control.
- Ordered-logit incremental tests asking whether spiral ratio adds information beyond clarification-turn count and question-token volume.
- Turn-adjusted cross-representation stability analysis for spiral ratio.
- Path length, endpoint displacement, spiral ratio, radial dynamics, local excess/curvature, and return behavior.
- Spearman tests linking AskMind rubric complexity to trajectory geometry.
- Original-to-perturbed AskBench evaluation geometry.

## Important data fact
The supplied AskBench training files contain 13,094 JSONL rows, but only 6,547 rows contain `conversation_history`. The software uses only those 6,547 rows as actual trajectories; it does not double-count paired originals.

## Licensing
Original SIG code is Apache-2.0. Third-party QuestBench and AskBench data remain under their upstream terms and are not relicensed by this repository. See `data/README.md` and `NOTICE`.

## Representation robustness result
The checked-in results deliberately include a falsification control. Raw path statistics correlate with rubric complexity even for random text vectors because longer clarification sequences mechanically create more path opportunities. The stricter ordered-logit analysis controls turn count and token volume: spiral ratio adds significant fit for word TF-IDF, LSA-256, and binary Jaccard representations, while the random-text control is null. See `results/tables/ordinal_incremental_value.csv`.

This distinction is intentional: the repository is designed to expose confounds, not merely maximize positive-looking statistics.
