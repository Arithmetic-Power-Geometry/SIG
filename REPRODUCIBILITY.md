# Reproducibility specification

- Random seed: 2026.
- QuestBench splits: grouped by underlying problem instance; no candidate from one instance crosses train/test.
- Ground-truth candidate identity and held-out values are outcomes only, not predictor features.
- AskBench trajectory states: perturbed initial prompt plus intermediate assistant clarification questions; final answers are excluded from geometry.
- AskBench representation: stateless `HashingVectorizer`, character 3-5 grams, 8192 features, L2 normalization; distance is cosine distance.
- AskBench complexity outcome: number of supplied `required_points` (AskMind) or `misleading_points` (AskOverconfidence).
- No proprietary LLM/API call is needed for the reported analysis.
- The official AskBench judge accuracy/coverage metrics are not re-created or claimed.


## Representation audit
`run_all.py` also executes `sig.representation_robustness`. It holds all 3,226 AskMind trajectories fixed and changes only the state representation/metric. Outputs include `representation_robustness.csv`, `ordinal_incremental_value.csv`, `cross_representation_stability.csv`, `turn_adjusted_spiral_stability.csv`, and associated PDF/PNG figures. The random-text control is deterministic under seed 2026.
