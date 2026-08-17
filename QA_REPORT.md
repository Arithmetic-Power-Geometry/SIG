# Software QA Report

- Unit tests: 6 passed in 0.68s
- Fast end-to-end reproduction ([1/5] Using checked-in QuestBench feature/results tables; use --from-raw for full feature reconstruction
[2/5] Loading checked-in AskBench trajectory feature table
[3/5] Recomputing AskBench evaluation perturbation geometry
[4/5] Refreshing AskBench figures and manifest
[5/5] Running representation/metric robustness analysis
          setting         feature    n  spearman_rho      p_value  rubric_min  rubric_max
          AskMind     n_questions 3226      0.210229 1.501746e-33           2           7
          AskMind     path_length 3226      0.176104 6.961745e-24           2           7
          AskMind    displacement 3226     -0.108079 7.539791e-10           2           7
          AskMind    spiral_ratio 3226      0.211180 7.603996e-34           2           7
          AskMind  turn_curvature 3226      0.177119 3.812000e-24           2           7
          AskMind return_fraction 3226      0.176703 4.881446e-24           2           7
AskOverconfidence     n_questions 3321      0.035659 3.989414e-02           2           4
AskOverconfidence     path_length 3321      0.041810 1.597003e-02           2           4
AskOverconfidence    displacement 3321      0.005390 7.561671e-01           2           4
AskOverconfidence    spiral_ratio 3321      0.035102 4.310358e-02           2           4
AskOverconfidence  turn_curvature 3321      0.033915 5.066549e-02           2           4
AskOverconfidence return_fraction 3321      0.033091 5.654948e-02           2           4

Completed successfully.): PASS
- Representation robustness pipeline: PASS
- Data identity check: the latest supplied QuestBench and AskBench files are byte-identical to the files in the prior reproducibility package used for the checked-in structural results.
- Seed: 2026
- Original SIG code license: Apache-2.0
- Third-party benchmark data retain upstream terms.
