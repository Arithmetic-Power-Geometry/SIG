from pathlib import Path
import argparse,json,sys,platform
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from sig import questbench
from sig.askbench import build_trajectory_table,build_eval_table,eval_correlations
from sig.representation_robustness import run as run_representation_robustness

def ask_correlations(traj):
    rows=[]
    for setting,g in traj.groupby('setting'):
        for feat in ['n_questions','path_length','displacement','spiral_ratio','turn_curvature','return_fraction']:
            r,p=spearmanr(g.rubric_points,g[feat])
            rows.append({'setting':setting,'feature':feat,'n':len(g),'spearman_rho':r,'p_value':p,'rubric_min':int(g.rubric_points.min()),'rubric_max':int(g.rubric_points.max())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--from-raw',action='store_true',help='rebuild all feature tables from raw benchmark files')
    args=ap.parse_args(); out=ROOT/'results'; (out/'tables').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True)
    if args.from_raw or not (out/'candidate_features.csv').exists():
        print('[1/5] Rebuilding QuestBench feature table and models from raw CSV files')
        qsum,qperf=questbench.run(ROOT/'data/questbench',out,seed=2026)
    else:
        print('[1/5] Using checked-in QuestBench feature/results tables; use --from-raw for full feature reconstruction')
    if args.from_raw or not (out/'askbench_trajectory_features.csv').exists():
        print('[2/5] Rebuilding AskBench trajectory features from JSONL histories')
        traj,counts=build_trajectory_table(ROOT/'data/askbench/train'); traj.to_csv(out/'askbench_trajectory_features.csv',index=False); counts.to_csv(out/'tables/askbench_counts.csv',index=False)
    else:
        print('[2/5] Loading checked-in AskBench trajectory feature table')
        traj=pd.read_csv(out/'askbench_trajectory_features.csv'); counts=pd.read_csv(out/'tables/askbench_counts.csv')
    tsum=traj.groupby('setting').agg(trajectories=('id','count'),median_states=('n_states','median'),median_questions=('n_questions','median'),median_spiral_ratio=('spiral_ratio','median'),median_path_length=('path_length','median'),median_displacement=('displacement','median')).reset_index(); tsum.to_csv(out/'tables/askbench_summary.csv',index=False)
    corr=ask_correlations(traj); corr.to_csv(out/'tables/askbench_rubric_geometry.csv',index=False)
    print('[3/5] Recomputing AskBench evaluation perturbation geometry')
    ev=build_eval_table(ROOT/'data/askbench/eval'); ev.to_csv(out/'askbench_eval_geometry.csv',index=False); ec=eval_correlations(ev); ec.to_csv(out/'tables/askbench_eval_correlations.csv',index=False)
    print('[4/5] Refreshing AskBench figures and manifest')
    import matplotlib.pyplot as plt
    mind=corr[corr.setting=='AskMind']; fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(mind.feature,mind.spearman_rho); ax.axhline(0,linewidth=.8); ax.set_ylabel('Spearman correlation with rubric-point count'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(out/'figures/askmind_rubric_geometry.pdf'); fig.savefig(out/'figures/askmind_rubric_geometry.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.barh(ec.file,ec.spearman_rho); ax.axvline(0,linewidth=.8); ax.set_xlabel('Spearman rho: perturbation displacement vs rubric points'); fig.tight_layout(); fig.savefig(out/'figures/askbench_eval_rho.pdf'); fig.savefig(out/'figures/askbench_eval_rho.png',dpi=220); plt.close(fig)
    manifest={'seed':2026,'questbench_candidate_rows':126500,'questbench_instances':11790,'askbench_raw_rows':int(counts.raw_rows.sum()),'askbench_usable_trajectories':int(counts.usable_trajectories.sum()),'askbench_eval_rows':int(len(ev)),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'full_from_raw':bool(args.from_raw)}
    print('[5/5] Running representation/metric robustness analysis')
    run_representation_robustness(ROOT/'data/askbench/train',out,seed=2026)
    manifest['representation_robustness']='char-ngram hash; word TF-IDF; LSA-256; binary Jaccard; random-text control'
    (out/'run_manifest_v3.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(corr.to_string(index=False)); print('\nCompleted successfully.')
if __name__=='__main__': main()
