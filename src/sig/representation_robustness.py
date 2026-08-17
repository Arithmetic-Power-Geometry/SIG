from __future__ import annotations
import hashlib, json, math, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr, chi2
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from statsmodels.miscmodels.ordinal_model import OrderedModel

EPS=1e-12
INTERROG=re.compile(r'(\?|？|\bwhat\b|\bwhich\b|\bwhy\b|\bhow\b|\bcould you\b|\bcan you\b|\bwould you\b|\bclarif|\bprovide\b|请问|能否|可以.*吗|什么|哪个|为何|为什么)',re.I)


def _read_jsonl(path:Path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def _states(obj, setting='AskMind'):
    ch=obj.get('conversation_history') or []
    if not ch: return []
    initial=str(obj.get('degraded_question' if setting=='AskMind' else 'overconfidence_question') or ch[0].get('content',''))
    assistants=[str(x.get('content','')) for x in ch if x.get('role')=='assistant']
    if not assistants: return [initial]
    # Exclude final assistant answer. Keep intermediate clarification-like turns.
    inter=assistants[:-1]
    qs=[t for t in inter if INTERROG.search(t)]
    if not qs and inter: qs=inter
    return [initial]+qs


def load_askmind(train_dir:Path):
    traj=[]; texts=[]
    for obj in _read_jsonl(train_dir/'mind.jsonl'):
        if not obj.get('conversation_history'): continue
        st=_states(obj,'AskMind')
        if not st: continue
        start=len(texts); texts.extend(st); end=len(texts)
        pts=obj.get('required_points') or []
        q=st[1:]
        traj.append(dict(id=obj.get('id',''), start=start, end=end, rubric_points=len(pts),
                         n_states=len(st), n_questions=max(0,len(st)-1),
                         total_question_tokens=sum(len(re.findall(r'\w+',x,flags=re.UNICODE)) for x in q),
                         total_question_chars=sum(len(x) for x in q)))
    return pd.DataFrame(traj), texts


def _cosine_row_distance(X,i,j):
    if sparse.issparse(X):
        sim=float(X[i].multiply(X[j]).sum())
    else:
        sim=float(np.dot(X[i],X[j]))
    return float(np.clip(1.0-sim,0.0,2.0))


def _jaccard_row_distance(X,i,j):
    a=X[i]; b=X[j]
    inter=float(a.multiply(b).sum())
    union=float(a.getnnz()+b.getnnz()-inter)
    return 0.0 if union<=0 else 1.0-inter/union


def _geometry(X, start, end, metric='cosine'):
    n=end-start
    if n<2:
        return dict(path_length=0.,displacement=0.,spiral_ratio=1.,turn_curvature=0.,return_fraction=0.,max_radius=0.,mean_radius=0.)
    dist=_jaccard_row_distance if metric=='jaccard' else _cosine_row_distance
    idx=range(start,end)
    step=np.array([dist(X,i,i+1) for i in range(start,end-1)])
    radii=np.array([dist(X,start,i) for i in idx])
    L=float(step.sum()); D=float(radii[-1]); S=L/max(D,EPS) if D>EPS else (1. if L<=EPS else L/EPS)
    dr=np.diff(radii); returns=float(np.mean(dr < -1e-9)) if len(dr) else 0.
    excess=[]
    for i in range(start+1,end-1):
        e=dist(X,i-1,i)+dist(X,i,i+1)-dist(X,i-1,i+1)
        excess.append(max(0.,e))
    ex=float(sum(excess)); curvature=ex/max(L,EPS) if L>EPS else 0.
    return dict(path_length=L,displacement=D,spiral_ratio=float(S),turn_curvature=float(curvature),return_fraction=returns,
                max_radius=float(radii.max()),mean_radius=float(radii.mean()))


def _random_embeddings(texts, dim=384, seed=2026):
    out=np.empty((len(texts),dim),dtype=np.float32)
    for i,t in enumerate(texts):
        h=hashlib.blake2b((str(seed)+'|'+t).encode('utf8'),digest_size=16).digest()
        s=int.from_bytes(h[:8],'little',signed=False) % (2**32-1)
        rng=np.random.default_rng(s); v=rng.standard_normal(dim).astype(np.float32)
        out[i]=v/max(np.linalg.norm(v),EPS)
    return out


def build_representations(texts, seed=2026):
    reps={}
    char=HashingVectorizer(n_features=2**13,alternate_sign=False,analyzer='char_wb',ngram_range=(3,5),norm='l2',lowercase=True)
    reps['char-ngram hash']=('cosine',char.transform(texts))
    tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),lowercase=True,min_df=2,max_features=40000,sublinear_tf=True,norm='l2')
    Xtf=tf.fit_transform(texts)
    reps['word TF-IDF']=('cosine',Xtf)
    # Latent semantic analysis: corpus-derived, label-free semantic factor space.
    k=min(256,max(2,min(Xtf.shape)-1))
    lsa=make_pipeline(TruncatedSVD(n_components=k,algorithm='randomized',n_iter=7,random_state=seed),Normalizer(copy=False))
    reps['LSA-256']=('cosine',lsa.fit_transform(Xtf))
    cv=CountVectorizer(analyzer='word',ngram_range=(1,2),lowercase=True,binary=True,min_df=1,max_features=50000)
    Xbin=cv.fit_transform(texts).astype(np.float32)
    reps['binary Jaccard']=('jaccard',Xbin)
    reps['random-text control']=('cosine',_random_embeddings(texts,dim=384,seed=seed))
    return reps


def partial_spearman(x,y,controls):
    x=np.asarray(x,float); y=np.asarray(y,float); C=np.asarray(controls,float)
    mask=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(C),axis=1)
    x=rankdata(x[mask]); y=rankdata(y[mask]); C=np.column_stack([rankdata(C[mask,j]) for j in range(C.shape[1])])
    Z=np.column_stack([np.ones(len(x)),C])
    bx=np.linalg.lstsq(Z,x,rcond=None)[0]; by=np.linalg.lstsq(Z,y,rcond=None)[0]
    rx=x-Z@bx; ry=y-Z@by
    return spearmanr(rx,ry)


def bh_fdr(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); q=np.empty(n,float); prev=1.
    for rank,idx in reversed(list(enumerate(order,start=1))):
        prev=min(prev,p[idx]*n/rank); q[idx]=prev
    return np.clip(q,0,1)


def ordinal_incremental_value(df):
    """Ordered-logit likelihood-ratio test: does spiral ratio add beyond turns and token volume?"""
    rows=[]
    for rep,g in df.groupby('representation',sort=False):
        y=g.rubric_points.astype(int)
        X0=pd.DataFrame({'n_questions':g.n_questions.astype(float),'log_tokens':np.log1p(g.total_question_tokens.astype(float))},index=g.index)
        X1=X0.copy(); sr=g.spiral_ratio.replace([np.inf,-np.inf],np.nan); X1['spiral_ratio']=sr.fillna(sr.median())
        X0=(X0-X0.mean())/X0.std(ddof=0).replace(0,1); X1=(X1-X1.mean())/X1.std(ddof=0).replace(0,1)
        m0=OrderedModel(y,X0,distr='logit').fit(method='bfgs',disp=False,maxiter=400)
        m1=OrderedModel(y,X1,distr='logit').fit(method='bfgs',disp=False,maxiter=400)
        lr=2*(m1.llf-m0.llf); p=float(chi2.sf(lr,1))
        rows.append({'representation':rep,'n':len(g),'spiral_coef_std':float(m1.params['spiral_ratio']),
                     'spiral_wald_p':float(m1.pvalues['spiral_ratio']),'likelihood_ratio_chi2':float(lr),
                     'likelihood_ratio_p':p,'delta_aic_baseline_minus_spiral':float(m0.aic-m1.aic)})
    z=pd.DataFrame(rows); z['lr_fdr_q']=bh_fdr(z.likelihood_ratio_p.values); return z

def run(train_dir:Path, out_dir:Path, seed=2026):
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'tables').mkdir(exist_ok=True); (out_dir/'figures').mkdir(exist_ok=True)
    meta,texts=load_askmind(train_dir)
    reps=build_representations(texts,seed=seed)
    long=[]
    for rep,(metric,X) in reps.items():
        for r in meta.itertuples(index=False):
            g=_geometry(X,int(r.start),int(r.end),metric=metric)
            long.append({'id':r.id,'representation':rep,'rubric_points':r.rubric_points,'n_questions':r.n_questions,
                         'total_question_tokens':r.total_question_tokens,'total_question_chars':r.total_question_chars,**g})
    df=pd.DataFrame(long); df.to_csv(out_dir/'representation_trajectory_features.csv',index=False)
    feats=['path_length','displacement','spiral_ratio','turn_curvature','return_fraction']
    rows=[]
    for rep,g in df.groupby('representation',sort=False):
        controls=np.column_stack([g.n_questions.values,g.total_question_tokens.values])
        for feat in feats:
            rho,p=spearmanr(g.rubric_points,g[feat])
            trho,tp=partial_spearman(g.rubric_points.values,g[feat].values,g[['n_questions']].values)
            prho,pp=partial_spearman(g.rubric_points.values,g[feat].values,controls)
            rows.append(dict(representation=rep,feature=feat,n=len(g),spearman_rho=float(rho),p_value=float(p),
                             partial_rho_turns=float(trho),partial_p_turns=float(tp),
                             partial_rho_turns_tokens=float(prho),partial_p_value=float(pp)))
    corr=pd.DataFrame(rows)
    corr['fdr_q']=bh_fdr(corr.p_value.values); corr['turn_fdr_q']=bh_fdr(corr.partial_p_turns.values); corr['partial_fdr_q']=bh_fdr(corr.partial_p_value.values)
    corr.to_csv(out_dir/'tables/representation_robustness.csv',index=False)
    ordinal=ordinal_incremental_value(df); ordinal.to_csv(out_dir/'tables/ordinal_incremental_value.csv',index=False)
    # Cross-representation stability: Spearman rank agreement of each SIG descriptor.
    stab=[]
    for feat in ['path_length','displacement','spiral_ratio','turn_curvature']:
        wide=df.pivot(index='id',columns='representation',values=feat)
        cols=list(wide.columns)
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                rho,p=spearmanr(wide[cols[i]],wide[cols[j]])
                stab.append(dict(feature=feat,representation_a=cols[i],representation_b=cols[j],n=len(wide),spearman_rho=float(rho),p_value=float(p)))
    stability=pd.DataFrame(stab); stability['fdr_q']=bh_fdr(stability.p_value.values)
    stability.to_csv(out_dir/'tables/cross_representation_stability.csv',index=False)
    # Stability after removing the dominant effect of number of clarification turns.
    residuals={}
    for rep,g in df.groupby('representation',sort=False):
        g=g.sort_values('id'); y=rankdata(g.spiral_ratio.values)
        dummies=pd.get_dummies(g.n_questions.astype(str),drop_first=True).values.astype(float)
        Z=np.column_stack([np.ones(len(g)),dummies])
        residuals[rep]=pd.Series(y-Z@np.linalg.lstsq(Z,y,rcond=None)[0],index=g.id.values)
    RW=pd.DataFrame(residuals)
    rstab=[]
    rcols=list(RW.columns)
    for i in range(len(rcols)):
        for j in range(i+1,len(rcols)):
            rr,pp=spearmanr(RW[rcols[i]],RW[rcols[j]])
            rstab.append({'representation_a':rcols[i],'representation_b':rcols[j],'n':len(RW),'spearman_rho':float(rr),'p_value':float(pp)})
    rstab=pd.DataFrame(rstab); rstab['fdr_q']=bh_fdr(rstab.p_value.values)
    rstab.to_csv(out_dir/'tables/turn_adjusted_spiral_stability.csv',index=False)
    # Directional replication summary excluding random negative control.
    substantive=corr[corr.representation!='random-text control'].copy()
    directional=[]
    expected={'path_length':1,'spiral_ratio':1,'turn_curvature':1,'displacement':-1}
    for feat,sgn in expected.items():
        x=substantive[substantive.feature==feat]
        directional.append({'feature':feat,'expected_direction':'positive' if sgn>0 else 'non-positive',
                            'representations':len(x),'direction_matches':int(((x.spearman_rho*sgn)>0).sum()),
                            'fdr_significant_matches':int((((x.spearman_rho*sgn)>0)&(x.fdr_q<.05)).sum()),
                            'turn_adjusted_direction_matches':int(((x.partial_rho_turns*sgn)>0).sum()),
                            'turn_adjusted_fdr_significant_matches':int((((x.partial_rho_turns*sgn)>0)&(x.turn_fdr_q<.05)).sum()),
                            'turns_tokens_direction_matches':int(((x.partial_rho_turns_tokens*sgn)>0).sum()),
                            'turns_tokens_fdr_significant_matches':int((((x.partial_rho_turns_tokens*sgn)>0)&(x.partial_fdr_q<.05)).sum())})
    pd.DataFrame(directional).to_csv(out_dir/'tables/directional_replication.csv',index=False)
    # Figures
    import matplotlib.pyplot as plt
    plot=corr[corr.feature.isin(['path_length','displacement','spiral_ratio','turn_curvature'])].copy()
    reps_order=['char-ngram hash','word TF-IDF','LSA-256','binary Jaccard','random-text control']
    feats_order=['path_length','spiral_ratio','turn_curvature','displacement']
    x=np.arange(len(reps_order)); width=.18
    fig,ax=plt.subplots(figsize=(10,5.5))
    for k,feat in enumerate(feats_order):
        vals=[float(plot[(plot.representation==r)&(plot.feature==feat)].spearman_rho.iloc[0]) for r in reps_order]
        ax.bar(x+(k-1.5)*width,vals,width,label=feat.replace('_',' '))
    ax.axhline(0,linewidth=.8); ax.set_xticks(x,reps_order,rotation=20,ha='right'); ax.set_ylabel('Spearman rho with rubric complexity'); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out_dir/'figures/representation_robustness.pdf'); fig.savefig(out_dir/'figures/representation_robustness.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for k,feat in enumerate(feats_order):
        vals=[float(plot[(plot.representation==r)&(plot.feature==feat)].partial_rho_turns.iloc[0]) for r in reps_order]
        ax.bar(x+(k-1.5)*width,vals,width,label=feat.replace('_',' '))
    ax.axhline(0,linewidth=.8); ax.set_xticks(x,reps_order,rotation=20,ha='right'); ax.set_ylabel('Turn-adjusted rank association'); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(out_dir/'figures/representation_partial_robustness.pdf'); fig.savefig(out_dir/'figures/representation_partial_robustness.png',dpi=220); plt.close(fig)
    # Stability heatmap for raw spiral ratio
    wide=df.pivot(index='id',columns='representation',values='spiral_ratio')[reps_order]
    M=wide.corr(method='spearman').values
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(M,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(reps_order)),reps_order,rotation=35,ha='right'); ax.set_yticks(range(len(reps_order)),reps_order)
    for i in range(len(reps_order)):
        for j in range(len(reps_order)): ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Spearman rho'); ax.set_title('Cross-representation stability of spiral ratio'); fig.tight_layout()
    fig.savefig(out_dir/'figures/spiral_stability_heatmap.pdf'); fig.savefig(out_dir/'figures/spiral_stability_heatmap.png',dpi=220); plt.close(fig)
    RMcorr=RW[reps_order].corr(method='spearman').values
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(RMcorr,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(reps_order)),reps_order,rotation=35,ha='right'); ax.set_yticks(range(len(reps_order)),reps_order)
    for i in range(len(reps_order)):
        for j in range(len(reps_order)): ax.text(j,i,f'{RMcorr[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Spearman rho'); ax.set_title('Turn-adjusted stability of spiral ratio'); fig.tight_layout()
    fig.savefig(out_dir/'figures/turn_adjusted_spiral_stability.pdf'); fig.savefig(out_dir/'figures/turn_adjusted_spiral_stability.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8)); oo=ordinal.set_index('representation').loc[reps_order].reset_index(); ax.bar(oo.representation,oo.delta_aic_baseline_minus_spiral); ax.axhline(0,linewidth=.8); ax.set_ylabel('AIC improvement from adding spiral ratio'); ax.tick_params(axis='x',rotation=20); fig.tight_layout()
    fig.savefig(out_dir/'figures/ordinal_incremental_value.pdf'); fig.savefig(out_dir/'figures/ordinal_incremental_value.png',dpi=220); plt.close(fig)
    return df,corr,stability

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--train-dir',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True); p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args(); _,c,s=run(a.train_dir,a.out_dir,a.seed); print(c.to_string(index=False)); print('\nStability rows:',len(s))
