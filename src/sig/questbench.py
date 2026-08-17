from __future__ import annotations
import ast, hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mannwhitneyu

EPS=1e-9

def _safe_literal(x, default):
    if not isinstance(x,str): return x
    s=x.strip()
    if s.startswith('frozenset(') and s.endswith(')'):
        try: return set(ast.literal_eval(s[len('frozenset('):-1]))
        except Exception: return default
    try: return ast.literal_eval(s)
    except Exception: return default

def _strip_neg(s:str)->str:
    s=str(s).strip().lower()
    return re.sub(r'^not\s+','',s).strip('() ')

def _tokens(s:str):
    return set(re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', str(s).lower()))

def _norm(v, mx): return float(v)/(float(mx)+EPS)

def equation_graph(equations_obj):
    G=nx.DiGraph()
    if isinstance(equations_obj,str):
        equations_obj=_safe_literal(equations_obj,{})
    keys=list(equations_obj.keys()) if isinstance(equations_obj,dict) else []
    for eq in keys:
        if '=' not in eq: continue
        lhs,rhs=eq.split('=',1)
        lhs_vars=re.findall(r'[A-Za-z][A-Za-z0-9_]*',lhs)
        rhs_vars=re.findall(r'[A-Za-z][A-Za-z0-9_]*',rhs)
        for y in lhs_vars:
            G.add_node(y)
            for x in rhs_vars:
                if x!=y: G.add_edge(x,y)
    return G

def goal_from_csp(csp:str):
    m=re.search(r'Goal:\s*\n\s*([A-Za-z][A-Za-z0-9_]*)',str(csp))
    return m.group(1) if m else None

def graph_features(G, candidate, goal):
    c,g=str(candidate),str(goal)
    nodes=list(G.nodes())
    if c not in G: G.add_node(c)
    if g and g not in G: G.add_node(g)
    und=G.to_undirected()
    n=max(1,len(G)); e=max(1,G.number_of_edges())
    try: ddir=nx.shortest_path_length(G,c,g) if g else np.nan
    except: ddir=np.nan
    try: dund=nx.shortest_path_length(und,c,g) if g else np.nan
    except: dund=np.nan
    deg=G.degree(c); indeg=G.in_degree(c); outdeg=G.out_degree(c)
    try: anc=len(nx.ancestors(G,c))
    except: anc=0
    try: desc=len(nx.descendants(G,c))
    except: desc=0
    return {
      'directed_goal_proximity': 0.0 if pd.isna(ddir) else 1/(1+ddir),
      'undirected_goal_proximity': 0.0 if pd.isna(dund) else 1/(1+dund),
      'degree_norm': deg/max(1,n-1), 'indegree_norm': indeg/max(1,n-1),
      'outdegree_norm': outdeg/max(1,n-1), 'ancestor_fraction': anc/max(1,n-1),
      'descendant_fraction': desc/max(1,n-1), 'graph_density': nx.density(G) if n>1 else 0.0,
      'graph_nodes': n, 'graph_edges': G.number_of_edges()
    }

def build_gsm(df, domain):
    rows=[]
    for ix,r in df.iterrows():
        poss=_safe_literal(r.get('Possible Questions'),[])
        gt=str(r.get('GT Question'))
        G=equation_graph(r.get('Equations','{}'))
        goal=goal_from_csp(r.get('CSP',''))
        group=hashlib.sha1(str(r.get('CSP','')).encode()).hexdigest()[:16]
        var_desc=_safe_literal(r.get('Variables'),{})
        goal_desc=str(var_desc.get(goal,''))
        for cand in poss:
            cand=str(cand)
            f=graph_features(G,cand,goal)
            f.update({
                'domain':domain,'group':group,'candidate':cand,'goal':goal,
                'label':int(cand==gt),'depth':float(r.get('depth',0)),
                'candidate_goal_token_jaccard': len(_tokens(var_desc.get(cand,''))&_tokens(goal_desc))/max(1,len(_tokens(var_desc.get(cand,''))|_tokens(goal_desc))),
                'candidate_name_goal_overlap': len(_tokens(cand)&_tokens(goal or ''))/max(1,len(_tokens(cand)|_tokens(goal or ''))),
                'choice_count':len(poss)
            })
            rows.append(f)
    return pd.DataFrame(rows)

def build_logic(df):
    rows=[]
    for ix,r in df.iterrows():
        rules=_safe_literal(r['rules'],[])
        G=nx.DiGraph()
        for rule in rules:
            if not isinstance(rule,list) or len(rule)<2: continue
            concl=_strip_neg(rule[-1]); G.add_node(concl)
            for prem in rule[:-1]: G.add_edge(_strip_neg(prem),concl)
        goal=_strip_neg(r['goal'])
        candset=sorted(list(_safe_literal(r['all_valid_qs'],set())))
        gtset=set(_safe_literal(r['gt_qs'],[]))
        known=set(_safe_literal(r.get('known_facts','[]'),[]))|set(_safe_literal(r.get('known_untrue_facts','[]'),[]))
        for cand in candset:
            f=graph_features(G,_strip_neg(cand),goal)
            f.update({
                'domain':'Logic-Q','group':f'logic-{ix}','candidate':str(cand),'goal':goal,
                'label':int(cand in gtset),'depth':float(r.get('max_depth',0)),
                'candidate_goal_token_jaccard':0.0,'candidate_name_goal_overlap':float(_strip_neg(cand)==goal),
                'choice_count':len(candset),'known_indicator':int(cand in known),
                'constraint_count':float(r.get('num_constraints',0)),'reported_num_vars':float(r.get('num_vars',0))
            })
            rows.append(f)
    return pd.DataFrame(rows)

def lit_parts(s):
    toks=re.findall(r'[A-Za-z][A-Za-z0-9_-]*',str(s).lower())
    toks=[t for t in toks if t!='not']
    return (toks[0] if toks else '', set(toks[1:]))

def build_planning(df):
    rows=[]
    for ix,r in df.iterrows():
        candset=list(_safe_literal(r['all_valid_qs'],[]))
        gtobj=_safe_literal(r['gt_qs'],set())
        gtset=set(gtobj) if not isinstance(gtobj,str) else {gtobj}
        conds=[x.strip() for x in str(r['conditions']).splitlines() if x.strip()]
        goals=[x.strip() for x in str(r['goals']).splitlines() if x.strip()]
        goal_pred=[lit_parts(x) for x in goals]
        cond_pred=[lit_parts(x) for x in conds]
        all_objs=set().union(*(o for _,o in goal_pred+cond_pred)) if goal_pred+cond_pred else set()
        for cand in candset:
            cp,co=lit_parts(cand)
            gp={p for p,_ in goal_pred}; go=set().union(*(o for _,o in goal_pred)) if goal_pred else set()
            condp={p for p,_ in cond_pred}; condo=set().union(*(o for _,o in cond_pred)) if cond_pred else set()
            same_pred_goal=int(cp in gp); same_pred_cond=int(cp in condp)
            obj_goal=len(co&go)/max(1,len(co|go)); obj_cond=len(co&condo)/max(1,len(co|condo))
            f={
                'directed_goal_proximity':float(same_pred_goal)*(0.5+0.5*obj_goal),
                'undirected_goal_proximity':max(obj_goal, 0.5*same_pred_goal),
                'degree_norm':len(co)/max(1,len(all_objs)), 'indegree_norm':0.0,'outdegree_norm':0.0,
                'ancestor_fraction':obj_cond,'descendant_fraction':obj_goal,
                'graph_density':0.0,'graph_nodes':float(r.get('num_vars',0)),'graph_edges':len(conds)+len(goals),
                'domain':'Planning-Q','group':f'plan-{ix}','candidate':str(cand),'goal':' ; '.join(goals),
                'label':int(cand in gtset),'depth':float(r.get('min_depth',0)),
                'candidate_goal_token_jaccard':obj_goal,'candidate_name_goal_overlap':float(same_pred_goal),
                'choice_count':len(candset),'known_indicator':int(str(cand) in conds),
                'constraint_count':len(conds)+len(goals),'reported_num_vars':float(r.get('num_vars',0)),
                'same_predicate_condition':same_pred_cond
            }
            rows.append(f)
    return pd.DataFrame(rows)

FEATURES=['directed_goal_proximity','undirected_goal_proximity','degree_norm','indegree_norm','outdegree_norm','ancestor_fraction','descendant_fraction','graph_density','depth','candidate_goal_token_jaccard','candidate_name_goal_overlap','choice_count','constraint_count','reported_num_vars','known_indicator','same_predicate_condition']

def make_candidate_table(data_dir:Path):
    parts=[]
    for fn,dom in [('GSM-Q.csv','GSM-Q'),('GSME-Q.csv','GSME-Q')]:
        parts.append(build_gsm(pd.read_csv(data_dir/fn),dom))
    parts.append(build_logic(pd.read_csv(data_dir/'Logic-Q.csv')))
    parts.append(build_planning(pd.read_csv(data_dir/'Planning-Q.csv')))
    out=pd.concat(parts,ignore_index=True,sort=False)
    for c in FEATURES:
        if c not in out: out[c]=0.0
    out[FEATURES]=out[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return out

def split_groups(df, seed=2026):
    groups=df['group'].astype(str)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed)
    tr,te=next(splitter.split(df,df['label'],groups))
    return tr,te

def evaluate_domain(dfd, seed=2026):
    tr,te=split_groups(dfd,seed)
    train,test=dfd.iloc[tr],dfd.iloc[te]
    feature_sets={
      'Proximity-only':['directed_goal_proximity','undirected_goal_proximity'],
      'Centrality-only':['degree_norm','indegree_norm','outdegree_norm','ancestor_fraction','descendant_fraction'],
      'SIG-structural':FEATURES,
    }
    res=[]; preds={}
    for name,fs in feature_sets.items():
        pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('clf',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=seed))])
        pipe.fit(train[fs],train['label'])
        p=pipe.predict_proba(test[fs])[:,1]
        preds[name]=p
        res.append({'model':name,'roc_auc':roc_auc_score(test.label,p),'average_precision':average_precision_score(test.label,p),'n_test_candidates':len(test),'positive_rate':test.label.mean()})
    # random forest nonlinear check
    rf=RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=5,class_weight='balanced_subsample',random_state=seed,n_jobs=-1)
    rf.fit(train[FEATURES],train.label); p=rf.predict_proba(test[FEATURES])[:,1]
    preds['SIG-nonlinear']=p
    res.append({'model':'SIG-nonlinear','roc_auc':roc_auc_score(test.label,p),'average_precision':average_precision_score(test.label,p),'n_test_candidates':len(test),'positive_rate':test.label.mean()})
    # top-1 question selection per instance for each score
    test2=test[['group','label']].copy()
    for name,p in preds.items():
        test2[name]=p
    top1={}
    for name in preds:
        chosen=test2.loc[test2.groupby('group')[name].idxmax()]
        top1[name]=float(chosen.label.mean())
    for r in res:r['top1_accuracy']=top1[r['model']]
    return pd.DataFrame(res),test,preds

def run(data_dir:Path,out_dir:Path,seed=2026):
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'tables').mkdir(exist_ok=True); (out_dir/'figures').mkdir(exist_ok=True)
    cand=make_candidate_table(data_dir)
    cand.to_csv(out_dir/'candidate_features.csv',index=False)
    # dataset summary
    summaries=[]
    for d,g in cand.groupby('domain'):
        summaries.append({'Domain':d,'Candidates':len(g),'Instances':g['group'].nunique(),'Ground-truth candidates':int(g.label.sum()),'Positive rate':g.label.mean(),'Median choices':g.groupby('group').size().median()})
    summary=pd.DataFrame(summaries).sort_values('Domain')
    summary.to_csv(out_dir/'tables/dataset_summary.csv',index=False)
    allres=[]; test_store={}
    for d,g in cand.groupby('domain'):
        rr,test,preds=evaluate_domain(g,seed)
        rr.insert(0,'domain',d); allres.append(rr); test_store[d]=(test,preds)
    results=pd.concat(allres,ignore_index=True)
    results.to_csv(out_dir/'tables/model_performance.csv',index=False)
    # geometry effect: GT vs non-GT for goal proximity
    effects=[]
    for d,g in cand.groupby('domain'):
        a=g.loc[g.label==1,'undirected_goal_proximity'].values; b=g.loc[g.label==0,'undirected_goal_proximity'].values
        u,p=mannwhitneyu(a,b,alternative='two-sided')
        effects.append({'Domain':d,'GT median':np.median(a),'Non-GT median':np.median(b),'Mann-Whitney U':u,'p_value':p,'rank_biserial':2*u/(len(a)*len(b))-1})
    pd.DataFrame(effects).to_csv(out_dir/'tables/geometry_effects.csv',index=False)
    # complexity stratification for SIG nonlinear top1
    comp=[]
    for d,(test,preds) in test_store.items():
        tt=test[['group','label','depth','choice_count']].copy(); tt['score']=preds['SIG-nonlinear']
        inst=tt.groupby('group').agg(depth=('depth','max'),choices=('choice_count','max'))
        chosen=tt.loc[tt.groupby('group')['score'].idxmax()].set_index('group')['label']
        inst['correct']=chosen
        if inst['depth'].nunique()>1:
            inst['quartile']=pd.qcut(inst['depth'].rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
        else: inst['quartile']='Q1'
        for q,z in inst.groupby('quartile',observed=False):
            comp.append({'Domain':d,'Depth quartile':str(q),'Top1 accuracy':z.correct.mean(),'Instances':len(z),'Median depth':z.depth.median()})
    pd.DataFrame(comp).to_csv(out_dir/'tables/complexity_results.csv',index=False)
    # theorem sanity: path length >= displacement for random trajectories in feature space
    rng=np.random.default_rng(seed); checks=[]
    X=cand[FEATURES].to_numpy(float)
    mu=X.mean(0); sd=X.std(0)+EPS; X=(X-mu)/sd
    for k in [3,5,8,13]:
        ratios=[]
        for _ in range(1000):
            ids=rng.choice(len(X),size=k,replace=False); pts=X[ids]
            L=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum(); D=np.linalg.norm(pts[-1]-pts[0]);
            if D>EPS: ratios.append(L/D)
        checks.append({'trajectory_nodes':k,'trials':len(ratios),'min_spiral_ratio':min(ratios),'median_spiral_ratio':np.median(ratios),'violations_below_1':int(np.sum(np.array(ratios)<1-1e-10))})
    pd.DataFrame(checks).to_csv(out_dir/'tables/theorem_sanity.csv',index=False)
    # Figures
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    piv=results.pivot(index='domain',columns='model',values='top1_accuracy').reindex(['GSM-Q','GSME-Q','Logic-Q','Planning-Q'])
    piv.plot(kind='bar',ax=ax); ax.set_ylabel('Top-1 sufficient-question accuracy'); ax.set_xlabel(''); ax.set_ylim(0,1); ax.legend(title='Model',fontsize=8); fig.tight_layout(); fig.savefig(out_dir/'figures/top1_accuracy.pdf'); fig.savefig(out_dir/'figures/top1_accuracy.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8));
    eff=pd.DataFrame(effects).set_index('Domain').reindex(['GSM-Q','GSME-Q','Logic-Q','Planning-Q'])
    eff[['GT median','Non-GT median']].plot(kind='bar',ax=ax); ax.set_ylabel('Median goal proximity'); ax.set_xlabel(''); ax.legend(); fig.tight_layout(); fig.savefig(out_dir/'figures/goal_proximity.pdf'); fig.savefig(out_dir/'figures/goal_proximity.png',dpi=220); plt.close(fig)
    cdf=pd.DataFrame(comp)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for d,z in cdf.groupby('Domain'):
        z=z.set_index('Depth quartile').reindex(['Q1','Q2','Q3','Q4']); ax.plot(z.index,z['Top1 accuracy'],marker='o',label=d)
    ax.set_ylabel('SIG nonlinear top-1 accuracy'); ax.set_xlabel('Problem-depth quartile'); ax.set_ylim(0,1); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_dir/'figures/complexity_curve.pdf'); fig.savefig(out_dir/'figures/complexity_curve.png',dpi=220); plt.close(fig)
    manifest={'seed':seed,'rows':len(cand),'instances':int(cand.group.nunique()),'domains':sorted(cand.domain.unique().tolist()),'features':FEATURES}
    (out_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return summary,results
