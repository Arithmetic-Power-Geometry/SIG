from __future__ import annotations
import json,re,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import spearmanr, mannwhitneyu
from .geometry import trajectory_geometry, cosine_distance

VEC=HashingVectorizer(n_features=2**13, alternate_sign=False, analyzer='char_wb', ngram_range=(3,5), norm='l2', lowercase=True)
INTERROG=re.compile(r'(\?|？|\bwhat\b|\bwhich\b|\bwhy\b|\bhow\b|\bcould you\b|\bcan you\b|\bwould you\b|\bclarif|\bprovide\b|请问|能否|可以.*吗|什么|哪个|为何|为什么)',re.I)

def _read_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for ln,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{ln}: {e}')

def canonical_expected(s):
    s=str(s).strip()
    m=re.search(r'(?:answer\s*(?:is|:)?\s*)([A-E])\b',s,re.I)
    if m:return ('choice',m.group(1).upper())
    m=re.search(r'boxed\s*\{([^{}]+)\}',s,re.I)
    if m:return ('value',m.group(1).strip())
    # compact numeric/expression answers
    if len(s)<=40:return ('value',s)
    return ('phrase',s)

def final_correct(expected, final_text):
    kind,val=canonical_expected(expected); t=str(final_text)
    if not val:return 0
    if kind=='choice':
        patterns=[rf'\banswer\s*(?:is|:)?\s*{re.escape(val)}\b',rf'\boption\s*{re.escape(val)}\b',rf'\b{re.escape(val)}\s*[\.)]']
        return int(any(re.search(p,t,re.I) for p in patterns))
    if kind=='value':
        nv=re.sub(r'\s+','',val.lower()).replace('\\','')
        nt=re.sub(r'\s+','',t.lower()).replace('\\','')
        return int(nv in nt)
    return int(val.lower() in t.lower())

def extract_states(obj, setting):
    ch=obj.get('conversation_history') or []
    if not ch:return [],''
    initial=str(obj.get('degraded_question' if setting=='AskMind' else 'overconfidence_question') or ch[0].get('content',''))
    assistants=[(i,str(x.get('content',''))) for i,x in enumerate(ch) if x.get('role')=='assistant']
    final=assistants[-1][1] if assistants else ''
    qturns=[]
    for i,text in assistants[:-1]:
        if INTERROG.search(text): qturns.append(text)
    # if all intermediate assistant turns were non-question-like, use them as inquiry transformations but never the final answer
    if not qturns and len(assistants)>1:qturns=[t for _,t in assistants[:-1]]
    return [initial]+qturns, final

def build_trajectory_table(train_dir:Path):
    rows=[]; raw_counts=[]
    for fn,setting in [('mind.jsonl','AskMind'),('overconfidence.jsonl','AskOverconfidence')]:
        nraw=nconv=0
        for obj in _read_jsonl(train_dir/fn):
            nraw+=1
            if not obj.get('conversation_history'):continue
            nconv+=1
            states,final=extract_states(obj,setting)
            if not states:continue
            X=VEC.transform(states)
            g=trajectory_geometry(X)
            qs=states[1:]
            chars=sum(len(x) for x in qs); toks=sum(len(re.findall(r'\w+',x,flags=re.UNICODE)) for x in qs)
            points=obj.get('required_points' if setting=='AskMind' else 'misleading_points') or []
            row=dict(setting=setting,id=obj.get('id',''),n_states=len(states),n_questions=max(0,len(states)-1),
                     total_question_chars=chars,total_question_tokens=toks,mean_question_chars=chars/max(1,len(qs)),
                     rubric_points=len(points),final_correct=final_correct(obj.get('expected_answer',''),final),
                     expected_answer=str(obj.get('expected_answer','')),final_response_chars=len(final))
            row.update(g); rows.append(row)
        raw_counts.append({'setting':setting,'raw_rows':nraw,'usable_trajectories':nconv})
    return pd.DataFrame(rows),pd.DataFrame(raw_counts)

BASE=['n_questions','total_question_chars','total_question_tokens','mean_question_chars']
GEO=['path_length','displacement','spiral_ratio','max_radius','mean_radius','radial_growth','radial_volatility','return_fraction','local_excess_mean','local_excess_total','turn_curvature','efficiency']

def evaluate_trajectories(df,seed=2026):
    allres=[]
    for setting in ['AskMind','AskOverconfidence']:
        d=df[df.setting==setting].copy()
        tr,te=train_test_split(np.arange(len(d)),test_size=.25,random_state=seed,stratify=d.final_correct)
        train=d.iloc[tr]; test=d.iloc[te]
        specs=[('Turn-count',['n_questions'],'logit'),('Lexical-length',BASE,'logit'),('SIG-geometry',GEO,'logit'),('SIG+surface',GEO+BASE,'logit'),('SIG-nonlinear',GEO+BASE,'rf')]
        for name,fs,kind in specs:
            if kind=='rf': model=RandomForestClassifier(n_estimators=350,max_depth=9,min_samples_leaf=8,class_weight='balanced_subsample',random_state=seed,n_jobs=-1)
            else:model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed))])
            model.fit(train[fs],train.final_correct); p=model.predict_proba(test[fs])[:,1]; yhat=(p>=.5).astype(int); y=test.final_correct.values
            allres.append(dict(setting=setting,model=name,n_train=len(train),n_test=len(test),positive_rate=float(y.mean()),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),accuracy=accuracy_score(y,yhat),balanced_accuracy=balanced_accuracy_score(y,yhat),f1=f1_score(y,yhat)))
    return pd.DataFrame(allres)

def geometry_effects(df):
    rows=[]
    for setting in ['AskMind','AskOverconfidence']:
        d=df[df.setting==setting]
        for feat in ['path_length','displacement','spiral_ratio','turn_curvature','return_fraction','n_questions']:
            a=d[d.final_correct==1][feat].values; b=d[d.final_correct==0][feat].values
            stat,p=mannwhitneyu(a,b,alternative='two-sided')
            # rank-biserial-like effect 2*AUC-1
            eff=2*stat/(len(a)*len(b))-1
            rows.append(dict(setting=setting,feature=feat,mean_correct=float(np.mean(a)),mean_incorrect=float(np.mean(b)),effect_r=float(eff),p_value=float(p)))
    return pd.DataFrame(rows)

def build_eval_table(eval_dir:Path):
    rows=[]
    for path in sorted(eval_dir.glob('*.jsonl')):
        setting='AskOverconfidence' if 'overconfidence' in path.name else 'AskMind'
        for obj in _read_jsonl(path):
            original=str(obj.get('ori_question','')); modified=str(obj.get('overconfidence_question' if setting=='AskOverconfidence' else 'degraded_question',''))
            X=VEC.transform([original,modified]); d=cosine_distance(X[0],X[1])
            pts=obj.get('misleading_points' if setting=='AskOverconfidence' else 'required_points') or []
            rows.append({'file':path.name,'setting':setting,'id':obj.get('id',''),'edit_displacement':d,'rubric_points':len(pts)})
    return pd.DataFrame(rows)

def eval_correlations(df):
    rows=[]
    for f,d in df.groupby('file'):
        if len(d)>2 and d.rubric_points.nunique()>1:
            r,p=spearmanr(d.edit_displacement,d.rubric_points)
        else:r,p=np.nan,np.nan
        rows.append({'file':f,'n':len(d),'mean_displacement':d.edit_displacement.mean(),'mean_rubric_points':d.rubric_points.mean(),'spearman_rho':r,'p_value':p})
    return pd.DataFrame(rows)
