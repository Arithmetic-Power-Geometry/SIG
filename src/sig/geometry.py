from __future__ import annotations
import math
import numpy as np
from scipy import sparse

EPS=1e-12

def cosine_distance(a,b):
    # inputs are row sparse vectors normalized by HashingVectorizer norm='l2'
    sim=float(a.multiply(b).sum())
    return float(np.clip(1.0-sim,0.0,2.0))

def trajectory_geometry(X):
    """Compute embedding-agnostic metric trajectory descriptors for ordered states."""
    n=X.shape[0]
    if n < 2:
        return dict(path_length=0.0, displacement=0.0, spiral_ratio=1.0,
                    max_radius=0.0, mean_radius=0.0, radial_growth=0.0,
                    radial_volatility=0.0, return_fraction=0.0,
                    local_excess_mean=0.0, local_excess_total=0.0,
                    turn_curvature=0.0, efficiency=1.0)
    step=np.array([cosine_distance(X[i],X[i+1]) for i in range(n-1)],dtype=float)
    radii=np.array([cosine_distance(X[0],X[i]) for i in range(n)],dtype=float)
    L=float(step.sum()); D=float(radii[-1])
    ratio=L/max(D,EPS) if D>EPS else (1.0 if L<=EPS else L/EPS)
    dr=np.diff(radii)
    returns=float(np.mean(dr< -1e-9)) if len(dr) else 0.0
    excess=[]
    for i in range(1,n-1):
        e=cosine_distance(X[i-1],X[i])+cosine_distance(X[i],X[i+1])-cosine_distance(X[i-1],X[i+1])
        excess.append(max(0.0,e))
    ex=np.array(excess or [0.0])
    return dict(
      path_length=L, displacement=D, spiral_ratio=float(ratio),
      max_radius=float(radii.max()), mean_radius=float(radii.mean()),
      radial_growth=float(radii[-1]-radii[0]),
      radial_volatility=float(np.std(dr)) if len(dr) else 0.0,
      return_fraction=returns,
      local_excess_mean=float(ex.mean()), local_excess_total=float(ex.sum()),
      turn_curvature=float(ex.sum()/max(L,EPS)), efficiency=float(D/max(L,EPS)) if L>EPS else 1.0)

def spiral_ratio_from_distances(step_distances, endpoint_distance):
    L=float(sum(step_distances)); D=float(endpoint_distance)
    if D<=EPS: return 1.0 if L<=EPS else L/EPS
    return L/D
