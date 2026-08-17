import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sig.representation_robustness import _geometry,_random_embeddings,bh_fdr

def test_random_embeddings_deterministic_and_normalized():
    a=_random_embeddings(['alpha','beta','alpha'],dim=32,seed=2026)
    b=_random_embeddings(['alpha','beta','alpha'],dim=32,seed=2026)
    assert np.allclose(a,b)
    assert np.allclose(a[0],a[2])
    assert np.allclose(np.linalg.norm(a,axis=1),1.0,atol=1e-6)

def test_geometry_path_bound_dense():
    X=np.array([[1.,0.],[0.,1.],[-1.,0.]],dtype=float)
    X=X/np.linalg.norm(X,axis=1,keepdims=True)
    g=_geometry(X,0,3,'cosine')
    assert g['path_length']+1e-12 >= g['displacement']
    assert g['spiral_ratio'] >= 1.0

def test_bh_fdr_monotone_valid():
    q=bh_fdr([.001,.02,.5])
    assert np.all((q>=0)&(q<=1))
    assert q[0] <= q[1] <= q[2]
