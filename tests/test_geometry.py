import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from sig.geometry import spiral_ratio_from_distances,trajectory_geometry
from sklearn.feature_extraction.text import HashingVectorizer

def test_spiral_lower_bound_for_metric_path():
    assert spiral_ratio_from_distances([1,1],1.5) >= 1

def test_straight_single_step_ratio_one():
    assert abs(spiral_ratio_from_distances([0.7],0.7)-1)<1e-12

def test_text_trajectory_finite():
    v=HashingVectorizer(n_features=256,alternate_sign=False,norm='l2')
    X=v.transform(['initial problem','what is missing?','why is it missing?'])
    g=trajectory_geometry(X)
    assert g['path_length'] >= g['displacement']-1e-10
    assert g['spiral_ratio'] >= 1-1e-10
