from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from sig.representation_robustness import run
if __name__=='__main__':
    run(ROOT/'data/askbench/train',ROOT/'results',seed=2026)
    print('Representation robustness completed successfully.')
