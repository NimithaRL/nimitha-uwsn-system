import numpy as np, yaml
def set_seed(seed): np.random.seed(seed)
def load_config(p):
    with open(p) as f: return yaml.safe_load(f)
