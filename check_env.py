import sys, numpy as np, pandas as pd
import scipy, h5py, pywt, sklearn
import matplotlib
from tqdm import tqdm

print("Python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("pywavelets:", pywt.__version__)
print("h5py:", h5py.__version__)
for _ in tqdm(range(10)): pass
print("环境检查 OK")
