# count_de48_fe.py
import os, json
ROOT = r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data"  # 改成你的 root
pairs = []
for subdir, _, files in os.walk(ROOT):
    # collect basenames in folder
    names = set(files)
    # find all DE48 files
    for fn in files:
        if fn.endswith("DE48.npz"):
            prefix = fn.replace("_DE48.npz", "")
            cand1 = prefix + "FE.npz"
            cand2 = prefix + "_FE.npz"
            if cand1 in names or cand2 in names:
                pairs.append(os.path.join(subdir, fn))
print("Found DE48+FE pairs:", len(pairs))
for p in pairs[:40]:
    print(" ", p)
