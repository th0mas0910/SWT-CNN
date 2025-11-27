# diag_scan_npz.py
import os, json
import numpy as np

ROOT = r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data"  # 改成你的 data 根目录
OUT = "diag_npz_report.json"

summary = []
for subdir, dirs, files in os.walk(ROOT):
    for fn in files:
        if not fn.lower().endswith(".npz"):
            continue
        p = os.path.join(subdir, fn)
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = list(z.files)
                info = {"path": p, "keys": keys}
                # 尝试记录常见通道长度（DE48/DE12/FE等）第一项长度
                for k in keys:
                    try:
                        arr = np.asarray(z[k])
                        info[f"{k}_shape"] = arr.shape
                        # if 1D, record length, if 2D, record (C,L)
                    except Exception as e:
                        info[f"{k}_shape"] = f"ERR:{e}"
                summary.append(info)
        except Exception as e:
            summary.append({"path": p, "error": str(e)})

# 写 json 报告
with open(OUT, "w", encoding="utf8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"scan complete, {len(summary)} files inspected. Report -> {OUT}")
# also print a short console sample
for item in summary[:20]:
    print(item["path"])
    print("  keys:", item.get("keys"))
    for k in item.get("keys", [])[:8]:
        print("   ", k, "shape:", item.get(f"{k}_shape"))
    print("---")
