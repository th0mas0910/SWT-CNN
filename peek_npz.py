import numpy as np
from pathlib import Path

p = Path(r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data\1730 RPM")  # 改成你的路径
f = next(p.glob("*.npz"))  # 取目录里第一个文件
data = np.load(f)

print("文件：", f.name)
print("可用键：", list(data.keys()))
for k in data.files:
    arr = data[k]
    try:
        shape = arr.shape
        dtype = arr.dtype
    except Exception:
        shape, dtype = "N/A", type(arr)
    print(f"  {k}: shape={shape}, dtype={dtype}")
