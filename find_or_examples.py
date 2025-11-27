from pathlib import Path
import re

ROOT = Path(r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data")

pat_or = re.compile(r'(?:^|_)(OR(?:@\d+)?|\bOR\d+\b)(?:_|\.|$)', re.IGNORECASE)

hits = []
unknowns = []
for p in ROOT.rglob("*.npz"):
    name = p.name
    if pat_or.search(name):
        hits.append(name)
    elif "OR" in name.upper():   # 粗筛：含有 OR 但没被上面的正则命中
        unknowns.append(name)

print("识别到 OR 的样例（前10）：")
for x in hits[:10]: print("  ", x)
print("\n含 OR 但未命中的样例（前10）：")
for x in unknowns[:10]: print("  ", x)
print(f"\n总计：匹配到 {len(hits)} 个，含 OR 未命中 {len(unknowns)} 个")
