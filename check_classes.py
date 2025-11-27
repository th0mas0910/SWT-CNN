from pathlib import Path
import re
import pandas as pd
import re
from pathlib import Path
import pandas as pd

# 贴同款 parse_fault（跟 baseline_svm.py 完全一致）
def parse_fault(name: str) -> str:
    up = name.upper()
    if re.search(r'(?:^|_)NORMAL(?:_|\.|$)', up): return "NORMAL"
    if re.search(r'(?:^|_)(OR(?:@\d+)?|\bOR\d+\b)(?:_|\.|$)', up): return "OR"
    if re.search(r'(?:^|_)IR(?:_|\.|$)', up): return "IR"
    if re.search(r'(?:^|_)B(?:_|\.|$)', up): return "B"
    return "UNKNOWN"

ROOT = Path(r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data")

rows = []
for p in ROOT.rglob("*.npz"):
    rows.append({"path": str(p), "fault": parse_fault(p.name)})

# 明确列名，避免空列表时没列
df = pd.DataFrame(rows, columns=["path", "fault"])

print(df["fault"].value_counts(dropna=False))
print("\nUNKNOWN 示例（前10）：")
print(df.loc[df["fault"] == "UNKNOWN", "path"].head(10).to_string(index=False))