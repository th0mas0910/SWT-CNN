# check_files_and_labels.py
import os, glob, numpy as np
root = r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data\1797 RPM"   # 改成你跑的 root
Lmin, Lmax = 100_000, 150_000   # 12k 档大致阈值（如需放宽可改）

def has_DE_12k(path):
    try:
        with np.load(path, allow_pickle=True) as z:
            if 'DE' not in z.files:
                return False
            n = max(z['DE'].shape)
            return Lmin <= n <= Lmax
    except Exception as e:
        return False

files = glob.glob(os.path.join(root, "*.npz"))
files = sorted(files)
selected = [p for p in files if has_DE_12k(p)]
print(f"找到 NPZ 总数 {len(files)}, 满足 12k-DE 的 {len(selected)} 个")
# 根据文件名提取 label（假设文件名里包含 _B_ 或 _IR_ 或 _OR_ 或 _Normal）
from collections import Counter
cnt = Counter()
for p in selected:
    name = os.path.basename(p)
    # 尝试从名字里找到类标签
    import re


    def detect_label_from_name(name: str):
        """
        更健壮的从文件名猜类别：
        能匹配形式： _OR_, OR@, .OR., OR-, OR3 之类常见模式，
        以及 'Normal' / 'normal'。
        返回: 'B' / 'IR' / 'OR' / 'NORMAL' or None
        """
        n = name
        # 先看 Normal (完整词)
        if re.search(r'(?i)(?:^|[_@.\-])Normal(?:[_@.\-]|$)', n):
            return 'NORMAL'
        # IR, OR, B 三类（IR 要在 OR 之前匹配）
        if re.search(r'(?i)(?:^|[_@.\-])IR(?:[_@.\-]|$)', n):
            return 'IR'
        if re.search(r'(?i)(?:^|[_@.\-])OR(?:[_@.\-]|$)', n):
            return 'OR'
        if re.search(r'(?i)(?:^|[_@.\-])B(?:[_@.\-]|$)', n):
            return 'B'
        # 兜底：尝试更宽松的包含匹配（谨慎）
        low = n.lower()
        if 'ir' in low.split('_') or 'ir' in low.split('@'):
            return 'IR'
        if 'or' in low.split('_') or 'or' in low.split('@'):
            return 'OR'
        if 'b' in low.split('_') or 'b' in low.split('@'):
            return 'B'
        return None

print("按文件计数（满足 12k-DE）：", dict(cnt))
