# list_unknown_npz.py
import os, glob, numpy as np
root = r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data\1797 RPM"  # 改为你当前 root
Lmin, Lmax = 100_000, 150_000

def has_DE_12k(path):
    try:
        with np.load(path, allow_pickle=True) as z:
            if 'DE' not in z.files:
                return False
            n = max(z['DE'].shape)
            return Lmin <= n <= Lmax
    except Exception as e:
        return False

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


files = sorted(glob.glob(os.path.join(root, "*.npz")))
selected = [p for p in files if has_DE_12k(p)]
unknown = []
for p in selected:
    name = os.path.basename(p)
    lbl = detect_label_from_name(name)
    if lbl is None:
        unknown.append(p)
print("满足 12k-DE 的共:", len(selected))
print("被判为 UNKNOWN 的数量:", len(unknown))
for p in unknown:
    with np.load(p, allow_pickle=True) as z:
        keys = list(z.files)
        de_len = max(z['DE'].shape) if 'DE' in z.files else None
    print(os.path.basename(p), "-> keys:", keys, "DE_len:", de_len)
