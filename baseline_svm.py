# -*- coding: utf-8 -*-
"""
批量读取 CWRU 的 NPZ（含 DE/FE/BA），解析标签与采样率，
对 DE 通道做窗口化（2048 点，50% 重叠），
提取简易特征并用 SVM 跑一个 baseline 分类。
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
from numpy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import hilbert
from scipy.stats import entropy

import re  # 这行你已导入也没问题，保留即可


def parse_fault(name: str) -> str:
    up = name.upper()
    # Normal
    if re.search(r'(?:^|_)NORMAL(?:_|\.|$)', up):
        return "NORMAL"
    # OR: OR、OR@3、OR3、OR_3
    if re.search(r'(?:^|_)(OR(?:@\d+)?|\bOR\d+\b)(?:_|\.|$)', up):
        return "OR"
    # IR
    if re.search(r'(?:^|_)IR(?:_|\.|$)', up):
        return "IR"
    # B（滚动体）
    if re.search(r'(?:^|_)B(?:_|\.|$)', up):
        return "B"
    return "UNKNOWN"


# === 配置区：把 ROOT 改成你的 NPZ 数据根目录（包含 1730 RPM/1750 RPM/...） ===
ROOT = Path(r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data")

# ---- 超参数 ----
CHANNEL = "DE"  # 先用驱动端
WIN = 2048  # 窗口长度
HOP = WIN // 2  # 50% 重叠
MAX_FILES_PER_CLASS = None  # 每类最多读多少个文件（None = 全部）


# 从文件路径解析 rpm/故障/测点，推断采样率
def parse_meta(p: Path):
    s = str(p).replace("\\", "/")
    m_rpm = re.search(r"/(\d+)\s*RPM/", s, re.IGNORECASE)
    rpm = int(m_rpm.group(1)) if m_rpm else None
    # ★ 用新的解析函数
    fault = parse_fault(p.name)
    # 测点/采样率：DE12/DE48/FE
    name_up = p.name.upper()
    if "DE48" in name_up:
        loc, fs = "DE48", 48000
    elif "DE12" in name_up:
        loc, fs = "DE12", 12000
    elif re.search(r'(?:^|_)FE(?:_|\.|$)', name_up):
        loc, fs = "FE", 12000  # 多数 FE 为 12k（你的数据如不同可改）
    else:
        loc, fs = "UNK", 12000
    return rpm, fault, loc, fs


# 简单时域/频域特征（可按需扩展）
from scipy.signal import hilbert
from scipy.stats import entropy
from numpy.fft import rfft, rfftfreq


def band_energy(X, f, f1, f2):
    """单边谱 X 在 [f1,f2) Hz 的能量（离散和）"""
    m = (f >= f1) & (f < f2)
    return float(X[m].sum())


def extract_feats(x: np.ndarray, fs: int, rpm: int | None = None):
    x = x.astype(float)
    N = len(x)

    # —— 时域 ——
    rms = float(np.sqrt(np.mean(x ** 2)))
    pp = float(np.max(x) - np.min(x))
    sk = float(stats.skew(x))
    kt = float(stats.kurtosis(x, fisher=True))
    crest = float(np.max(np.abs(x)) / (rms + 1e-12))

    # —— 频域（单边谱）——
    win = np.hanning(N)
    X = np.abs(rfft(x * win)) / N
    f = rfftfreq(N, 1 / fs)
    idx_peak = int(np.argmax(X))
    f_peak = float(f[idx_peak])
    spec_cent = float((f * X).sum() / (X.sum() + 1e-12))
    spec_bw = float(np.sqrt(((f - spec_cent) ** 2 * X).sum() / (X.sum() + 1e-12)))
    p = X / (X.sum() + 1e-12)
    spec_entropy = float(entropy(p + 1e-12))

    # —— 包络及其谱 ——
    env = np.abs(hilbert(x * win))
    env_rms = float(np.sqrt(np.mean(env ** 2)))
    env_kurt = float(stats.kurtosis(env, fisher=True))
    E = np.abs(rfft(env)) / N

    # 经验频带能量（12k/48k 都可用）
    E0_300 = band_energy(E, f, 0, 300)
    E300_600 = band_energy(E, f, 300, 600)
    E600_1200 = band_energy(E, f, 600, 1200)

    # 轴频附近能量（如果能解析到 rpm）
    fr = (rpm / 60.0) if rpm else 0.0

    def around(frq, bw=10):
        return band_energy(E, f, max(frq - bw, 0), frq + bw) if frq > 0 else 0.0

    E_fr = around(fr)
    E_2fr = around(2 * fr)
    E_3fr = around(3 * fr)

    return [
        rms, pp, sk, kt, crest,
        f_peak, spec_cent, spec_bw, spec_entropy,
        env_rms, env_kurt, E0_300, E300_600, E600_1200,
        E_fr, E_2fr, E_3fr
    ]


def windowed_segments(x: np.ndarray, win: int, hop: int):
    segs = []
    for start in range(0, len(x) - win + 1, hop):
        segs.append(x[start:start + win])
    return segs


# ---- 扫描数据并构建样本 ----
rows = []
counter = {}  # 按类别限量时使用
npz_files = sorted(ROOT.rglob("*.npz"))
if not npz_files:
    raise SystemExit(f"未在 {ROOT} 找到 .npz 文件")
FEAT_COLS = [
    "rms", "pp", "skew", "kurt", "crest",
    "f_peak", "spec_cent", "spec_bw", "spec_entropy",
    "env_rms", "env_kurt", "E0_300", "E300_600", "E600_1200",
    "E_fr", "E_2fr", "E_3fr"
]
for p in npz_files:
    rpm, fault, loc, fs = parse_meta(p)
    if fault not in ("NORMAL", "B", "IR", "OR"):
        continue
    # 每类最多读多少文件（降低训练时间）
    if MAX_FILES_PER_CLASS is not None:
        k = (fault, loc)
        counter[k] = counter.get(k, 0) + 1
        if counter[k] > MAX_FILES_PER_CLASS:
            continue

    d = np.load(p)
    if CHANNEL not in d.files:
        continue
    x = d[CHANNEL].squeeze()
    # 窗口化
    for seg in windowed_segments(x, WIN, HOP):
        feats = extract_feats(seg, fs, rpm)
        rows.append({
            "rpm": rpm, "fault": fault, "loc": loc, "fs": fs,
            "src": str(p),
            **dict(zip(FEAT_COLS, feats))
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("没有构造出样本，请检查文件命名与 CHANNEL 键（DE/FE/BA）")
print("样本数：", len(df), "  特征列：", [c for c in df.columns if c not in ("fault", "rpm", "loc", "fs")])
print(df["fault"].value_counts())

# ---- 训练/评估 ----
X = df[FEAT_COLS].values
y = df['fault'].values
groups = df['src'].values       # ← 这里按文件分组

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print("\n=== Confusion Matrix ===")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=sorted(set(y_test)),
                   columns=sorted(set(y_test))))
