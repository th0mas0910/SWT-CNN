import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ===================== 工具函数：解析文件名标签 =====================

import re


def detect_label_from_name(name: str):
    """
    更稳健的文件名解析：
    能识别形如：
      1797_B_7_DE12.npz
      1797_IR_7_DE12.npz
      1797_OR@12_21_DE12.npz
      1797_Normal.npz
    返回: 'B' / 'IR' / 'OR' / 'NORMAL' 或 None
    """
    n = name

    # Normal
    if re.search(r'(?i)(?:^|[_@.\-])Normal(?:[_@.\-]|$)', n):
        return 'NORMAL'

    # IR, OR, B
    if re.search(r'(?i)(?:^|[_@.\-])IR(?:[_@.\-]|$)', n):
        return 'IR'
    if re.search(r'(?i)(?:^|[_@.\-])OR(?:[_@.\-]|$)', n):
        return 'OR'
    if re.search(r'(?i)(?:^|[_@.\-])B(?:[_@.\-]|$)', n):
        return 'B'

    # 兜底：再宽松一点
    low = n.lower()
    if 'normal' in low:
        return 'NORMAL'
    if 'ir' in low.split('_') or 'ir' in low.split('@'):
        return 'IR'
    if 'or' in low.split('_') or 'or' in low.split('@'):
        return 'OR'
    if 'b' in low.split('_') or 'b' in low.split('@'):
        return 'B'
    return None


# ===================== 数据结构与加载 =====================

@dataclass
class SignalFile:
    path: str
    label: str  # 'B' / 'IR' / 'OR' / 'NORMAL'

def is_de12_file_for_now(name: str) -> bool:
    """
    当前阶段：只保留 12k DE 文件
    规则：
      - 含有 '_DE12' 的（B / IR / OR 等故障）
      - 或者是 Normal.npz
    """
    low = name.lower()
    if "normal" in low:
        return True
    if "_de12" in low:
        return True
    return False

def scan_npz_files(root: str,
                   class_set: List[str],
                   channels: str) -> List[SignalFile]:
    """
    扫描 root 目录下的所有 npz 文件，根据文件名解析标签，
    并过滤出包含所需通道的文件。
    """
    files: List[SignalFile] = []
    root = os.path.abspath(root)

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".npz"):
                continue

            # ★★ 在这里加一行，只保留 DE12 + Normal ★★
            if not is_de12_file_for_now(fn):
                continue

            path = os.path.join(dirpath, fn)
            lbl = detect_label_from_name(fn)
            if lbl is None or lbl not in class_set:
                continue

            # 检查是否真有我们要的通道
            try:
                with np.load(path, allow_pickle=True) as z:
                    z_keys = list(z.files)
                    need_ok = True
                    if channels == "DE" and ("DE" not in z_keys):
                        need_ok = False
                    elif channels == "FE" and ("FE" not in z_keys):
                        need_ok = False
                    elif channels == "BA" and ("BA" not in z_keys):
                        need_ok = False
                    elif channels == "multi":
                        # 至少要有一个通道
                        if not any(k in z_keys for k in ("DE", "FE", "BA")):
                            need_ok = False
                if not need_ok:
                    continue
            except Exception as e:
                print(f"[WARN] 读取 {path} 失败: {e}")
                continue

            files.append(SignalFile(path=path, label=lbl))

    print(f"在 {root} 下找到满足条件的 NPZ 文件数量: {len(files)}")
    return files



def load_npz_channels(path: str, channels: str) -> np.ndarray:
    """
    读取单个 npz 文件，按指定通道返回 [C, L] 数组 (float32).
    channels: 'DE' / 'FE' / 'BA' / 'multi'
    """
    with np.load(path, allow_pickle=True) as z:
        arrs = []
        if channels == "DE":
            arrs.append(z["DE"].astype(np.float32).reshape(-1))
        elif channels == "FE":
            arrs.append(z["FE"].astype(np.float32).reshape(-1))
        elif channels == "BA":
            arrs.append(z["BA"].astype(np.float32).reshape(-1))
        elif channels == "multi":
            for key in ("DE", "FE", "BA"):
                if key in z.files:
                    arrs.append(z[key].astype(np.float32).reshape(-1))
        else:
            raise ValueError(f"未知通道: {channels}")
    if len(arrs) == 0:
        raise ValueError(f"{path} 未找到任何通道")
    # pad 到相同长度（理论上 CWRU 同一个文件各键长度相同；保险起见取最短）
    min_len = min(a.shape[0] for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.stack(arrs, axis=0)  # [C, L]


# ===================== 统计均值 / 方差（按通道） =====================

def compute_mean_std(train_files: List[SignalFile],
                     channels: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    在所有训练文件上，按通道统计 mean / std。
    返回形状 [C] 的 mean, std。
    """
    sum_c = None
    sumsq_c = None
    count = 0
    for sf in train_files:
        x = load_npz_channels(sf.path, channels)  # [C, L]
        x = x.astype(np.float64)
        C, L = x.shape
        if sum_c is None:
            sum_c = np.zeros(C, dtype=np.float64)
            sumsq_c = np.zeros(C, dtype=np.float64)
        sum_c += x.sum(axis=1)
        sumsq_c += np.square(x).sum(axis=1)
        count += L
    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean ** 2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# ===================== SWT 前端（denoise / stack） =====================

def apply_swt_denoise(x: np.ndarray,
                      wavelet: str,
                      level: int) -> np.ndarray:
    """
    x: [C, L]，对每个通道进行 SWT 去噪，输出同形状 [C, L]
    """
    C, L = x.shape
    out = []
    for ci in range(C):
        sig = x[ci]
        coeffs = pywt.swt(sig, wavelet, level, trim_approx=False)
        new_coeffs = []
        for cA, cD in coeffs:
            # 估计噪声水平（简单 median 方法）
            sigma = np.median(np.abs(cD)) / 0.6745
            uth = sigma * math.sqrt(2 * math.log(L))
            cA_thr = pywt.threshold(cA, uth, mode='soft')
            cD_thr = pywt.threshold(cD, uth, mode='soft')
            new_coeffs.append((cA_thr, cD_thr))
        rec = pywt.iswt(new_coeffs, wavelet)
        out.append(rec.astype(np.float32))
    return np.stack(out, axis=0)
def apply_swt_stack(x: np.ndarray,
                    wavelet: str,
                    level: int) -> np.ndarray:
    """
    x: [C, L] -> 将所有尺度的 (cA, cD) 系数按通道堆叠，输出 [C*(2*level), L]
    """
    C, L = x.shape
    chans = []
    for ci in range(C):
        sig = x[ci]
        coeffs = pywt.swt(sig, wavelet, level, trim_approx=False)
        for cA, cD in coeffs:
            chans.append(cA.astype(np.float32))
            chans.append(cD.astype(np.float32))
    return np.stack(chans, axis=0)  # [C*2*level, L]


def apply_swt_denoise(x: np.ndarray,
                      wavelet: str,
                      level: int,
                      shrink: float = 0.5) -> np.ndarray:
    """
    x: [C, L]
    更温和的 SWT 去噪：
      - 只对细节系数 cD 做软阈值，cA 保留
      - 阈值 = shrink * 通用阈值（shrink < 1 更温柔）
    """
    C, L = x.shape
    out = []
    for ci in range(C):
        sig = x[ci]
        coeffs = pywt.swt(sig, wavelet, level=level, trim_approx=False)
        new_coeffs = []
        for cA, cD in coeffs:
            # 用第一层细节的 median 估计噪声，而不是每层都重新估计
            sigma = np.median(np.abs(cD)) / 0.6745 + 1e-8
            uth = shrink * sigma * math.sqrt(2 * math.log(L))

            cA_thr = cA      # 低频近似直接保留
            cD_thr = pywt.threshold(cD, uth, mode='soft')
            new_coeffs.append((cA_thr, cD_thr))

        rec = pywt.iswt(new_coeffs, wavelet)
        out.append(rec.astype(np.float32))
    return np.stack(out, axis=0)



# ===================== Dataset：窗口级训练 =====================

def apply_swt_stack(x, swt_wavelet, swt_level):
    pass


class SignalWindowDataset(Dataset):
    def __init__(self,
                 files: List[SignalFile],
                 label2idx: Dict[str, int],
                 channels: str,
                 win: int,
                 hop: int,
                 swt: str = "off",
                 swt_wavelet: str = "db4",
                 swt_level: int = 3,
                 mean: np.ndarray = None,
                 std: np.ndarray = None):
        """
        参数：
          files: 文件列表（train 或 val）
          win, hop: 滑窗大小与步长
          swt: 'off' / 'denoise' / 'stack'
          mean, std: 通道均值 / 方差（来自训练集）
        """
        self.files = files
        self.label2idx = label2idx
        self.channels = channels
        self.win = win
        self.hop = hop
        self.swt = swt
        self.swt_wavelet = swt_wavelet
        self.swt_level = swt_level
        self.mean = mean
        self.std = std

        # 先把所有文件读入到内存（CWRU 规模够用）
        self.file_signals: List[np.ndarray] = []  # [C,L]
        self.file_labels: List[int] = []
        self.indexer: List[Tuple[int, int]] = []  # (file_idx, start)

        for fi, sf in enumerate(self.files):
            sig = load_npz_channels(sf.path, self.channels)  # [C, L]
            C, L = sig.shape
            # 生成窗口索引
            for st in range(0, L - self.win + 1, self.hop):
                self.indexer.append((fi, st))
            self.file_signals.append(sig)
            self.file_labels.append(self.label2idx[sf.label])

        print(f"[SignalWindowDataset] 总窗口数 = {len(self.indexer)}")

    def __len__(self):
        return len(self.indexer)

    def __getitem__(self, idx):
        fi, st = self.indexer[idx]
        sig = self.file_signals[fi]  # [C, L]
        x = sig[:, st:st + self.win]  # [C, win]
        # SWT 前端
        if self.swt == "denoise":
            x = apply_swt_denoise(x, self.swt_wavelet, self.swt_level)
        elif self.swt == "stack":
            x = apply_swt_stack(x, self.swt_wavelet, self.swt_level)
        # 标准化
        if self.mean is not None and self.std is not None:
            # mean, std 形状 [C0]，需要与当前通道数匹配 / 或广播
            m = np.asarray(self.mean, dtype=np.float32)
            s = np.asarray(self.std, dtype=np.float32)
            if m.ndim == 1:
                m = m.reshape(-1, 1)
                s = s.reshape(-1, 1)
            # 若 swt=stack 导致通道数变多，可简单广播
            if m.shape[0] != x.shape[0]:
                # 通道不一致时，使用标量均值 / 方差（不太严谨，但先简单处理）
                ms = float(m.mean())
                ss = float(s.mean())
                x = (x - ms) / (ss + 1e-8)
            else:
                x = (x - m) / (s + 1e-8)
        y = self.file_labels[fi]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ===================== 简单 1D-CNN 模型 =====================

class CNN1D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, C, L]
        x = self.conv(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


# ===================== 训练与验证 =====================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total += xb.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_loss / total, total_correct / total, all_labels, all_preds


def plot_confusion(cm, class_names, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ===================== 主流程 =====================
from collections import defaultdict  # 如果上面没 import，就在文件头部加这一行

def stratified_split_files(all_files, val_ratio: float, seed: int):
    """
    按类别分层划分 train / val：
    - 每个类别单独打乱、单独按比例划分
    - 至少保留 1 个样本到验证集（当该类文件数 > 1 时）
    """
    rnd = random.Random(seed)

    # 1) 按 label 分组
    label2files = defaultdict(list)
    for f in all_files:
        label2files[f.label].append(f)

    train_files, val_files = [], []

    # 2) 每一类单独划分
    for lbl, files in label2files.items():
        rnd.shuffle(files)  # 先在本类内部打乱
        if len(files) > 1:
            n_val = max(1, int(len(files) * val_ratio))
        else:
            n_val = 0  # 只有 1 个文件就全放训练集，避免 val 里只有 1 个样本太不稳定

        if n_val > 0:
            val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    # 3) 最后整体再打乱一次（可选）
    rnd.shuffle(train_files)
    rnd.shuffle(val_files)

    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="SWT+1D-CNN 基线（窗口级训练）")
    parser.add_argument("--root", type=str, required=True,
                        help="数据集根目录（可直接指向 1797 RPM 等子目录）")
    parser.add_argument("--channels", type=str, default="DE",
                        choices=["DE", "FE", "BA", "multi"])
    parser.add_argument("--win", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=1024)
    parser.add_argument("--class_set", type=str, default="B,IR,OR,NORMAL")
    parser.add_argument("--task", type=str, default="intra",
                        choices=["intra"], help="先只做同工况 intra")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--swt", type=str, default="off",
                        choices=["off", "denoise", "stack"])
    parser.add_argument("--swt_wavelet", type=str, default="db4")
    parser.add_argument("--swt_level", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from collections import Counter

    def show_label_distribution(ds, idx2label):
        c = Counter()
        for _, y in ds:
            c[int(y)] += 1
        nice = {idx2label[i]: n for i, n in c.items()}
        return nice

    # 类别集合
    class_set = [c.strip() for c in args.class_set.split(",") if c.strip()]
    print("类别集合:", class_set)

    # 扫描文件
    all_files = scan_npz_files(args.root, class_set, args.channels)
    assert len(all_files) > 0, f"在 {args.root} 未找到符合条件的 NPZ 文件"
    print(f"总文件数 = {len(all_files)}")

    # 按 label 统计
    from collections import Counter
    cnt = Counter(sf.label for sf in all_files)
    print("按文件计数:", dict(cnt))

    # train_files, val_files = stratified_split_files(
    #     all_files,
    #     val_ratio=args.val_ratio,
    #     seed=args.seed,
    # )
    # print(f"训练文件数 = {len(train_files)}, 验证文件数 = {len(val_files)}")

    # label 编码
    uniq_labels = sorted(set(sf.label for sf in all_files),
                         key=lambda x: class_set.index(x))
    label2idx = {lb: i for i, lb in enumerate(uniq_labels)}
    idx2label = {i: lb for lb, i in label2idx.items()}
    print("label2idx:", label2idx)

    # 统计 train 文件通道均值 / 方差
    mean, std = compute_mean_std(all_files, args.channels)
    print("均值：", mean, " 方差：", std)

    # 保存配置与归一化参数
    with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "label_encoder.json"), "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "normalization.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()},
                  f, indent=2, ensure_ascii=False)

    # 构造 Dataset / DataLoader（窗口级）
    from torch.utils.data import random_split

    # 先构造“全量窗口”的 Dataset
    full_ds = SignalWindowDataset(
        all_files,
        label2idx,
        args.channels,
        args.win,
        args.hop,
        swt=args.swt,
        swt_wavelet=args.swt_wavelet,
        swt_level=args.swt_level,
        mean=mean,
        std=std,
    )
    print(f"[SignalWindowDataset] 窗口总数 = {len(full_ds)}")

    # 按窗口随机划分 train / val
    total = len(full_ds)
    n_val = int(total * args.val_ratio)
    n_train = total - n_val

    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    print(f"train 窗口数 = {len(train_ds)}, val 窗口数 = {len(val_ds)}")
    print("train 窗口按类别计数：", show_label_distribution(train_ds, idx2label))
    print("val   窗口按类别计数：", show_label_distribution(val_ds, idx2label))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始训练：设备={device}, AMP={args.amp}, SWT={args.swt}")

    # 模型
    # 估计输入通道数：用一个样本看一下
    xb0, _ = train_ds[0]
    in_ch = xb0.shape[0]
    model = CNN1D(in_ch, num_classes=len(label2idx)).to(device)

    # 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # 训练循环
    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, _, _ = eval_model(
            model, val_loader, criterion, device
        )
        print(f"Epoch {epoch:03d}/{args.epochs:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "in_ch": in_ch,
                "label2idx": label2idx,
                "idx2label": idx2label,
                "args": vars(args),
                "mean": mean,
                "std": std,
            }
            torch.save(best_state, os.path.join(args.save_dir, "best.pt"))
            print(f"  ↑ 新最佳，保存到 {os.path.join(args.save_dir, 'best.pt')}")

    print(f"训练完成。最佳验证准确率 = {best_acc:.4f}")

    # 使用最佳模型重新计算验证集指标
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    val_loss, val_acc, y_true, y_pred = eval_model(
        model, val_loader, criterion, device
    )

    # 分类报告
    labels = list(range(len(idx2label)))  # [0,1,2,3,...]
    target_names = [idx2label[i] for i in labels]  # ['B','IR','OR','NORMAL',...]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,  # 👈 指定标签顺序，避免“2 个类 vs 4 个名字”的报错
        target_names=target_names,
        digits=2,
        zero_division=0,  # 没预测到的类就按 0 处理，不再警告
    )
    with open(os.path.join(args.save_dir, "val_report_final.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("验证集分类报告：")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # 👈 同样指定 labels
    plot_confusion(
        cm,
        target_names,
        os.path.join(args.save_dir, "val_confusion_final.png")
    )

    print(f"结果已保存到：{os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
