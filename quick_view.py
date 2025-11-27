import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert
from numpy.fft import rfft, rfftfreq

file = Path(r"C:\Users\windows\Desktop\CWRU_Bearing_NumPy-main\Data\1730 RPM\1730_B_14_DE12.npz")  # 改成你的路径
d = np.load(file)

# 1) 取驱动端通道，并变成一维
x = d["DE"].squeeze()         # (122136,)
fs = 12000                    # DE12 → 12kHz；DE48 → 48000；FE 通常也是 12k（按你的数据而定）

# 2) 画前 2000 点波形
n = min(2000, x.size)
plt.figure()
plt.plot(x[:n])
plt.title(f"DE waveform (first {n} samples)")
plt.xlabel("Sample"); plt.ylabel("Amplitude")
plt.tight_layout()

# 3) 基本振幅谱（幅值-频率）
N = 32768  # 取一个2的幂长度做FFT（可调）
X = np.abs(rfft(x[:N] * np.hanning(N))) / N
f = rfftfreq(N, d=1/fs)
plt.figure()
plt.plot(f, X)
plt.xlim(0, fs/2)
plt.title("Amplitude Spectrum (DE)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.tight_layout()

# 4) 包络谱（常用于轴承故障特征）
analytic = hilbert(x[:N] * np.hanning(N))
env = np.abs(analytic)
E = np.abs(rfft(env)) / N
plt.figure()
plt.plot(f, E)
plt.xlim(0, fs/2)
plt.title("Envelope Spectrum (DE)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
