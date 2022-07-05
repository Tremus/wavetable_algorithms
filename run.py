import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import List

N = 1 << 11

X_TICKS = [int(N / 4) * i for i in range(5)]
X_LABELS = [str(i) for i in X_TICKS]

Y_TICKS = [-1, -0.5, 0, 0.5, 1]
Y_LABELS = [str(i) for i in Y_TICKS]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7,2))


x = list(range(N))


def create_sine(nth_harmonic = 1) -> List[float]:
    # RMS 0.7071067811865476
    p = 1 / N * 2 * math.pi * nth_harmonic
    return [math.sin(i * p) for i in x]

def create_saw() -> List[float]:
    # RMS 0.5773504068406402
    return [-1 + (i / N * 2) for i in x]

def create_fft_add_nths(n_stride = 1, num_harmonics = 4) -> List[float]:
    """sounds good when n_stride = 2/3/4"""
    data = []
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    p = 1 / N * 2 * math.pi
    for i in range(N):
        v0 = i * p
        v1 = 0
        for idx, h in enumerate(harmonics):
            v1 += math.sin(v0 * h)
            # v1 += math.sin(v0 * h) / math.sqrt(2 + idx)
        data.append(v1)
    return data

def create_fft_add_nths_sqrt(n_stride = 1, num_harmonics = 4) -> List[float]:
    data = []
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    divisors = [1 / math.sqrt(1 + i) for i in range(num_harmonics)]
    p = 1 / N * 2 * math.pi
    for i in range(N):
        v0 = i * p
        v1 = 0
        for idx, h in enumerate(harmonics):
            v1 += math.sin(v0 * h) * divisors[idx]
        data.append(v1)
    return data

def create_lp_saw(n_stride = 1, num_harmonics = 4) -> List[float]:
    """
    Square when n_stride = 2
    Anything >= 4 will start to look like a harmonic stretch on a Saw wave
    """
    data = []
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    p = 1 / N * 2 * math.pi
    for i in range(N):
        v0 = i * p
        v1 = 0
        for h in harmonics:
            # nth harmonic is divided by n
            v1 += math.sin(v0 * h) / h
        data.append(v1)
    return data


# some serum WTs
create_fft_add_2nds = lambda : create_fft_add_nths(2)
create_fft_add_3rds = lambda : create_fft_add_nths(3)
create_fft_add_4ths = lambda : create_fft_add_nths(4)

y = np.array(create_sine(1))

# normalise the values to be between -1,1
y = y * (1 / max(np.max(y), abs(np.min(y))))

ax0.set_xticks(X_TICKS, X_LABELS)
ax0.set_xlim(0, N)
ax0.set_yticks(Y_TICKS, Y_LABELS)
ax0.set_ylim(-1, 1)

rms = np.sqrt(np.mean(np.array(y)**2))
ax0.set_title("RMS: {:.3f}".format(rms))

ax0.plot(x, y)
ax0.grid()

Y = np.fft.fft(y)
crop = 1 << 4
magnitudes = [math.hypot(c.real, c.imag) for c in Y[:crop]]
magnitudes = np.array(magnitudes) / (sum(magnitudes) / 100)
ax1.bar(x[:crop], magnitudes)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()
