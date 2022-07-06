# Plotting cool some shapes with Trémus

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.widgets import Slider
from typing import List

N = 1 << 11

X_TICKS = [int(N / 4) * i for i in range(5)]
X_LABELS = [str(i) for i in X_TICKS]

Y_TICKS = [-1, -0.5, 0, 0.5, 1]
Y_LABELS = [str(i) for i in Y_TICKS]

x = np.arange(N)


def create_sine(freq) -> np.ndarray:
    # nth_harmonic = 1 + wt_pos * 15
    p = (2 * np.pi * freq) / N
    return np.sin(x * p)

def create_saw(wt_pos) -> np.ndarray:
    # return [-1 + (i / N * 2) for i in x]
    return (x / N * 2) - 1

def create_fft_add_nths(wt_pos, num_harmonics = 4) -> List[float]:
    """sounds good when n_stride = 2/3/4"""
    n_stride = round(1 + wt_pos * 3)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    data = np.zeros(N)
    for h in harmonics:
        s = create_sine(h)
        data = data + s
    return data

def create_fft_add_nths_sqrt(wt_pos, num_harmonics = 4) -> List[float]:
    n_stride = round(1 + wt_pos * 7)
    print(n_stride)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    amplitudes = [1 / math.sqrt(1 + i) for i in range(num_harmonics)]
    data = np.zeros(N)
    for h, amp in zip(harmonics, amplitudes):
        s = create_sine(h) * amp
        data = data + s
    return data

def create_lp_saw(wt_pos, num_harmonics = 8) -> List[float]:
    """
    Square when n_stride = 2
    Anything >= 4 will start to look like a harmonic stretch on a Saw wave
    Starts to get slow after 300 or so harmonics and an IFFT may be faster
    """
    n_stride = round(1 + wt_pos * 7)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    data = np.zeros(N)
    for h in harmonics:
        s = create_sine(h) / h
        data = data + s
    return data


# The parametrized function to be plotted
def get_y(wt_pos):
    y = create_lp_saw(wt_pos, 8)
    # normalise to be between -1,1
    y = y * (1 / max(np.max(y), abs(np.min(y))))
    return y

def get_magnitudes(y, limit: int):
    Y = np.fft.fft(y)
    Y = Y[:limit]
    # get magnitude, hypot of a complex number
    magnitudes = np.hypot(np.real(Y), np.imag(Y))
    return magnitudes

# Create the figure and the line that we will manipulate
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
y = get_y(0)
line0, = ax0.plot(x, y)

limit = 1 << 4
bar_container = ax1.bar(x[:limit], get_magnitudes(y, limit))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=N, decimals=0))

ax0.set_xticks(X_TICKS, X_LABELS)
ax0.set_xlim(0, N)
ax0.set_yticks(Y_TICKS, Y_LABELS)
ax0.set_ylim(-1, 1)
ax0.grid()

ax1.set_ylim(0, N)


# adjust the main plot to make room for the sliders
plt.subplots_adjust(bottom=0.25)

# Make a horizontal slider to control the frequency.
wt_slider = Slider(
    ax=plt.axes([0.25, 0.1, 0.65, 0.03]),
    label='Wavetable',
    valmin=0,
    valmax=1,
    valinit=0)

# The function to be called anytime a slider's value changes
def update(val):
    y = get_y(val)
    line0.set_ydata(y)
    mags = get_magnitudes(y, limit)
    for rect, mag in zip(bar_container, mags):
        rect.set_height(mag)
    fig.canvas.draw_idle()


# register the update function with each slider
wt_slider.on_changed(update)


plt.show()