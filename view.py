# Plotting cool some shapes with Trémus

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.widgets import Slider
from src.methods import *
from src.config import N

create_wave_function = lambda wt_pos: create_exp(wt_pos)

X_TICKS = [int(N / 4) * i for i in range(5)]
X_LABELS = [str(i) for i in X_TICKS]

Y_TICKS = [-1, -0.5, 0, 0.5, 1]
Y_LABELS = [str(i) for i in Y_TICKS]
LIMIT = 1 << 6

x = np.arange(N)


# The parametrized function to be plotted
def get_y(wt_pos: float):
    wave = create_wave_function(wt_pos)
    wave = normalise_0dB(wave)

    # print('RMS', window_rms(wave))

    return wave

def get_magnitudes(y: npt.ArrayLike) -> NDDoubleArr:
    Y: NDCDoubleArr = np.fft.fft(y)
    Y = Y[:LIMIT]
    # get magnitude, hypot of a complex number
    magnitudes: NDDoubleArr = np.hypot(np.real(Y), np.imag(Y))
    return magnitudes

# Create the figure and the line that we will manipulate
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
y = get_y(0)
line0, = ax0.plot(x, y)

bar_container = ax1.bar(x[:LIMIT], get_magnitudes(y), width=0.9)
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
    mags = get_magnitudes(y)
    for rect, mag in zip(bar_container, mags):
        rect.set_height(mag)
    fig.canvas.draw_idle()


# register the update function with each slider
wt_slider.on_changed(update)

plt.show()
