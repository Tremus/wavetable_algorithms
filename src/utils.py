import math
import numpy as np
import numpy.typing as npt
from .config import N

NDDoubleArr = npt.NDArray[np.float64]
NDCDoubleArr = npt.NDArray[np.complex128]

def skew(x: float, a: float) -> float:
    if a >= 1: return 1
    if a <= 0: return 0
    b = a * x;
    return b / (1 - a - x + b + b)

def lininterp(amt, start, end):
    return start + amt * (end - start)

def get_magnitude(real: float, imag: float) -> float:
    return math.hypot(real, imag)

def get_phase(real: float, imag: float) -> float:
    return math.atan2(imag, real)

def polar_to_rectangular(mag: float, phase: float) -> complex:
    re = mag * math.cos(phase)
    im = mag * math.sin(phase)
    return complex(re, im)

def np_polar_to_rectangular(arr: NDCDoubleArr) -> NDCDoubleArr:
    mags = np.real(arr)
    phases = np.imag(arr)
    re = mags * np.cos(phases)
    im = mags * np.sin(phases)
    return re + im * 1j

def normalise_0dB(x: npt.ArrayLike):
    # normalise to be between -1,1
    return x * (1 / max(np.max(x), abs(np.min(x))))

def window_rms(x):
    x2 = np.power(x,2)
    window = np.ones(N)/float(N)
    return np.sqrt(np.convolve(x2, window, 'valid'))[0]
