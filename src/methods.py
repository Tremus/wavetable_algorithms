import numpy as np
import math
from src.config import N
from src.utils import *

t = np.arange(0, 1, 1 / N)

def create_sine(freq):
    p = (2 * np.pi * freq)
    return np.sin(t * p)

def create_saw(wt_pos):
    return np.arange(-1, 1, 2 / N)

def sine_add(wt_pos):
    num_harmonics = wt_pos * 15
    data = create_sine(1)
    for Hz in range(2, 1 + int(num_harmonics)):
        data = data + create_sine(Hz)
    if num_harmonics > int(num_harmonics):
        amp = num_harmonics - int(num_harmonics)
        next_harmonic = 1 + int(num_harmonics)
        data = data + amp * create_sine(next_harmonic)
    return data

def create_fft_add_nths(wt_pos, num_harmonics = 4):
    """sounds good when n_stride = 2/3/4"""
    n_stride = round(1 + wt_pos * 3)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    data = np.zeros(N)
    for h in harmonics:
        s = create_sine(h)
        data = data + s
    return data

def create_fft_add_nths_sqrt(wt_pos, num_harmonics = 4):
    n_stride = round(1 + wt_pos * 7)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    amplitudes = [1 / math.sqrt(1 + i) for i in range(num_harmonics)]
    data = np.zeros(N)
    for h, amp in zip(harmonics, amplitudes):
        s = create_sine(h) * amp
        data = data + s
    return data

def create_lp_saw(wt_pos, num_harmonics = 8):
    """
    Square when n_stride = 2
    Anything >= 4 will start to look like a harmonic stretch on a Saw wave
    Starts to get slow after 300 or so harmonics and an IFFT may be faster
    """
    n_stride = round(1 + wt_pos * 7)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    data = np.zeros(N)
    for h in harmonics:
        s = create_sine(h) * (1 / h)
        data = data + s
    return data

def create_exp(wt_pos):
    """Looks like a kick"""
    saw = np.arange(1, 0, -1 / N)
    amount = 0.5 + wt_pos * 2
    return np.sin(0.25 * np.power(2 * np.e, amount * saw + 1.5))

def create_hourglass(wt_pos):
    saw = np.arange(-1, 1, 2 / N)
    v0 = np.arctan(np.pi * saw)
    v1 = np.cos((2 * np.pi) * saw * (1 + wt_pos * 7))
    return v0 * v1

def create_logn(wt_pos):
    n = 1 + wt_pos * 11
    b = pow(2, n)
    t_1_2 = 1 + t * b
    lnb = math.log2(b)
    v0 = np.log2(t_1_2) / lnb
    v1 = (v0 * 2) - 1
    return v1

def create_skew_saw(wt_pos):
    """Arc saw"""
    half = 0.5 + wt_pos * 0.499
    norm_saw = [skew(i / N, half) for i in range(N)]
    return np.array(norm_saw) * 2 - 1

def create_tanh_saw(wt_pos):
    """Round saw"""
    saw = np.arange(-1, 1, 2 / N)
    return np.tanh(saw * (0.1 + wt_pos * 7.9))

def create_tanh_triangle(wt_pos):
    """Round triangle"""
    half_saw = np.arange(-1, 1, 4 / N)
    triangle = np.concatenate((half_saw, half_saw * -1))
    assert len(triangle) == N
    return np.tanh(triangle * (0.1 + wt_pos * 7.9))


def create_PWM_square(wt_pos):
    pos = 0.05 + wt_pos * 0.9
    idx = int(pos * N)
    ones = np.full(idx, 1)
    neg_ones = np.full(N - idx, -1)
    return np.concatenate((ones, neg_ones))

def create_PWM_saw(wt_pos):
    pos = wt_pos * 0.9
    idx = int(pos * N)
    # make idx an even number
    if idx % 2 == 1:
        idx -= 1
    zeros = np.full(idx, 0)

    half_saw_len = int((N - idx) / 2)
    saw_pos = np.arange(0, 1, 1 / half_saw_len)
    saw_pos = saw_pos[:half_saw_len]
    saw_neg = saw_pos - 1

    assert len(zeros) == idx
    assert (idx + 2 * half_saw_len) == N
    assert len(saw_pos) == half_saw_len

    PWM_SAW = np.concatenate((saw_pos, saw_neg, zeros))
    assert len(PWM_SAW) == N

    return PWM_SAW
