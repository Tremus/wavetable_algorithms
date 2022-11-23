import numpy as np
import math
from src.config import N
from src.utils import *

t: NDDoubleArr = np.arange(0, 1, 1 / N)
assert len(t) == N
M_PI = math.pi
M_PI2 = math.pi / 2
DEFAULT_PHASE = -1.5707963267948966j

def create_sine(freq: float) -> NDDoubleArr:
    assert freq <= (N / 2)
    p: float = (2 * np.pi * freq)
    return np.sin(t * p)

def create_saw() -> NDDoubleArr:
    return np.arange(-1, 1, 2 / N)

def create_triangle() -> NDDoubleArr:
    ramp_up = np.arange(-1, 1, 4 / N)
    ramp_down = np.flip(ramp_up)
    return np.concatenate([ramp_up, ramp_down])

def create_basic_shapes(wt_pos:float):
    # 5 shapes
    num = round(wt_pos * 4)
    if num == 0:
        return create_sine(1)
    if num == 1:
        pt1 = np.arange(0, 1, 1 / (N // 4))
        pt2 = 1 - pt1
        pt3 = -1 + pt2
        pt4 = -1 + pt1
        return np.concatenate([pt1, pt2, pt3, pt4])
    if num == 2:
        pt1 = np.arange(0, 1, 2 / N)
        pt2 = -1 + pt1
        return np.concatenate([pt1, pt2])
    # square
    if num == 3:
        return create_PWM_square(0.5)
    # pulse
    return create_PWM_square(0.225)

def sine_add(wt_pos: float) -> NDDoubleArr:
    num_harmonics = wt_pos * 15
    data = create_sine(1)
    for Hz in range(2, 1 + int(num_harmonics)):
        data = data + create_sine(Hz)
    if num_harmonics > int(num_harmonics):
        amp = num_harmonics - int(num_harmonics)
        next_harmonic = 1 + int(num_harmonics)
        data = data + amp * create_sine(next_harmonic)
    return data

def create_fft_add_nths(wt_pos: float, num_harmonics = 4) -> NDDoubleArr:
    """sounds good when n_stride = 2/3/4"""
    n_stride = round(1 + wt_pos * 3)
    harmonics = [1 + i * n_stride for i in range(num_harmonics)]
    data = np.zeros(N)
    for h in harmonics:
        s = create_sine(h)
        data = data + s
    return data


def create_fft_add_nths_sqrt(wt_pos: float, num_harmonics = 4, max_stride = 8, smooth = False) -> NDDoubleArr:
    """Works similar to a 'harmonic stretch' spectral filter on saw waves"""
    assert max_stride > 1
    n_stride = 1 + wt_pos * (max_stride - 1)
    if not smooth:
        n_stride = round(n_stride)
    data = np.zeros(N)
    for i in range(num_harmonics):
        harmonic = 1 + i * n_stride
        amplitude = 1 / math.sqrt(1 + i)
        s = create_sine(math.floor(harmonic))
        if smooth:
            s2 = create_sine(math.ceil(harmonic))
            interp_amt = harmonic - math.floor(harmonic)
            s = lininterp(interp_amt, s, s2)
        s *= amplitude
        data += s
    return data

def create_lp_saw(wt_pos: float, num_harmonics = 8) -> NDDoubleArr:
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

def create_exp(wt_pos: float) -> NDDoubleArr:
    """Looks like a kick"""
    saw: NDDoubleArr = np.arange(1, 0, -1 / N)
    amount = 0.5 + wt_pos * 2
    return np.sin(0.25 * np.power(2 * np.e, amount * saw + 1.5))

def create_hourglass(wt_pos: float) -> NDDoubleArr:
    saw = np.arange(-1, 1, 2 / N)
    v0 = np.arctan(np.pi * saw)
    v1 = np.cos((2 * np.pi) * saw * (1 + wt_pos * 7))
    return v0 * v1

def create_logn(wt_pos: float) -> NDDoubleArr:
    n = 1 + wt_pos * 11
    b = pow(2, n)
    t_1_2 = 1 + t * b
    lnb = math.log2(b)
    v0 = np.log2(t_1_2) / lnb
    v1 = (v0 * 2) - 1
    return v1

def create_skew_saw(wt_pos: float) -> NDDoubleArr:
    """Arc saw"""
    half = 0.5 + wt_pos * 0.499
    norm_saw = [skew(i / N, half) for i in range(N)]
    return np.array(norm_saw) * 2 - 1

def create_tanh_saw(wt_pos: float) -> NDDoubleArr:
    """Round saw"""
    saw = np.arange(-1, 1, 2 / N)
    return np.tanh(saw * (0.1 + wt_pos * 7.9))

def create_tanh_triangle(wt_pos: float) -> NDDoubleArr:
    """Round triangle"""
    half_saw = np.arange(-1, 1, 4 / N)
    triangle = np.concatenate((half_saw, half_saw * -1))
    assert len(triangle) == N
    return np.tanh(triangle * (0.1 + wt_pos * 7.9))

def create_PWM_square(wt_pos: float) -> NDDoubleArr:
    pos = 0.05 + wt_pos * 0.9
    idx = int(pos * N)
    ones = np.full(idx, 1)
    neg_ones = np.full(N - idx, -1)
    return np.concatenate((ones, neg_ones))

def create_PWM_saw(wt_pos: float) -> NDDoubleArr:
    pos = wt_pos * 0.9
    idx = int(pos * N)
    # make idx an even number
    if idx % 2 == 1:
        idx -= 1
    zeros = np.full(idx, 0)

    half_saw_len = (N - idx) // 2
    saw_pos: NDDoubleArr = np.arange(0, 1, 1 / half_saw_len)
    saw_pos = saw_pos[:half_saw_len]
    saw_neg = saw_pos - 1

    assert len(zeros) == idx
    assert (idx + 2 * half_saw_len) == N
    assert len(saw_pos) == half_saw_len

    PWM_SAW: NDDoubleArr = np.concatenate((saw_pos, saw_neg, zeros))
    assert len(PWM_SAW) == N

    return PWM_SAW

def create_FFT_random(wt_pos: float) -> NDCDoubleArr:
    # zeros
    arr: NDCDoubleArr = np.full(N, np.complex128(0+DEFAULT_PHASE))
    # set some magnitudes
    arr[1] = np.complex128(1+DEFAULT_PHASE)
    arr[3] = np.complex128((0.7 - 0.3 * wt_pos)+(DEFAULT_PHASE * wt_pos * 2))
    arr[5] = np.complex128((0.5 - 0.5 * wt_pos)+DEFAULT_PHASE)
    arr[7] = np.complex128((1 * wt_pos)+(DEFAULT_PHASE * wt_pos * 1.3))
    arr[9] = np.complex128((0.4 * wt_pos)+(DEFAULT_PHASE * wt_pos * 2.5))

    arr = np_polar_to_rectangular(arr)

    return np.fft.ifft(arr)
