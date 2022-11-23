import soundfile as sf
from src.methods import *

assert N == 2048

NUM_FRAMES = 256
assert NUM_FRAMES > 1
assert NUM_FRAMES <= 256

inc = 1 / (NUM_FRAMES - 1)

def export_to_flac(name: str, data: npt.ArrayLike):
    sf.write(
        file='%s.flac' % name,
        data=data,
        samplerate=48000 * 2,
        subtype='PCM_24')
def export_to_wav(name: str, data: npt.ArrayLike):
    sf.write(
        file='%s_wav.wav' % name,
        data=data,
        samplerate=48000 * 2,
        subtype='FLOAT')
def export_to_aiff(name: str, data: npt.ArrayLike):
    sf.write(
        file='%s_aiff.aiff' % name,
        data=data,
        samplerate=48000 * 2,
        subtype='FLOAT')
def export_to_ogg(name: str, data: npt.ArrayLike):
    sf.write(
        file='%s_ogg.ogg' % name,
        data=data,
        samplerate=48000 * 2,
        subtype='VORBIS')

def create_wave(wt_pos: float):
    # wave = create_fft_add_nths(wt_pos)
    wave = create_fft_add_nths_sqrt(wt_pos, 4, 16, False)
    wave = normalise_0dB(wave)
    return wave

frames = []
for i in range(NUM_FRAMES):
    pos = min(i * inc, 1.0)
    wave = create_exp(pos)
    wave = normalise_0dB(wave)
    frames.append(wave)

data = np.concatenate(frames)
# data = normalise_0dB(data)
assert len(data) == (N * NUM_FRAMES)

export_to_flac('exp_%d' % NUM_FRAMES, data)
