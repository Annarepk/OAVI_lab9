import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def find_high_energy_windows(samples, sample_rate, window_duration=0.1):
    nperseg = int(sample_rate * window_duration)

    frequencies, times, Sxx = signal.spectrogram(
        samples,
        sample_rate,
        scaling='spectrum',
        window='hann',
        nperseg=nperseg,
        noverlap=0
    )

    # Энергия = сумма по всем частотам
    energy_total = np.sum(Sxx, axis=0)

    # Находим окно с максимальной энергией
    max_energy_index = np.argmax(energy_total)
    max_time = times[max_energy_index]

    print(f"Самая высокая суммарная энергия наблюдается на моменте {max_time:.2f} секунд.")
    return max_time, times, energy_total



def spectrogram_plot(samples, sample_rate, t=10000):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window='hann')
    spec = np.log10(my_spectrogram)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())

    plt.ylim(top = t)
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')


def denoise(samples, sample_rate, cutoff_freuency, passes=1):
    z = signal.savgol_filter(samples, 100, 3)
    b, a = signal.butter(3, cutoff_freuency / sample_rate)
    zi = signal.lfilter_zi(b, a)
    for _ in range(passes):
        z, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    return z

def to_pcm(y):
    return np.int16(y / np.max(np.abs(y)) * 32000)