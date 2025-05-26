import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def highEnergyWindows(samples, sampleRate, windowDuration=0.1):
    nperseg = int(sampleRate * windowDuration)

    frequencies, times, Sxx = signal.spectrogram(
        samples,
        sampleRate,
        scaling='spectrum',
        window='hann',
        nperseg=nperseg,
        noverlap=0
    )

    # Энергия = сумма по всем частотам
    energyTotal = np.sum(Sxx, axis=0)

    # Находим окно с максимальной энергией
    maxEnergyIndex = np.argmax(energyTotal)
    maxTime = times[maxEnergyIndex]

    print(f"Самая высокая суммарная энергия наблюдается на моменте {maxTime:.2f} секунд.")
    return maxTime, times, energyTotal



def spectrogram(samples, sampleRate, t=10000):
    frequencies, times, mySpectrogram = signal.spectrogram(samples, sampleRate, scaling='spectrum', window='hann')
    spec = np.log10(mySpectrogram)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())

    plt.ylim(top = t)
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')


def denoise(samples, sampleRate, cutoffFreuency, passes=1):
    z = signal.savgol_filter(samples, 100, 3)
    b, a = signal.butter(3, cutoffFreuency / sampleRate)
    zi = signal.lfilter_zi(b, a)
    for _ in range(passes):
        z, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    return z

def PCM(y):
    return np.int16(y / np.max(np.abs(y)) * 32000)