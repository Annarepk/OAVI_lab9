from scipy.io import wavfile
import matplotlib.pyplot as plt
from noise import highEnergyWindows, spectrogram, denoise, PCM

directory = "audio"
inputFile = f"{directory}/music.wav"

dpi = 500

sampleRate, samples = wavfile.read(inputFile)
if samples.ndim > 1:
    samples = samples[:, 0]  # если стерео, берём только один канал

plt.figure(dpi=dpi)
spectrogram(samples, sampleRate, 20000)
plt.savefig(f"{directory}/spectrogram.png", dpi=dpi)
plt.clf()

print("The spectrogram is saved...")

denoised0 = denoise(samples, sampleRate, cutoffFreuency=1000, passes=0)
spectrogram(denoised0, sampleRate, 20000)
plt.savefig(f"{directory}/denoised_spectrogram_savgol.png", dpi=dpi)
plt.clf()

print("The Savitsky-Goley denoised spectrogram is saved...")



denoised = denoise(samples, sampleRate, cutoffFreuency=1000)
spectrogram(denoised, sampleRate)
plt.savefig(f"{directory}/denoised_spectrogram_once.png", dpi=dpi)
plt.clf()



wavfile.write(f"{directory}/denoised_once.wav", sampleRate, PCM(denoised))

denoised2 = denoise(samples, sampleRate, cutoffFreuency=1000, passes=2)
spectrogram(denoised2, sampleRate)
plt.savefig(f"{directory}/denoised_spectrogram_twice.png", dpi=dpi)
plt.clf()

wavfile.write(f"{directory}/denoised_twice.wav", sampleRate, PCM(denoised2))

# Анализ энергии
maxTime, times, energyTotal = highEnergyWindows(samples, sampleRate)
plt.plot(times, energyTotal)
plt.title('Общая энергия сигнала по времени')
plt.xlabel('Время [с]')
plt.ylabel('Энергия')
plt.grid(True)
plt.savefig(f"{directory}/energy_peaks.png", dpi=500)
plt.clf()

