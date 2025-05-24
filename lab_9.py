from scipy.io import wavfile
import matplotlib.pyplot as plt
from noise import find_high_energy_windows, spectrogram_plot, denoise, to_pcm

directory = "audio"
inputFile = f"{directory}/music.wav"

dpi = 500

sample_rate, samples = wavfile.read(inputFile)
if samples.ndim > 1:
    samples = samples[:, 0]  # если стерео, берём только один канал

plt.figure(dpi=dpi)
spectrogram_plot(samples, sample_rate, 20000)
plt.savefig(f"{directory}/spectrogram.png", dpi=dpi)
plt.clf()

print("The spectrogram is saved...")

denoised_0 = denoise(samples, sample_rate, cutoff_freuency=1000, passes=0)
spectrogram_plot(denoised_0, sample_rate, 20000)
plt.savefig(f"{directory}/denoised_spectrogram_savgol.png", dpi=dpi)
plt.clf()

print("The Savitsky-Goley denoised spectrogram is saved...")



denoised = denoise(samples, sample_rate, cutoff_freuency=1000)
spectrogram_plot(denoised, sample_rate)
plt.savefig(f"{directory}/denoised_spectrogram_once.png", dpi=dpi)
plt.clf()



wavfile.write(f"{directory}/denoised_once.wav", sample_rate, to_pcm(denoised))

denoised_2 = denoise(samples, sample_rate, cutoff_freuency=1000, passes=2)
spectrogram_plot(denoised_2, sample_rate)
plt.savefig(f"{directory}/denoised_spectrogram_twice.png", dpi=dpi)
plt.clf()

wavfile.write(f"{directory}/denoised_twice.wav", sample_rate, to_pcm(denoised_2))

# Анализ энергии
max_time, times, energy_total = find_high_energy_windows(samples, sample_rate)
plt.plot(times, energy_total)
plt.title('Общая энергия сигнала по времени')
plt.xlabel('Время [с]')
plt.ylabel('Энергия')
plt.grid(True)
plt.savefig(f"{directory}/energy_peaks.png", dpi=500)
plt.clf()

