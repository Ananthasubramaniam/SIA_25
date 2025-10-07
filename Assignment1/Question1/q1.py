import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal


duration = 10
Fs = 44100    

print("Recording...")
audio_data = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float32')
sd.wait()
print("Recording over!")

# Flatten 
audio_data = audio_data.flatten()
audio_data = audio_data / np.max(np.abs(audio_data))


write("original_speech.wav", Fs, (audio_data * 32767).astype(np.int16))


nyquist_rate = 16000
oversample_rate = 32000
undersample_rate = 8000


audio_over = signal.resample_poly(audio_data, oversample_rate, Fs)
audio_nyquist = signal.resample_poly(audio_data, nyquist_rate, Fs)
audio_under = signal.resample_poly(audio_data, undersample_rate, Fs)


recon_over = signal.resample_poly(audio_over, Fs, oversample_rate)
recon_nyquist = signal.resample_poly(audio_nyquist, Fs, nyquist_rate)
recon_under = signal.resample_poly(audio_under, Fs, undersample_rate)


recon_over /= np.max(np.abs(recon_over))
recon_nyquist /= np.max(np.abs(recon_nyquist))
recon_under /= np.max(np.abs(recon_under))


write("reconstructed_oversampled.wav", Fs, (recon_over * 32767).astype(np.int16))
write("reconstructed_nyquist.wav", Fs, (recon_nyquist * 32767).astype(np.int16))
write("reconstructed_undersampled.wav", Fs, (recon_under * 32767).astype(np.int16))


time_original = np.linspace(0, duration, len(audio_data))
time_over = np.linspace(0, duration, len(audio_over))
time_nyquist = np.linspace(0, duration, len(audio_nyquist))
time_under = np.linspace(0, duration, len(audio_under))

fig, axs = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle("Speech Signal Sampling and Reconstruction", fontsize=16)


axs[0,0].plot(time_original, audio_data, color='black')
axs[0,0].set_title("Original Speech (44.1 kHz)")
axs[0,1].plot(time_over, audio_over, color='blue')
axs[0,1].set_title("Oversampled Signal (32 kHz)")


axs[1,0].plot(time_nyquist, audio_nyquist, color='green')
axs[1,0].set_title("Nyquist Rate Sampling (16 kHz)")
axs[1,1].plot(time_under, audio_under, color='red')
axs[1,1].set_title("Under-Sampled Signal (8 kHz)")


axs[2,0].plot(time_original, recon_over, color='blue')
axs[2,0].set_title("Reconstructed from Oversampled")
axs[2,1].plot(time_original, recon_nyquist, color='green')
axs[2,1].set_title("Reconstructed from Nyquist Sampled")


axs[3,0].plot(time_original, recon_under, color='red')
axs[3,0].set_title("Reconstructed from Undersampled (Aliased)")
axs[3,1].plot(time_nyquist, audio_nyquist, 'g--')
axs[3,1].set_title("Nyquist Sampled Alternate View")


for ax in axs.flat:
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.set_ylim([-1.1, 1.1])  

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("\nPlaying back reconstructed signals...")
sd.play(recon_over, Fs); sd.wait()
sd.play(recon_nyquist, Fs); sd.wait()
sd.play(recon_under, Fs); sd.wait()

print("\nAll files saved:")
print(" - original_speech.wav")
print(" - reconstructed_oversampled.wav")
print(" - reconstructed_nyquist.wav")
print(" - reconstructed_undersampled.wav")
