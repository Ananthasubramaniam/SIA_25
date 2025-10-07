import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

Fs = 44100
duration = 5

print("Recording...")
audio = sd.rec(int(Fs * duration), samplerate=Fs, channels=1, dtype='float32')
sd.wait()
print("Recording over!")

audio = audio.flatten()

def conv(x, h):
    N, M = len(x), len(h)
    y = np.zeros(N + M - 1)
    for i in range(M):
        y[i:i+N] += h[i] * x
    return y

autocorr = conv(audio, audio[::-1])

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(audio, color='black')
plt.title("Original Speech Signal")
plt.subplot(2, 1, 2)
plt.plot(autocorr, color='purple')
plt.title("Autocorrelation")
plt.tight_layout()
plt.show()
