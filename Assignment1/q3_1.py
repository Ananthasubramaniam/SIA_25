import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import random

Fs = 44100
duration = 5
print("Recording...")
audio = sd.rec(int(Fs * duration), samplerate=Fs, channels=1, dtype='float32')
sd.wait()
print("Recording over!")

audio = audio.flatten()
segment_length = 2000
start = random.randint(0, len(audio) - segment_length - 1)
template = audio[start:start + segment_length]

# Cross-correlation using convolution
cross_corr = np.convolve(audio, template[::-1], mode='full')

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(audio, color='black'); plt.title("Original Speech Signal")
plt.subplot(3, 1, 2)
plt.plot(template, color='blue'); plt.title("Template Segment")
plt.subplot(3, 1, 3)
plt.plot(cross_corr, color='green'); plt.title("Cross-Correlation using np.convolve()")
plt.tight_layout()
plt.show()

