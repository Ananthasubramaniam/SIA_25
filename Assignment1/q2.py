import numpy as np
import matplotlib.pyplot as plt

def conv(x, h):
    N, M = len(x), len(h)
    y = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        s = 0
        for k in range(M):
            if 0 <= n - k < N:
                s += h[k] * x[n - k]
        y[n] = s
    return y

def deconvolve(y, h):
    y, h = np.array(y, dtype=float), np.array(h, dtype=float)
    N = len(y) - len(h) + 1
    x = np.zeros(N)
    for n in range(N):
        s = y[n]
        for k in range(1, len(h)):
            if n - k >= 0:
                s -= h[k] * x[n - k]
        x[n] = s / h[0]
    return x

# Example
x_original = np.array([1, 2, 3])
h = np.array([1, 2])
y = conv(x_original, h)
x_recovered = deconvolve(y, h)
print("Original Signal (x[n]):", x_original)
print("Impulse Response (h[n]):", h)
print("Convolved Output (y[n]):", y)
print("Deconvolved Signal (Recovered x[n]):", x_recovered)


# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.stem(range(len(x_original)), x_original)
plt.title("Original Signal (x[n])")

plt.subplot(2, 2, 2)
plt.stem(range(len(h)), h)
plt.title("Impulse Response (h[n])")

plt.subplot(2, 2, 3)
plt.stem(range(len(y)), y)
plt.title("Convolved Output (y[n])")

plt.subplot(2, 2, 4)
plt.stem(range(len(x_recovered)), x_recovered)
plt.title("Deconvolved Signal (Recovered x[n])")

plt.tight_layout()
plt.show()
