import numpy as np

def conv(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        s = 0
        for k in range(M):
            if 0 <= n - k < N:
                s += h[k] * x[n - k]
        y[n] = s
    return y

def deconvolve(y, h):
    y = np.array(y, dtype=float)
    h = np.array(h, dtype=float)
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
x_original = [1, 2, 3]
h = [1, 2]
y = conv(x_original, h)
print("y:", y)

x_recovered = deconvolve(y, h)

print("Original x:", x_original)
print("Recovered x:", x_recovered)

