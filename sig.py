#!/usr/bin/env python
import random
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt  # $ pip install matplotlib
import matplotlib.animation as animation
from math import pi

Td = 0.001
Nperiod = int(0.1/Td)
N = Nperiod
x1 = deque([0], maxlen=Nperiod)
x2 = [i for i in range(N)]
x2 = x2[:int(N / 2)]
y1 = deque([0], maxlen=Nperiod)
y2 = deque([0], maxlen=N)
fig, (ax1,ax2,ax3) = plt.subplots(3, 1)#plt.subplots()

line, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], "bx", lw=2)
line3, = ax3.plot([], [], "bo", lw=2)

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    Xk = np.dot(M, x)
    return Xk

def FFT_recursive(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        return DFT_slow(x)
    else:
        X_even = FFT_recursive(x[::2])
        X_odd = FFT_recursive(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                               X_even + factor[int(N / 2):] * X_odd])

def cos_sin(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    k = k[:int(N/2)]
    time1 = time.time()
    COS = np.cos(2 * np.pi * k * n / N)
    SIN = np.sin(2 * pi * k * n / N)
    Ak = np.dot(COS, x)
    print (Ak)
    Bk = np.dot(SIN, x)
    Xcossin = np.sqrt(Ak**2 + Bk**2)
    # for i in k:
    #     Acos = []
    #     Asin = []
    #     for l in range(N):
    #         Acos.append(x[l] * cos(2 * pi * i * l / N))
    #         Asin.append(x[l] * sin(2 * pi * i * l / N))
    #     Xcossin.append(sqrt(sum(Acos, 0) ** 2 + sum(Asin, 0) ** 2))
    time2 = time.time()
    print("DCST = ", time2 - time1)
    return Xcossin

def init():
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line, line2, line3,

def update(dy):
    xx = x1[-1] + Td
    x1.append(xx)  # update data
    #y.append(y[-1] + dy)
    yy = np.sin(x1[-1]*20*pi) #+ 0.95 * np.sin(x1[-1]*40*pi) + 0.8 * np.sin(x1[-1]*60*pi) + 0.6 * np.sin(x1[-1]*100*pi) + 0.5 * np.sin(x1[-1]*120*pi) + random.randint(-5, 5) / 4
    y1.append(yy)
    y2.append(yy)


    line.set_data(x1, y1)
    if xx >= Td * N:
        #print('y2 = ',y2)
        #y3 = list(y1)
        #print('y3 = ', y3)
        #y = abs(np.fft.fft(y2))
        y_fft = abs(FFT_recursive(y2))
        y_fft = y_fft[:int(N/2)]
        y_cs = cos_sin(y2)
        #print('y = ', y)
        line2.set_data(x2, y_fft)
        line3.set_data(x2, y_cs)
    ax1.relim()  # update axes limits
    ax1.autoscale_view(True, True, True)
    ax2.relim()  # update axes limits
    ax2.autoscale_view(True, True, True)
    ax3.relim()  # update axes limits
    ax3.autoscale_view(True, True, True)
    #print(time.time())
    return line, line2, line3

ani = animation.FuncAnimation(fig, update, interval=Td)
ani.save('fft.gif', writer='imagemagick')
plt.show()
