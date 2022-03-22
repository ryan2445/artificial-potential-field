from math import cos, sin, atan2
import matplotlib.pyplot as plt
import numpy as np

n = 2
delta_t = 0.05
t = []
for i in np.arange(0, 10 + delta_t, delta_t):
    t.append(i)
_lambda = 8.5
vr_max = 50
qv = [[0] * n] * len(t)
pv = 1.2
theta_t = [0] * len(t)
qr = [[0] * n] * len(t)
v_rd = [0] * len(t)
theta_r = [0] * len(t)
qrv = [[0] * n] * len(t)
prv = [[0] * n] * len(t)
qrv[0] = np.subtract(qv, qr)
prv[0] = [pv * cos(theta_t[0]) - v_rd[0] * cos(theta_r[0]), pv * sin(theta_t[0]) - v_rd[0] * sin(theta_r[0])]
noise_mean = 0.5
noise_std = 0.5

for i in range(1, len(t)):
    qv_x = 60 - 15 * cos(t[i]) #+ noise_std * np.random.randn() + noise_mean
    qv_y = 30 + 15 * sin(t[i]) #+ noise_std * np.random.randn() + noise_mean
    for j in range(i, len(qv)): qv[j] = [qv_x, qv_y]
    qt_diff =  np.subtract(qv[i], qv[i-1])
    theta_t[i] = atan2(qt_diff[1], qt_diff[0])

    qr[i] = np.add(qr[i-1], np.multiply(v_rd[i] * delta_t, [cos(theta_r[i-1]), sin(theta_r[i-1])]))
    qrv[i] = qv[i] - qr[i]
    prv[i] = [pv * cos(theta_t[i]) - v_rd[i] * cos(theta_r[i]), pv * sin(theta_t[i]) - v_rd[i] * sin(theta_r[i])]


