from math import cos, sin, atan2, sqrt, asin
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

n = 2
delta_t = 0.05
t = []
for i in np.arange(0, 10 + delta_t, delta_t): t.append(i)
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
qrv[0] = np.subtract(qv[0], qr[0])
prv[0] = [pv * cos(theta_t[0]) - v_rd[0] * cos(theta_r[0]), pv * sin(theta_t[0]) - v_rd[0] * sin(theta_r[0])]
noise_mean = 0.5
noise_std = 0.5
error = [0] * len(t)

for i in range(1, len(t)):
    #   LINEAR
    qv_x = 5 + t[i] #+ noise_std * np.random.randn() + noise_mean
    qv_y = 5 + t[i] #+ noise_std * np.random.randn() + noise_mean

    #   SIN WAVE
    #qv_x = t[i] #+ noise_std * np.random.randn() + noise_mean
    #qv_y = 30 + 15 * sin(t[i]) #+ noise_std * np.random.randn() + noise_mean

    qv[i] = [qv_x, qv_y]
    qt_diff =  np.subtract(qv[i], qv[i-1])
    theta_t[i] = atan2(qt_diff[1], qt_diff[0])

    relative = atan2(qrv[i-1][1], qrv[i-1][0])
    v_rd[i] = sqrt(norm(pv)**2 + (2 * _lambda * norm(qrv[i-1]) * norm(pv) * abs(cos(theta_t[i] - relative))) + (_lambda**2 * norm(qrv[i-1])**2))
    theta_r[i] = relative + asin((norm(pv) * sin(theta_t[i] - relative)) / v_rd[i])

    qr[i] = np.add(qr[i-1], np.multiply(v_rd[i] * delta_t, [cos(theta_r[i-1]), sin(theta_r[i-1])]))
    qrv[i] = np.subtract(qv[i], qr[i])
    prv[i] = [pv * cos(theta_t[i]) - v_rd[i] * cos(theta_r[i]), pv * sin(theta_t[i]) - v_rd[i] * sin(theta_r[i])]

    error[i] = norm(np.subtract(qv[i], qr[i]))

qv_x_plot = [qv[j][0] for j in range(1, len(qv))]
qv_y_plot = [qv[j][1] for j in range(1, len(qv))]
plt.scatter(qv_x_plot, qv_y_plot, label="target", s=10)
qr_x_plot = [qr[k][0] for k in range(1, len(qr))]
qr_y_plot = [qr[k][1] for k in range(1, len(qr))]
plt.scatter(qr_x_plot, qr_y_plot, label="robot", s=0.5)
plt.legend()
plt.savefig("1-i.png")
plt.clf()

error_x = list(range(1, len(error)))
error_y = [error[i] for i in range(1, len(error))]
plt.scatter(error_x, error_y, label="error", s=1)
plt.legend()
plt.savefig("1-ii.png")
plt.clf()

theta_r_x = list(range(1, len(theta_r)))
theta_r_y = [theta_r[i] for i in range(1, len(theta_r))]
plt.plot(theta_r_x, theta_r_y, label="robot heading")
plt.legend()
plt.savefig("1-iii.png")
plt.clf()

v_rd_x = list(range(1, len(v_rd)))
v_rd_y = [v_rd[i] for i in range(1, len(v_rd))]
plt.plot(v_rd_x, v_rd_y, label="robot velocity")
plt.legend()
plt.savefig("1-iv.png")
plt.clf()