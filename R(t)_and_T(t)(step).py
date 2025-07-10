import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
hbar = 1.0
m = 1.0
N = 1024
L = 500.0
dx = L / N
x = np.linspace(-L / 2, L / 2, N)
dt = 0.1
steps = 3000  # total time steps
measure_every = 1  # store R, T every N steps

# --- Initial Wave Packet ---
x0 = -25.0
sigma = 5.0
k0 = 1.0
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize

# --- Step Potential ---
V0 = 0.7
V = np.zeros_like(x)
V[x >= 0] = V0

# --- Operators ---
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
T_k = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar)
V_x_half = np.exp(-1j * V * dt / (2 * hbar))

# --- Arrays to Store Results ---
t_values = []
R_values = []
T_values = []

# --- Time Evolution ---
for step in range(steps):
    # Evolve psi by one time step using split-operator method
    psi *= V_x_half
    psi_k = np.fft.fft(psi)
    psi_k *= T_k
    psi = np.fft.ifft(psi_k)
    psi *= V_x_half

    # Store R and T every N steps
    if step % measure_every == 0:
        prob_density = np.abs(psi)**2
        R = np.sum(prob_density[x < 0]) * dx
        T = np.sum(prob_density[x >= 0]) * dx
        t_values.append(step * dt)
        R_values.append(R)
        T_values.append(T)

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(t_values, R_values, label='Reflection R(t)', color='blue')
plt.plot(t_values, T_values, label='Transmission T(t)', color='green')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Time (t)")
plt.ylabel("Coefficient")
plt.title("Time Evolution of Reflection and Transmission Coefficients")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
