import numpy as np
import matplotlib.pyplot as plt

hbar = 1.0 # Physical constants 
m = 1.0
N = 1024 # Simulation Domain
L = 1000.0
dx = L / N
x = np.linspace(-L / 2, L / 2, N)
dt = 0.1
steps = 3000

V0 = 0.3 #  Double Barrier potential 
a = 60   # inner half-width
b = 80   # outer half-width
V = np.zeros_like(x)
V[(np.abs(x) >= a) & (np.abs(x) <= b)] = V0

E_vals = np.linspace(0, 65.0, 1000) # Energy Sweep Range (0 to 65)
transmissions = []

for E in E_vals: # Main loop over different Energy values
    k0 = np.sqrt(2 * m * E) / hbar  # Calculate wave number from energy

    x0 = 0.0     #  Initial wave packet
    sigma = 15.0
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalization

    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi     # Split operator setup
    T_k = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar)
    V_x_half = np.exp(-1j * V * dt / (2 * hbar))

    for _ in range(steps):     # Time Evolution
        psi *= V_x_half
        psi_k = np.fft.fft(psi)
        psi_k *= T_k
        psi = np.fft.ifft(psi_k)
        psi *= V_x_half

    final_prob_density = np.abs(psi)**2     # Compute Transmission
    mask_right = x > b + 50  # sufficiently far beyond the barrier
    T = np.sum(final_prob_density[mask_right]) * dx
    transmissions.append(T)

plt.figure(figsize=(8, 5)) # Plot Transmission Spectrum
plt.plot(E_vals, transmissions, color='red')
plt.axvline(x=V0, color='black', linestyle='--', label='$V_0$')
plt.xlabel("Energy $(E)$")
plt.ylabel("Transmission Coefficient $(T)$")
plt.title("Transmission vs Energy for Double Barrier Potential")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
