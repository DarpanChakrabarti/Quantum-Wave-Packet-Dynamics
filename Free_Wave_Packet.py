import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

hbar = 1.0 # --- Physical Constants (natural units) ---
m = 1.0
N = 1024 # --- Simulation Domain ---
L = 400.0
dx = L / N
L_view = L/2
x = np.linspace(-L_view / 2, L_view / 2, N)

x_left = -L_view/2 # --- Viewport Control ---
x_right = L_view/2

dt = 0.1 # --- Time Parameters ---
steps = int((L)*(1/dt)) # Calculating the number of steps to-
times = np.arange(0, steps * dt, dt)        # cover the whole animation
sigmas = []

k = np.fft.fftfreq(N, d=dx) * 2 * np.pi # --- Momentum grid for FFT ---
x0 = -(L_view/2)+20 # --- Initial Wave Packet ---
sigma0 = 5.0
k0 = 1.0

norm = (1 / (2 * np.pi * sigma0**2))**0.25 # Normalize wavefunction
psi = norm * np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# --- Time Evolution Operators ---
T_k = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar)

fig, ax=plt.subplots() # --- Plot Setup for Animation ---
line_re,=ax.plot(x, np.real(psi), color='blue', label='Re($\\psi$)', linestyle='--')
line_prob,=ax.plot(x, np.abs(psi)**2, color='red', label='$|\\psi|^2$', linestyle='-')
time_text=ax.text(0.395, 0.2, '', transform=ax.transAxes, fontsize=12, color='Red')

ax.set_xlim(x_left, x_right)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.set_title("Free Gaussian Wave Packet")
ax.grid(True)
ax.legend()

def init(): # --- Initialization function for blitting ---
    line_re.set_ydata(np.real(psi))
    line_prob.set_ydata(np.abs(psi)**2)
    time_text.set_text('')
    return line_re, line_prob, time_text

def update(frame): # --- Animation Update Function ---
    global psi
    t = frame * dt

    if frame == (steps-1): # Ending the animation
        ani.event_source.stop()

    psi_k = np.fft.fft(psi)  # Time evolution
    psi_k *= T_k
    psi = np.fft.ifft(psi_k)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    prob_density = np.abs(psi)**2   # Compute sigma(t)
    x_mean = np.sum(x * prob_density) * dx
    x2_mean = np.sum(x**2 * prob_density) * dx
    sigma_t = np.sqrt(x2_mean - x_mean**2)
    if len(sigmas) < steps:
        sigmas.append(sigma_t)

    line_re.set_ydata(np.real(psi)) # Update plot
    line_prob.set_ydata(prob_density)
    time_text.set_text(f'Time(t) = {t:.0f}s')
    return line_re, line_prob, time_text

# --- Animation ---
ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, interval=1, blit=True) 
plt.show()
sigma_theory = []
sigma_theory.append(sigma0*(1+(times/(2*sigma0**2)))**0.5)

plt.figure(figsize=(7, 4)) # --- Plot sigma(t) after animation ---
plt.plot(times[:len(sigmas)], sigmas, color='green', label='$\\sigma$(t)')
plt.xlabel("Time (t)")
plt.ylabel("Wave Packet Width $\\sigma$(t)")
plt.title("Spread of the Wave Packet Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
