import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants ---
hbar = 1.0
m = 1.0
N = 2048  # Simulation domain
L = 1000.0
dx = L / N
x = np.linspace(-L / 2, L / 2, N)
dt = 0.1
steps = 30000

# --- Initial Wave Packet ---
x0 = 0.0
sigma = 15.0
k0 = 1.0
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize

# --- Step Potential and Initial Energy ---
E = (hbar**2 * k0**2) / (2 * m) # hbar^2 k0^2 / 2m
V0 = 0.5
a = 60   # Inner half-width of the barriers
b = 80  # Outer half-width of barriers
V = np.zeros_like(x)
V[(np.abs(x) >= a) & (np.abs(x) <= b)] = V0  # Barriers at +-a to +-b
print("Energy: ", E, "V0: ", V0)

# --- Momentum Grid and Operators ---
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
T_k = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar)
V_x_half = np.exp(-1j * V * dt / (2 * hbar))  # Split-operator (half step)

# --- Set Up Plot ---
fig, ax = plt.subplots()
line_re, = ax.plot(x, np.real(psi), color='blue', label='Re($\\psi$)', linestyle='--')
line_prob, = ax.plot(x, np.abs(psi)**2, color='red', label='$|\\psi|^2$', linestyle='-')
line_V, = ax.plot(x, V / V0 * 0.4, color='black', label='V(x)', linewidth=1.2)  # Scaled for visibility
time_text = ax.text(0.35, 0.2, '', transform=ax.transAxes, fontsize=12, color='red')

x_left, x_right = -300, 300
ax.set_xlim(x_left, x_right)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x")
ax.set_ylabel("$Re(\\psi)$ / Probability Density")
ax.set_title("Gaussian Wave Packet Scattering off a Double Barrier potential")
ax.grid(True)
ax.legend()

sigmas = []
times = []

# --- Initialization ---
def init():
    line_re.set_ydata(np.real(psi))
    line_prob.set_ydata(np.abs(psi)**2)
    time_text.set_text('')
    return line_re, line_prob, line_V, time_text

# --- Time Update ---
def update(frame):
    global psi
    t = frame * dt

    psi *= V_x_half # V/2 operator
    psi_k = np.fft.fft(psi) # F-Transform
    psi_k *= T_k # T operator
    psi = np.fft.ifft(psi_k) # Inverse F-Transform
    psi *= V_x_half # V/2 operator
    prob_density = np.abs(psi)**2     # Compute prob density

    if t == 350: # Pause at designated time
        ani.event_source.stop()
        
    line_re.set_ydata(np.real(psi))
    line_prob.set_ydata(prob_density)
    time_text.set_text(f'Time(t) = {t:.0f}s')
    return line_re, line_prob, line_V, time_text

# --- Run Animation ---
ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, interval=1, blit=True)

plt.show()
