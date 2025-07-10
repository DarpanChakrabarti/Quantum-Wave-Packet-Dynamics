import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Constants ---
hbar = 1.0
m = 1.0
N = 1024
L = 1000.0
dx = L / N
x = np.linspace(-L / 2, L / 2, N)
dt = 0.1
steps = 30000

# --- Initial Wave Packet ---
x0 = -30.0
sigma = 5.0
k0 = 1.0
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize

# --- Step Potential and Initial Energy ---
E = (hbar**2 * k0**2) / (2 * m) # hbar^2 k0^2 / 2m
V0 = 0.3
a = 10  # Half-width of barrier
V = np.zeros_like(x)
V[np.abs(x) <= a] = V0  # Barrier from -a to +a
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

x_left, x_right = -100, 100
ax.set_xlim(x_left, x_right)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.set_title("Gaussian Wave Packet Encountaring a Barrier Potential")
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

    if t == 200: # Pause at designated time
        ani.event_source.stop()

    line_re.set_ydata(np.real(psi))
    line_prob.set_ydata(prob_density)
    time_text.set_text(f'Time(t) = {t:.0f}s')
    return line_re, line_prob, line_V, time_text

# --- Run Animation ---
ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, interval=1, blit=True)

plt.show()

# --- Compute R and T after simulation ends ---

# Get final probability density
final_prob_density = np.abs(psi)**2

# Integrate in the left and right regions
x_barrier_left = -a
x_barrier_right = a

# Compute masks
mask_left = x < x_barrier_left
mask_right = x > x_barrier_right

R = np.sum(final_prob_density[mask_left]) * dx  
T = np.sum(final_prob_density[mask_right]) * dx
B = 1 - R + T # probability density that remains inside the barrier

print(f"Reflection coefficient R = {R:.4f}")
print(f"Transmission coefficient T = {T:.4f}")
print(f"R + T = {R + T + B:.4f}")
