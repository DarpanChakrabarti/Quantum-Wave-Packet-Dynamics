import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

hbar = 1.0
m = 1.0
N = 1024 #   Simulation Domain 
L = 500.0
dx = L / N
x = np.linspace(-L / 2, L / 2, N)

x_left = -100 #  Viewport
x_right = 100
dt = 0.1 #Time Parameters 
steps = 30000

k = np.fft.fftfreq(N, d=dx) * 2 * np.pi #  Momentum Grid 
x0 = -80.0 #  Initial Wave Packet 
sigma = 5.0
k0 = 1.0

psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx) # normalization

T_k = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar) # Time Evolution Operator 

fig, ax=plt.subplots() # Set Up Plot
line_re,=ax.plot(x, np.real(psi), color='blue', label='Re($\\psi$)', linestyle='--')
line_prob,=ax.plot(x, np.abs(psi)**2, color='red', label='$|\\psi|^2$', linestyle='-')
time_text = ax.text(0.37, 0.2, '', transform=ax.transAxes, fontsize=12, color='Red')

ax.set_xlim(x_left, x_right)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x")
ax.set_ylabel("$Re(\\psi)$ / Probability Density")
ax.set_title("Free Gaussian Wave Packet")
ax.grid(True)
ax.legend()

sigmas = [] # Lists to store sigma(t) and time
times = []

def init(): # Initialization for Animation
    line_re.set_ydata(np.real(psi))
    line_prob.set_ydata(np.abs(psi)**2)
    time_text.set_text('')
    return line_re, line_prob, time_text

def update(frame): # Update Function
    global psi
    t = frame * dt
    psi_k = np.fft.fft(psi)    # Time evolution
    psi_k *= T_k
    psi = np.fft.ifft(psi_k)

    prob_density = np.abs(psi)**2     # Calculate sigma(t)
    x_mean = np.sum(x * prob_density) * dx
    x2_mean = np.sum(x**2 * prob_density) * dx
    sigma_t = np.sqrt(x2_mean - x_mean**2)
    
    if frame < steps: # Prevents appending beyond final frame (safety check)
        sigmas.append(sigma_t)
        times.append(t)
    if t == 100:            # Ending the animation at designated time
        ani.event_source.stop()
        
    line_re.set_ydata(np.real(psi))    # Update plot
    line_prob.set_ydata(prob_density)
    time_text.set_text(f'Time(t) = {t:.0f}s')
    return line_re, line_prob, time_text

# Run Animation
ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, interval=10, blit=True) 
# Smaller interval makes the animation faster
plt.show()

plt.figure(figsize=(7, 4)) # Plot sigma(t) vs Time
plt.plot(times, sigmas, color='green')
plt.xlabel("Time (s)")
plt.ylabel("Wave Packet Width sigma(t)")
plt.title("Spreading of the Wave Packet Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
