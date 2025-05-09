import matplotlib.pyplot as plt
import numpy as np

# Simulated 2D embeddings for illustration
# Let's assume the reference R is at the origin
origin = np.array([0, 0])
w_a = np.array([2, 1])   # Task A embedding
w_b = np.array([1, 3])   # Task B embedding
w_c = np.array([-1.5, 1.5])   # Task B embedding
w_d = np.array([-3, 1.5])   # Task B embedding
w_e = np.array([-2.5, 0.5])   # Task B embedding
w_f = np.array([2.5, 2])   # Task B embedding

# Plot
plt.figure(figsize=(6, 6))
plt.quiver(*origin, *w_a, angles='xy', scale_units='xy', scale=1, color='blue', label='Task A')
plt.quiver(*origin, *w_b, angles='xy', scale_units='xy', scale=1, color='green', label='Task B')
plt.quiver(*origin, *w_c, angles='xy', scale_units='xy', scale=1, color='red', label='Task C')
plt.quiver(*origin, *w_d, angles='xy', scale_units='xy', scale=1, color='yellow', label='Task D')
plt.quiver(*origin, *w_e, angles='xy', scale_units='xy', scale=1, color='orange', label='Task E')
plt.quiver(*origin, *w_f, angles='xy', scale_units='xy', scale=1, color='purple', label='Task F')
plt.quiver(*origin, 0, 0, angles='xy', scale_units='xy', scale=1, color='pink', alpha=0.3, label='Reference R (origin)')

# Cosine similarity angle annotation
angle = np.arccos(np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b)))
angle_degrees = np.degrees(angle)

plt.text(1.1, 1.1, f'{angle_degrees:.1f}°', fontsize=12, color='red')

# Axes
plt.xlim(-3.5, 3.5)
plt.ylim(-0.5, 3.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()
plt.title("Illustration of Wasserstein Embeddings and Cosine Similarity")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")

plt.savefig("figure3.pdf")