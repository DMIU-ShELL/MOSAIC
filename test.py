import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate task embeddings (2D)
n_tasks = 12
embeddings = np.random.randn(n_tasks, 2)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
unit_embeddings = embeddings / norms

# Reference embedding
ref_embedding = np.array([1.0, 1.0])
ref_unit = ref_embedding / np.linalg.norm(ref_embedding)

# Cosine similarities
cos_sims = unit_embeddings @ ref_unit
angle_threshold = np.arccos(0.5)  # similarity > 0.5
selected_similarity = cos_sims > 0.5

# Simulate performance
own_perf = np.random.uniform(0.4, 0.7, n_tasks)
peer_perf = np.random.uniform(0.5, 0.9, n_tasks)
performance_pass = peer_perf > own_perf
final_selected = selected_similarity & performance_pass

# --- Plot 1: Angular Similarity ---
fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_title("Heuristic 1: Cosine Similarity > 0.5", fontsize=13)

# Draw vectors and highlight based on threshold
for i, vec in enumerate(unit_embeddings):
    color = 'blue' if selected_similarity[i] else 'lightgray'
    ax1.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color)
    ax1.text(vec[0]*1.1, vec[1]*1.1, f"T{i}", fontsize=10, color='black')

# Reference vector
ax1.quiver(0, 0, ref_unit[0], ref_unit[1], angles='xy', scale_units='xy', scale=1.2, color='black', label='Reference Task')

# Draw arcs for angle visualization
theta = np.linspace(0, angle_threshold, 100)
arc_radius = 0.7
arc_x = arc_radius * np.cos(theta)
arc_y = arc_radius * np.sin(theta)
ax1.plot(arc_x, arc_y, 'r--', lw=1, label='cos(θ) > 0.5 region')

# Formatting
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(0, 1.5)
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_xlabel("Latent Dimension 1")
ax1.set_ylabel("Latent Dimension 2")
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal')

# --- Plot 2: Peer vs Own Performance (Filtered by Similarity) ---
fig, ax2 = plt.subplots(figsize=(10, 4))
filtered_indices = np.where(selected_similarity)[0]
bars = peer_perf[filtered_indices] - own_perf[filtered_indices]
colors = ['green' if performance_pass[i] else 'lightgray' for i in filtered_indices]

ax2.bar([f"T{i}" for i in filtered_indices], bars, color=colors)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_title("Heuristic 2: Peer > Own Performance (Among Similar Tasks)", fontsize=13)
ax2.set_ylabel("Peer - Own Performance")
ax2.set_xlabel("Task Index")
ax2.grid(True)

plt.tight_layout()
plt.savefig('figure.pdf')
plt.close()