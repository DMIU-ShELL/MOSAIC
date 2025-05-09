import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2D task embeddings
n_tasks = 12
embeddings = np.random.randn(n_tasks, 2)

# Normalize to unit vectors for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
unit_embeddings = embeddings / norms

# Define reference task embedding (your task)
ref_embedding = np.array([1.0, 1.0])
ref_unit = ref_embedding / np.linalg.norm(ref_embedding)

# Compute cosine similarities
cos_sims = unit_embeddings @ ref_unit
threshold = 0.5
selected = cos_sims > threshold

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Cosine Similarity with Reference Task (Threshold > 0.5)", fontsize=13)

# Plot vectors
for i, vec in enumerate(unit_embeddings):
    color = 'blue' if selected[i] else 'lightgray'
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color)
    ax.text(vec[0]*1.1, vec[1]*1.1, f"T{i}", fontsize=10, color='black')

# Plot reference vector
ax.quiver(0, 0, ref_unit[0], ref_unit[1], angles='xy', scale_units='xy', scale=1.2, color='black', label='Reference Task')

# Draw cosine similarity threshold region (cos^-1(0.5) ≈ 60°)
angle_threshold = np.arccos(threshold)
theta = np.linspace(0, angle_threshold, 100)
arc_radius = 0.7
arc_x = arc_radius * np.cos(theta)
arc_y = arc_radius * np.sin(theta)
ax.plot(arc_x, arc_y, 'r--', lw=1, label='cos(θ) > 0.5 region')

# Formatting
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 1.5)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("Latent Dimension 1")
ax.set_ylabel("Latent Dimension 2")
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('figure1.pdf')