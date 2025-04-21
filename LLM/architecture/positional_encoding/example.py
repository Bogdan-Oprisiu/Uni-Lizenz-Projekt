# Set parameters for dummy data and model dimensions
import torch
from matplotlib import pyplot as plt

from architecture.positional_encoding.learned_positional_encoding import LearnedPositionalEncoding
from architecture.positional_encoding.regular_positional_encoding import RegularPositionalEncoding

batch_size = 1
seq_len = 50  # Sequence length for visualization
d_model = 16  # Small model dimension for clarity

# Create dummy embeddings as zeros so that the output equals the positional encoding
dummy_embeddings = torch.zeros(batch_size, seq_len, d_model)

# Instantiate both positional encoding modules
regular_encoder = RegularPositionalEncoding(d_model, dropout=0.0, max_len=5000)
learned_encoder = LearnedPositionalEncoding(d_model, max_len=5000)

# Apply the positional encodings
regular_encoded = regular_encoder(dummy_embeddings)  # [1, seq_len, d_model]
learned_encoded = learned_encoder(dummy_embeddings)  # [1, seq_len, d_model]

# Remove batch dimension and convert to numpy arrays for plotting
regular_pe = regular_encoded.squeeze(0).detach().numpy()  # Shape: [seq_len, d_model]
learned_pe = learned_encoded.squeeze(0).detach().numpy()  # Shape: [seq_len, d_model]

# ----- Plot 1: Line plots for selected dimensions -----
dims_to_plot = [0, 1, 2, 3]

plt.figure(figsize=(12, 5))

# Line plots for Regular (Sinusoidal) Positional Encoding
plt.subplot(1, 2, 1)
for d in dims_to_plot:
    plt.plot(regular_pe[:, d], label=f"Dim {d}")
plt.title("Regular (Sinusoidal) Positional Encoding")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.legend()

# Line plots for Learned Positional Encoding
plt.subplot(1, 2, 2)
for d in dims_to_plot:
    plt.plot(learned_pe[:, d], label=f"Dim {d}")
plt.title("Learned Positional Encoding")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.legend()

plt.tight_layout()
plt.show()

# ----- Plot 2: Heatmaps for the complete positional encoding matrices -----
plt.figure(figsize=(12, 5))

# Heatmap for Regular Positional Encoding
plt.subplot(1, 2, 1)
plt.imshow(regular_pe, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Heatmap - Regular Positional Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")

# Heatmap for Learned Positional Encoding
plt.subplot(1, 2, 2)
plt.imshow(learned_pe, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Heatmap - Learned Positional Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")

plt.tight_layout()
plt.show()
