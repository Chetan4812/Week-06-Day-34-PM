import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Load image ────────────────────────────────────────────────────────────────
# Uses matplotlib's built-in sample image (no external file needed)
img_path = os.path.join(matplotlib.get_data_path(), 'sample_data', 'grace_hopper.jpg')
img = plt.imread(img_path).astype(float)

# To use your own image instead:
# img = plt.imread('your_image.jpg').astype(float)

print(f"Image shape : {img.shape}  (H x W x channels)")
print(f"Pixel range : [{img.min():.0f}, {img.max():.0f}]")

# ── PCA compression per colour channel ────────────────────────────────────────

n_components_list = [5, 20, 50, 100]
results = []

fig, axes = plt.subplots(2, len(n_components_list) + 1, figsize=(16, 7))

# Original image
axes[0, 0].imshow(img.astype(np.uint8))
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')
axes[1, 0].axis('off')
axes[1, 0].text(
    0.5, 0.5,
    f'Size: {img.shape[0]}x{img.shape[1]}\n3 channels\n(full)',
    ha='center', va='center', fontsize=9,
    transform=axes[1, 0].transAxes
)

total_pixels = img.shape[0] * img.shape[1] * img.shape[2]

for idx, n_comp in enumerate(n_components_list, start=1):
    reconstructed_channels = []
    total_compressed = 0
    variance_per_channel = []

    for ch in range(3):                          # R, G, B
        channel = img[:, :, ch]                  # shape (H, W)

        pca = PCA(n_components=n_comp)
        scores        = pca.fit_transform(channel)     # (H, n_comp)
        reconstructed = pca.inverse_transform(scores)  # (H, W)
        reconstructed_channels.append(reconstructed)
        variance_per_channel.append(pca.explained_variance_ratio_.sum())

        # Storage cost: scores + components + mean
        total_compressed += scores.size + pca.components_.size + pca.mean_.size

    # Stack RGB channels back
    img_rec = np.stack(reconstructed_channels, axis=2)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)

    # Metrics
    mse               = np.mean((img - img_rec.astype(float)) ** 2)
    compression_ratio = total_pixels / total_compressed
    mean_var          = np.mean(variance_per_channel) * 100

    results.append({
        'n_components':       n_comp,
        'MSE':                round(mse, 2),
        'Compression Ratio':  round(compression_ratio, 2),
        'Variance Explained': f'{mean_var:.1f}%',
    })

    print(f"n_components={n_comp:4d} | MSE={mse:7.2f} | "
          f"Compression={compression_ratio:.2f}x | Var={mean_var:.1f}%")

    # Plot reconstructed image
    axes[0, idx].imshow(img_rec)
    axes[0, idx].set_title(f'n = {n_comp}', fontweight='bold')
    axes[0, idx].axis('off')

    # Stats below image
    axes[1, idx].axis('off')
    axes[1, idx].text(
        0.5, 0.5,
        f'MSE: {mse:.1f}\nRatio: {compression_ratio:.1f}x\nVar: {mean_var:.0f}%',
        ha='center', va='center', fontsize=9,
        transform=axes[1, idx].transAxes
    )

plt.suptitle(
    'PCA Image Compression — grace_hopper.jpg\n'
    'Each column: different n_components; lower MSE = better quality',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('pca_image_compression.png', dpi=150)
plt.show()

# ── Summary Table ─────────────────────────────────────────────────────────────

df = pd.DataFrame(results)
print("\nCompression Summary:")
print(df.to_string(index=False))

print("""
Key Insight:
  n=5   → high compression, high MSE → very blurry, face barely recognisable
  n=20  → moderate compression, much lower MSE → face clear, fine detail lost
  n=50  → low compression, low MSE → near-original quality
  n=100 → minimal compression, very low MSE → almost identical to original

  The elbow is usually around n=20-50: you retain 90%+ variance
  while achieving 5-10x compression — the sweet spot for image compression.
""")
