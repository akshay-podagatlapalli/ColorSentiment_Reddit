# Required imports
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

# Load the emotion data
emotion_data = pd.read_csv("output2.csv")

# Define a mapping of emotion types to specific colors
emotion_colors = {
    'joy': 'yellow',
    'sadness': 'blue',
    'anger': 'red',
    'fear': 'purple',
    'love': 'green',
    'surprise': 'orange'
}

# Create a continuous colormap for all emotions
emotion_cmap = mcolors.LinearSegmentedColormap.from_list(
    "emotion_cmap", [(i / (len(emotion_colors) - 1), color) for i, color in enumerate(emotion_colors.values())]
)

# Process the data by title
scores_dict = {}
emotion_types_dict = {}

for title, group in emotion_data.groupby('title'):
    scores = group['score'].to_numpy()
    emotion_types = group['type'].to_numpy()
    scores_dict[title] = scores
    emotion_types_dict[title] = emotion_types

x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x, y)

# Visualization loop
for i, (title, scores) in enumerate(scores_dict.items()):
    emotion_types = emotion_types_dict[title]
    base_field = np.interp(
        np.linspace(0, len(scores) - 1, 500), np.arange(len(scores)), scores
    )
    
    # Generate Perlin-like noise
    def generate_perlin_noise(size, scale):
        grid = np.random.rand(size[0] // scale + 1, size[1] // scale + 1)
        smooth = gaussian_filter(grid, sigma=scale)
        noise = np.kron(smooth, np.ones((scale, scale)))
        return noise[:size[0], :size[1]]

    size = (500, 500)
    scale = 10000
    perlin_noise = generate_perlin_noise(size, scale)

    noise = np.random.normal(loc=0.0, scale=5, size=(500, 500))

    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)

    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())  # Normalize Z for gradient colors

    # Map scores to colors using the custom colormap
    gradient_colors = emotion_cmap(Z_norm)

    # Plot the results
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_colors, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    wrapped_title = "\n".join(textwrap.wrap(title, width=100))
    plt.title(wrapped_title, fontsize=14, color='black', pad=10)
    plt.axis('off')  # Hide axes for a cleaner look

    # Create a colorbar with emotion names
    norm = mcolors.Normalize(vmin=0, vmax=len(emotion_colors) - 1)
    sm = plt.cm.ScalarMappable(cmap=emotion_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, len(emotion_colors)))
    cbar.ax.set_yticklabels(emotion_colors.keys())  # Replace ticks with emotion names
    cbar.set_label('Emotion Types', rotation=270, labelpad=20)
    
    plt.show()
