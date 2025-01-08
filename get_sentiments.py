# getting all the required modules 
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation

sent_data = pd.read_csv("data/output.csv")

scores_dict = {}

for title, group in sent_data.groupby('title'):
    scores = group['result'].to_numpy()
    scores_dict[title] = scores

x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x, y)

for i in range(0, 10):
    value_array = list(scores_dict.values())[i]
    title = list(scores_dict.keys())[i]
    base_field = np.interp(
        np.linspace(0, len(value_array) - 1, 500), np.arange(len(value_array)), value_array
    )
    def generate_perlin_noise(size, scale):
        grid = np.random.rand(size[0] // scale + 1, size[1] // scale + 1)
        smooth = gaussian_filter(grid, sigma=scale)
        noise = np.kron(smooth, np.ones((scale, scale)))
        return noise[:size[0], :size[1]]

    # Generate Perlin-like noise
    size = (500, 500)
    scale = 10000  # Adjust scale for different patterns
    perlin_noise = generate_perlin_noise(size, scale)

    noise = np.random.normal(loc=0.0, scale=5, size=(500,500))

    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)

    Z_norm = (Z - Z.min()) / (Z.max() - Z.min()) * 2 - 1

    levels = np.linspace(-1, 1, 10)  # Define discrete levels
    Z_discrete = np.digitize(Z_norm, levels) / len(levels)

    Z_combined = (Z_norm + perlin_noise) / 2
    Z_combined_norm = (Z_combined - Z_combined.min()) / (Z_combined.max() - Z_combined.min()) * 2 - 1

    cmap = mcolors.LinearSegmentedColormap.from_list("", ['red', 'blue', 'green'])
    cyclic_cmap = mcolors.LinearSegmentedColormap.from_list("", ['purple', 'cyan', 'yellow', 'purple'])
    discrete_cmap = mcolors.ListedColormap(['red', 'orange', 'yellow', 'green', 'blue', 'purple'])

    #colors = cyclic_cmap((Z_discrete + 1) / 2)  
    colors = cmap((Z_combined_norm + 1) / 2)

    plt.figure(figsize=(8, 8)) 
    plt.imshow(colors, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    wrapped_title = "\n".join(textwrap.wrap(title, width=100))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=14, color='black', pad=0)
    plt.axis('off')  # Hide axes for a cleaner look
    plt.show()