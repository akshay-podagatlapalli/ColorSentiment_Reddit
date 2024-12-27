# getting all the required modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

sent_data = pd.read_csv("output.csv")

scores_dict = {}

for title, group in sent_data.groupby('title'):
    scores = group['result'].to_numpy()
    scores_dict[title] = scores

x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x, y)

for i in range(0, 9):
    value_array = list(scores_dict.values())[i]
    # value_array = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]

    base_field = np.interp(
        np.linspace(0, len(value_array) - 1, 500), np.arange(len(value_array)), value_array
    )

    noise = np.random.normal(loc=0.0, scale=5, size=(500,500))

    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)

    Z_norm = (Z - Z.min()) / (Z.max() - Z.min()) * 2 - 1


    cmap = mcolors.LinearSegmentedColormap.from_list("", ['red', 'blue', 'green'])
    colors = cmap((Z_norm + 1) / 2)  # Map to the colormap


    plt.imshow(colors, extent=[1, 0, 1, 0], origin='lower', aspect='auto')
    plt.axis('off')  # Hide axes for a cleaner look
    plt.show()