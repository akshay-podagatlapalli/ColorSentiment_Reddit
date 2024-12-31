import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# Load the dataset
emotion_data = pd.read_csv("output2.csv")  # Ensure it includes 'timestamp', 'type', 'score'
emotion_data['time'] = pd.to_datetime(emotion_data['time'])  # Convert to datetime

# Sort by time
emotion_data = emotion_data.sort_values('time')

# Normalize times to animation timeline
start_time = emotion_data['time'].min()
emotion_data['time_normalized'] = (emotion_data['time'] - start_time).dt.total_seconds()

# Map emotions to colors
emotion_colors = {
    'joy': 'yellow',
    'sadness': 'blue',
    'anger': 'red',
    'fear': 'purple',
    'love': 'green',
    'surprise': 'orange'
}

emotion_cmap = LinearSegmentedColormap.from_list(
    "emotion_cmap", list(emotion_colors.values())
)

# Animation parameters
num_frames = 100
max_time = emotion_data['time_normalized'].max()
time_per_frame = max_time / num_frames

# Create the animation function
fig, ax = plt.subplots(figsize=(6, 6))
X, Y = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
Z = np.zeros_like(X)

def update(frame):
    ax.clear()
    current_time = frame * time_per_frame
    
    # Filter comments up to the current time
    filtered_data = emotion_data[emotion_data['time_normalized'] <= current_time]
    
    # Create a base field for color blending
    for _, row in filtered_data.iterrows():
        x, y = np.random.uniform(0, 1, 2)  # Random position for the comment
        radius = 0.1  # Fixed radius for visual effect
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist < radius
        
        # Map the emotion to a color and add it to the field
        color_idx = list(emotion_colors.keys()).index(row['type'])
        Z[mask] = color_idx + 1  # Use color index for blending
    
    # Normalize and map to colors
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    ax.imshow(emotion_cmap(Z_norm), extent=[0, 1, 0, 1], origin='lower')
    ax.set_title(f"Time: {current_time:.2f} seconds", fontsize=14)
    ax.axis('off')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
ani.save('emotion_animation.mp4', writer='ffmpeg')
plt.show()
