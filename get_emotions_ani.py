# Required imports
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

# Load the emotion data
emotion_data = pd.read_csv("data/output2.csv")

# Define a mapping of emotion types to specific colors
emotion_colors = {
    'joy': 'yellow',
    'sadness': 'blue',
    'anger': 'red',
    'fear': 'purple',
    'love': 'green',
    'surprise': 'orange'
}

# Process the data by title
scores_dict = {}
emotion_types_dict = {}
timestamps_dict = {}

for title, group in emotion_data.groupby('title'):
    scores = group['score'].to_numpy()
    emotion_types = group['type'].to_numpy()
    timestamps = pd.to_datetime(group['time']).to_numpy()  # Assuming 'time' is the timestamp column
    scores_dict[title] = scores
    emotion_types_dict[title] = emotion_types
    timestamps_dict[title] = timestamps

# Define figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Set up the animation variables
x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x, y)

# Initialize emotion slices and colors
emotion_slice_width = 1 / 6  # 6 slices, each covering 1/6th of the frame
emotion_order = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']  # Order of emotions

# Function to update the frame
def update(frame):
    ax.clear()  # Clear previous frame
    
    # Loop through each post (assuming multiple posts in scores_dict)
    for title, scores in scores_dict.items():
        emotion_types = emotion_types_dict[title]
        timestamps = timestamps_dict[title]

        # Select data for the current time frame
        current_scores = scores[:frame + 1]
        current_emotions = emotion_types[:frame + 1]
        
        # Interpolate to create a smooth transition (mapping time)
        interpolated_scores = np.interp(
            np.linspace(0, len(current_scores) - 1, 500), np.arange(len(current_scores)), current_scores
        )

        # Prepare an empty frame (6 slices of 1/6th each)
        frame_data = np.zeros((500, 500, 3))  # Empty frame with 3 color channels

        # Assign colors to each slice based on emotion intensity
        for i, emotion in enumerate(emotion_order):
            emotion_score = interpolated_scores[i]  # Get current emotion score
            emotion_color = np.array(mcolors.to_rgb(emotion_colors[emotion]))  # Convert to NumPy array

            # Calculate the "spill" of each emotion into its neighboring slices
            start_col = int(i * emotion_slice_width * 500)
            end_col = int((i + 1) * emotion_slice_width * 500)

            # Fill the slice with the emotion color
            frame_data[:, start_col:end_col, :] = emotion_color * emotion_score

            # Spill the emotion color into neighboring slices based on intensity
            if i > 0:  # Spill to the left neighbor
                spill_amount = 0.2 * emotion_score  # Adjust spill amount
                frame_data[:, start_col - int(spill_amount):start_col, :] += emotion_color * spill_amount

            if i < len(emotion_order) - 1:  # Spill to the right neighbor
                spill_amount = 0.2 * emotion_score  # Adjust spill amount
                frame_data[:, end_col:end_col + int(spill_amount), :] += emotion_color * spill_amount

        # Plot the result
        ax.imshow(frame_data, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
        
        # Title wrapping for better readability
        wrapped_title = "\n".join(textwrap.wrap(title, width=100))
        ax.set_title(wrapped_title, fontsize=14, color='black', pad=10)
        ax.axis('off')  # Hide axes for a cleaner look

# Create the animation
ani = FuncAnimation(fig, update, frames=500, interval=200, repeat=False)

# Show the animation
plt.tight_layout()
plt.show()
