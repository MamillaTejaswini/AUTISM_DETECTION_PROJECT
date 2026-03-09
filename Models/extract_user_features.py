

import pandas as pd
import numpy as np
from scipy.stats import entropy

df = pd.read_csv("user_gaze_data.csv")

df = df.dropna(subset=["gaze_x", "gaze_y"])

x = df["gaze_x"].values
y = df["gaze_y"].values

# For real-time, assign artificial duration = 1
duration = np.ones(len(df))

# 1. Total fixations (same as training)
total_fixations = len(df)

# 2. Mean duration
mean_duration = np.mean(duration)

# 3. Std duration
std_duration = np.std(duration)

# 4. Scanpath length
distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
scanpath_length = np.sum(distances)

# 5. Mean saccade
mean_saccade = np.mean(distances) if len(distances) > 0 else 0

# 6. Dispersion (variance)
dispersion = np.var(x) + np.var(y)

# 7. Gaze entropy (20 bins like training)
heatmap, _, _ = np.histogram2d(x, y, bins=20)
heatmap = heatmap.flatten()
heatmap = heatmap[heatmap > 0]
gaze_entropy = entropy(heatmap)

print("\n=== USER FEATURES (MATCHED TO TRAINING) ===\n")
print("Total Fixations:", total_fixations)
print("Mean Duration:", mean_duration)
print("Std Duration:", std_duration)
print("Scanpath Length:", scanpath_length)
print("Mean Saccade:", mean_saccade)
print("Dispersion:", dispersion)
print("Gaze Entropy:", gaze_entropy)