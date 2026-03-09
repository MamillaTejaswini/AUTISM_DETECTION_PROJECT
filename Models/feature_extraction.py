import numpy as np
from scipy.stats import entropy, skew, kurtosis

def extract_features_from_arrays(x, y, duration):

    if len(x) < 3:
        return None

    # ------------------------------
    # Basic Fixation Stats
    # ------------------------------

    total_fixations = len(x)

    mean_duration = np.mean(duration)
    std_duration = np.std(duration)

    duration_skew = skew(duration)
    duration_kurt = kurtosis(duration)

    long_fix_ratio = np.sum(duration > np.percentile(duration,75)) / total_fixations


    # ------------------------------
    # Scanpath Features
    # ------------------------------

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    scanpath_length = np.sum(distances)
    mean_saccade = np.mean(distances)
    std_saccade = np.std(distances)


    # ------------------------------
    # Velocity Features
    # ------------------------------

    velocity_mean = np.mean(distances)
    velocity_std = np.std(distances)
    velocity_max = np.max(distances)


    # ------------------------------
    # Spatial Dispersion
    # ------------------------------

    var_x = np.var(x)
    var_y = np.var(y)

    dispersion = var_x + var_y
    var_ratio = var_x / (var_y + 1e-6)


    # ------------------------------
    # Gaze Entropy
    # ------------------------------

    heatmap, _, _ = np.histogram2d(x, y, bins=20)
    heatmap = heatmap.flatten()
    heatmap = heatmap[heatmap > 0]

    gaze_entropy = entropy(heatmap)


    # ------------------------------
    # Fixation Density
    # ------------------------------

    fixation_density = total_fixations / (dispersion + 1e-6)
    

    # ------------------------------
    # Center Bias
    # ------------------------------

    center_x = 0.5
    center_y = 0.5

    center_dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)
    center_distance_mean = np.mean(center_dist)


    # ------------------------------
    # Gaze Instability
    # ------------------------------

    gaze_instability = np.std(distances)


    # ------------------------------
    # Spatial Exploration
    # ------------------------------

    grid_x = np.floor(x*10)
    grid_y = np.floor(y*10)

    visited = set(zip(grid_x,grid_y))
    unique_cells = len(visited)
    


    # ------------------------------
    # Path Efficiency
    # ------------------------------

    straight_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    path_efficiency = straight_distance / (scanpath_length + 1e-6)


    return [

        scanpath_length,
        mean_saccade,
        std_saccade,
        dispersion,
        gaze_entropy,

        velocity_mean,
        velocity_std,
        velocity_max,

        var_ratio,
        fixation_density,

        mean_duration,
        std_duration,
        duration_skew,
        duration_kurt,
        long_fix_ratio,

        center_distance_mean,
        gaze_instability,
        unique_cells,
        path_efficiency,
       
    ]