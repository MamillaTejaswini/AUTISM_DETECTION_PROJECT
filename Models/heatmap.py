import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrices
cm_q = np.array([[30, 1],
                 [1, 27]])

cm_g = np.array([[51, 9],
                 [9, 51]])

# Labels
labels = ["Normal", "ASD"]

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    
    # Save figure (high resolution for paper)
    plt.savefig(filename, dpi=300)
    plt.show()

# Plot both matrices
plot_confusion_matrix(cm_q, 
                      "Confusion Matrix - Questionnaire Model", 
                      "confusion_matrix_questionnaire.png")

plot_confusion_matrix(cm_g, 
                      "Confusion Matrix - Eye-Gaze Model", 
                      "confusion_matrix_eyegaze.png")