import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Write name of feature to compare and filepaths to model metrics
comparison_name = 'Eigenvectors'

baseline = pd.read_csv('../results/baselines/2025-02-07_18-24-57__200_2_16_True_congestion_baseline.csv')
evects = pd.read_csv('../results/baselines/2025-02-06_11-59-11__200_2_16_True_congestion_evects.csv')

# Create figure
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle(f'Baseline Results vs No {comparison_name}')

# Plot precision
ax[0].plot(baseline['Epoch'], baseline['PrecisionNet'], label='Baseline')
ax[0].plot(evects['Epoch'], evects['PrecisionNet'], label=comparison_name)
ax[0].set_title('Precision')
ax[0].set(xlabel='Epochs')
ax[0].legend()

# Plot recall
ax[1].plot(baseline['Epoch'], baseline['RecallNet'], label='Baseline')
ax[1].plot(evects['Epoch'], evects['RecallNet'], label=comparison_name)
ax[1].set_title('Recall')
ax[1].set(xlabel='Epochs')
ax[1].legend()

# Plot f_score
ax[2].plot(baseline['Epoch'], baseline['FscoreNet'], label='Baseline')
ax[2].plot(evects['Epoch'], evects['FscoreNet'], label=comparison_name)
ax[2].set_title('F_score')
ax[2].set(xlabel='Epochs')
ax[2].legend()

# Save image
fig.savefig(f"../results/plots/{comparison_name}.png", dpi=300, bbox_inches='tight')