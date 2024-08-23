import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Read the txt file
df = pd.read_csv('bands_info.txt', sep='\t')

# Pivot the data to get a table of lanes by bands
pivot_df = df.pivot_table(index='Lane', columns='Band', values='Size', fill_value=0)

# Normalize the data
pivot_df_normalized = (pivot_df - pivot_df.mean()) / pivot_df.std()

# Compute the distance matrix
distance_matrix = sch.distance.pdist(pivot_df_normalized, metric='euclidean')

# Create the linkage matrix
linkage_matrix = sch.linkage(distance_matrix, method='ward')

# Calculate the cophenetic correlation coefficient
cophenetic_corr, coph_dists = sch.cophenet(linkage_matrix, distance_matrix)

# Save the cophenetic correlation coefficient to a text file
with open('Coefficient.txt', 'w') as file:
    file.write(f"Cophenetic Correlation Coefficient: {cophenetic_corr}\n")

# Create the full distance matrix
distance_matrix_square = sch.distance.squareform(distance_matrix)

# Convert the distance matrix to a DataFrame
distance_df = pd.DataFrame(distance_matrix_square, index=pivot_df.index, columns=pivot_df.index)

# Calculate the maximum distance
max_distance = np.max(distance_matrix)

# Calculate the similarity percentage
similarity_df = 100 * (1 - distance_df / max_distance)

# Save the similarity percentage table to a CSV file
similarity_df.to_csv('similarity_percentage.csv')

# Plot the dendrogram
fig, ax = plt.subplots(figsize=(12, 8))

# Remove only the x-axis labels
ax.xaxis.set_ticks([])
ax.xaxis.set_tick_params(width=0)  # Optional: ensures that ticks are not drawn

# Remove the axes but keep the labels
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Draw the dendrogram
dendrogram_data = sch.dendrogram(
    linkage_matrix,
    labels=pivot_df.index.to_numpy(),
    leaf_rotation=0,
    orientation='left',
    ax=ax,
    color_threshold=0,  # You can adjust this threshold
    above_threshold_color='black',  # Color for links above the threshold
)

# Create a colormap for the similarity scale
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap = plt.get_cmap('coolwarm')

# Add the colorbar to represent the similarity scale
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only for the colorbar

# Add the colorbar to the plot with adjusted position
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0)  # Adjust pad to move colorbar closer
cbar.set_label('Similarity Percentage (%)')

plt.subplots_adjust(left=0.1, right=0.3, top=0.9, bottom=0.1)

# Save the high-resolution plot
plt.savefig('dendrogram.png', dpi=600)

# Show the plot
plt.show()
