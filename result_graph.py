import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv('results.csv', header=None, names=['Level', 'Features', 'Accuracy'])

# Save a copy for reuse or submission
df.to_csv("results.csv", index=False) 

# Filter out all entries except improving + final attempt
plot_df = df[df['Level'] != 'FINAL'].copy()
plot_df.reset_index(drop=True, inplace=True)

plt.figure(figsize=(12, 6))

# Setup bar colors: all blue by default
colors = ['skyblue'] * len(plot_df)

# Highlight second-to-last as final best (yellow)
if len(plot_df) >= 2:
    colors[-2] = 'orange'  # second-last bar = best
    # last bar (final attempt) remains skyblue

# Plot bars
plt.bar(range(len(plot_df)), plot_df['Accuracy'],
        tick_label=plot_df['Features'], color=colors)

# Set axis and title
plt.xlabel("Feature Subsets")
plt.ylabel("Accuracy (%)")
plt.title("Only Best Subsets Through Search")
plt.xticks(rotation=45)

# Legend
from matplotlib.patches import Patch
plt.legend(handles=[
    Patch(color='skyblue', label='Trial Subsets'),
    Patch(color='orange', label='Final Best Subset')
], loc='lower left', bbox_to_anchor=(-0.05, -0.2))  # ← here


bars = plt.bar(range(len(plot_df)), plot_df['Accuracy'],
               tick_label=plot_df['Features'], color=colors)

# Annotate each bar with accuracy
for bar, acc in zip(bars, plot_df['Accuracy']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
         height - 3,  # shift down slightly inside bar
         f'{acc:.1f}%',
         ha='center', va='top', color='black', fontsize=9, weight='bold')

plt.tight_layout()
plt.show()
