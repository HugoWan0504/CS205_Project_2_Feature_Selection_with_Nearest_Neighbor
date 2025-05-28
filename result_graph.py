import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load and re-save clean CSV (optional)
csv_path = "results.csv"  # You can change this if needed
df = pd.read_csv(csv_path, header=None, names=['Level', 'Features', 'Accuracy'])
df.to_csv("results.csv", index=False)

# Filter for improving bests + final subset trial
plot_df = df[df['Level'] != 'FINAL'].copy().reset_index(drop=True)

# Create plot
fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.2), 6))

# Set up colors
colors = ['skyblue'] * len(plot_df)
if len(plot_df) >= 2:
    colors[-2] = 'orange'  # Highlight final best subset

# Draw bars
bars = ax.bar(range(len(plot_df)), plot_df['Accuracy'], color=colors)

# Add accuracy % inside the bars
for i, (bar, acc) in enumerate(zip(bars, plot_df['Accuracy'])):
    height = bar.get_height()
    try:
        acc_float = float(acc)
        ax.text(bar.get_x() + bar.get_width() / 2,
                height - 3,
                f'{acc_float:.1f}%',
                ha='center', va='top',
                fontsize=9,
                weight='bold' if i == len(plot_df) - 2 else 'normal',
                color='black')
    except ValueError:
        continue

# Determine the method used from the filename
method_type = "backward" if "backward" in csv_path.lower() else "forward"

# Determine total features (used only for backward)
if method_type == "backward":
    total_features = set(map(str, range(1, max(map(int, " ".join(df['Features']).split())) + 1)))

# Add feature labels under each bar
for i, (bar, feature_str) in enumerate(zip(bars, plot_df['Features'])):
    features = set(feature_str.strip().split())

    if method_type == "backward":
        removed = sorted(total_features - features, key=lambda x: int(x))
        if len(removed) <= 8:
            label = f"All but {' '.join(removed)}"
        else:
            label = f"{len(removed)} removed"
    else:
        # forward selection: show selected features directly
        label = " ".join(sorted(features, key=lambda x: int(x)))

    ax.text(bar.get_x() + bar.get_width() / 2,
            -5,
            label,
            ha='center', va='top',
            fontsize=8, rotation=15)


def format_features(features, row_len=5):
    # Break into lines of 'row_len' features per row
    return "\n".join(" ".join(str(f) for f in features[i:i+row_len]) 
                     for i in range(0, len(features), row_len))

# Generate x-axis labels
labels = [format_features(fs) for fs in df['Features']]

# Hide x-axis ticks and labels
ax.set_xticks([])

ax.set_xlabel("Feature Subsets (shown below each bar)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Only Best Subsets Through Search")

# Legend
plt.legend(handles=[
    Patch(color='skyblue', label='Trial Subsets'),
    Patch(color='orange', label='Final Best Subset')
], loc='upper left', bbox_to_anchor=(0.01, 1.2))


# Adjust layout for spacing
plt.subplots_adjust(bottom=0.20)
plt.tight_layout(pad=4.0)

# Show plot
plt.show()
