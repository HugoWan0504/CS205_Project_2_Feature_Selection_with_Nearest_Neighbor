import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', header=None, names=['Level', 'Features', 'Accuracy'])

# Separate final best
final_df = df[df['Level'] == 'FINAL']
plot_df = df[df['Level'] != 'FINAL'].copy()
plot_df.reset_index(drop=True, inplace=True)

plt.figure(figsize=(10, 5))

# Plot each improving best
plt.bar(range(len(plot_df)), plot_df['Accuracy'], 
        tick_label=plot_df['Features'], 
        color='skyblue', label='Improving Best')

# Highlight final best
if not final_df.empty:
    plt.bar(len(plot_df), final_df['Accuracy'].values[0],
            color='orange', label='Final Best Subset')
    all_labels = list(plot_df['Features']) + [final_df['Features'].values[0]]
    plt.xticks(range(len(all_labels)), all_labels, rotation=45)
else:
    plt.xticks(rotation=45)

plt.xlabel("Feature Subsets")
plt.ylabel("Accuracy (%)")
plt.title("Only Best Subsets Through Search")
plt.legend()
plt.tight_layout()
plt.show()
