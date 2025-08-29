import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_mae_comparison(mae_scores, model_names, title, file_name):
    # Creates figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))

    # Creates bars
    bars = ax.bar(model_names, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'])

    # Adds texts, titles and labels
    ax.set_ylabel('Mean Absolute Error (MAE) ($)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, max(mae_scores) * 1.2) # Sets y axis limit
    plt.xticks(fontsize=11)

    # Adds exact values above bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f'${yval:,.0f}', va='bottom', ha='center', fontsize=12)

    fig.tight_layout()
    plt.savefig(file_name)
    print(f"\nGraphic '{file_name}' succesfully saved.")