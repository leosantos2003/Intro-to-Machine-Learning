import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_bars(predictions, actual_values, title, file_name):
    n_items = len(predictions)
    labels = [f'House {i+1}' for i in range(n_items)]
    x = np.arange(len(labels))  # Labels position
    width = 0.35  # Bars width

    # Creates figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))

    # Creates prediction bars and actual values
    rects_pred = ax.bar(x - width/2, predictions, width, label='Prediction', color='#3498db')
    rects_actual = ax.bar(x + width/2, actual_values, width, label='Actual Value', color='#2ecc71')

    # Adds texts, title and labels
    ax.set_ylabel('Sale Price ($)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Puts values above the bars for better visualization
    ax.bar_label(rects_pred, padding=3, fmt='${:,.0f}', rotation=45)
    ax.bar_label(rects_actual, padding=3, fmt='${:,.0f}', rotation=45)

    fig.tight_layout()
    plt.savefig(file_name)
    print(f"\nGraphic '{file_name}' succesfully saved.")