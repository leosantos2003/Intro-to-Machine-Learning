import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_mae_vs_leaf_nodes(scores_dict):
    # Extracts keys (leaf_nodes) and values (mae) from dictionary
    leaf_nodes = list(scores_dict.keys())
    mae_values = list(scores_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_nodes, mae_values, marker='o', linestyle='-')
    
    plt.title('Model Performance vs. Tree Size', fontsize=16)
    plt.xlabel('Max Leaf Nodes', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.grid(True)
    
    # Adds notes for minimum error point
    min_mae = min(mae_values)
    best_size = leaf_nodes[mae_values.index(min_mae)]
    plt.annotate(f'Minimum MAE: ${min_mae:,.0f}\n with {best_size} leaves',
                 xy=(best_size, min_mae),
                 xytext=(best_size + 50, min_mae + 500), # Posição do texto
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.savefig('mae_vs_leaf_nodes.png')
    print(f"\nGraphic 'mae_vs_leaf_nodes.png' succesfully saved.")

def plot_prediction_comparison(predictions, actuals, title, file_name):
    n_items = len(predictions)
    labels = [f'House {i+1}' for i in range(n_items)]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects_pred = ax.bar(x - width/2, predictions, width, label='Prediction', color='#e74c3c')
    rects_actual = ax.bar(x + width/2, actuals, width, label='Real Value', color='#f1c40f')

    ax.set_ylabel('Sale Price ($)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects_pred, padding=3, fmt='${:,.0f}', rotation=45)
    ax.bar_label(rects_actual, padding=3, fmt='${:,.0f}', rotation=45)

    fig.tight_layout()
    plt.savefig(file_name)
    print(f"Graphic '{file_name}' succesfully saved.")