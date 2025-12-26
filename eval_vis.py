import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from data_classes import EvalComparison

def plot_eval_comparison(eval_comp: EvalComparison):
    """
    Create a figure with histograms for each evaluation metric and display average values
    for multiple models in a table.
    
    Args:
        eval_comp: EvalComparison object containing evaluations from multiple models
    """
    # Flatten all results and group by model
    model_results = defaultdict(list)
    for eval_list in eval_comp.evals:
        for result in eval_list:
            model_results[result.model_name].append(result)
    
    # Get unique model names and metrics
    model_names = sorted(model_results.keys())
    metric_names = ['llm_judge_score', 'latency', 'cost', 'compression', 'harmonic']
    # Metrics where lower is better (reverse color scale)
    # lower_is_better = ['latency', 'cost', 'compression']
    lower_is_better = []

    # Helper function to safely calculate mean, filtering out None values
    def safe_mean(values):
        """Calculate mean of values, filtering out None"""
        filtered = [v for v in values if v is not None]
        if len(filtered) == 0:
            return None
        return np.mean(filtered)
    
    # Calculate averages for each model and metric
    averages_table = {}
    for model_name in model_names:
        results = model_results[model_name]
        averages_table[model_name] = {
            'llm_judge_score': safe_mean([r.llm_judge_score for r in results]),
            'latency': safe_mean([r.latency for r in results]),
            'cost': safe_mean([r.cost for r in results]),
            'compression': safe_mean([r.compression for r in results]),
            'harmonic': safe_mean([r.harmonic for r in results]),
        }
    
    # Create figure with table and histograms
    # Layout: table at top, then 3x2 grid for 6 metrics
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 2, 2, 2], hspace=0.3, wspace=0.3)
    
    # Create table (spans both columns in first row)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Prepare table data with appropriate formatting
    table_data = []
    for model_name in model_names:
        row = [model_name]
        for metric in metric_names:
            value = averages_table[model_name][metric]
            if value is None:
                row.append("N/A")
            elif metric in ['latency', 'cost']:
                # Format latency and cost with more precision if needed
                row.append(f"{value:.6f}")
            else:
                row.append(f"{value:.4f}")
        table_data.append(row)
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Model'] + [m.upper() for m in metric_names],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(2, 3.5)
    
    # Calculate min and max for each metric column for normalization
    metric_mins = {}
    metric_maxs = {}
    for metric in metric_names:
        values = [averages_table[model_name][metric] for model_name in model_names 
                 if averages_table[model_name][metric] is not None]
        if len(values) > 0:
            metric_mins[metric] = min(values)
            metric_maxs[metric] = max(values)
        else:
            metric_mins[metric] = 0
            metric_maxs[metric] = 1
    
    # Function to convert normalized value (0-1) to red-green color
    def value_to_color(normalized_value):
        """Convert normalized value (0-1) to color from red (0) to green (1)"""
        # Clamp to [0, 1]
        normalized_value = max(0, min(1, normalized_value))
        # Red to green: (1, 0, 0) to (0, 1, 0)
        red = 1 - normalized_value
        green = normalized_value
        blue = 0
        return (red, green, blue)
    
    # Style table header
    for i in range(len(metric_names) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style table cells with color gradient
    for i in range(1, len(model_names) + 1):
        row_idx = i
        model_name = model_names[i - 1]
        
        # Model column (first column) - keep alternating background
        if row_idx % 2 == 0:
            table[(row_idx, 0)].set_facecolor('#f0f0f0')
        else:
            table[(row_idx, 0)].set_facecolor('white')
        
        # Metric columns - apply color gradient
        for j, metric in enumerate(metric_names):
            col_idx = j + 1  # +1 because first column is Model
            value = averages_table[model_name][metric]
            
            # Handle None values
            if value is None:
                table[(row_idx, col_idx)].set_facecolor('#d3d3d3')  # Light gray for N/A
                table[(row_idx, col_idx)].set_text_props(color='black')
                continue
            
            # Normalize value to 0-1 range for this metric column
            if metric_maxs[metric] == metric_mins[metric]:
                # All values are the same, use neutral color
                normalized = 0.5
            else:
                normalized = (value - metric_mins[metric]) / (metric_maxs[metric] - metric_mins[metric])
            
            # For metrics where lower is better, reverse the normalization
            if metric in lower_is_better:
                normalized = 1 - normalized
            
            # Get color based on normalized value
            color = value_to_color(normalized)
            table[(row_idx, col_idx)].set_facecolor(color)
            
            # Adjust text color for readability (dark text on light, light text on dark)
            if normalized > 0.5:
                # Green background - use dark text
                table[(row_idx, col_idx)].set_text_props(color='black', weight='bold')
            else:
                # Red background - use white text
                table[(row_idx, col_idx)].set_text_props(color='white', weight='bold')
    
    ax_table.set_title('Average Metrics by Model', fontsize=14, fontweight='bold', pad=20)
    
    # Create histogram subplots (3x2 grid for 5 metrics)
    axes = []
    for i in range(3):
        for j in range(2):
            axes.append(fig.add_subplot(gs[1 + i, j]))
    
    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Create histogram for each metric
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        # Collect all values for this metric from all models (filtering None)
        all_values = []
        labels = []
        for model_name in model_names:
            values = [getattr(r, metric_name) for r in model_results[model_name] 
                     if getattr(r, metric_name, None) is not None]
            all_values.append(values)
            labels.append(model_name)
        
        # Skip if no data for this metric
        if all(len(v) == 0 for v in all_values):
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric_name.upper()} Distribution', fontsize=13, fontweight='bold')
            continue
        
        # Create histogram with different colors for each model
        # Use alpha blending for overlapping histograms
        n_bins = 20
        all_combined = [val for values in all_values for val in values]
        if len(all_combined) > 0:
            bins = np.linspace(min(all_combined), max(all_combined), n_bins + 1)
        else:
            bins = np.linspace(0, 1, n_bins + 1)
        
        for i, (model_name, values) in enumerate(zip(model_names, all_values)):
            if len(values) > 0:
                ax.hist(values, bins=bins, alpha=0.6, 
                       color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add average lines for each model (without labels to avoid legend clutter)
        for i, model_name in enumerate(model_names):
            avg_value = averages_table[model_name][metric_name]
            if avg_value is not None:
                ax.axvline(avg_value, color=colors[i], linestyle='--', 
                          linewidth=2, alpha=0.8)
        
        # Set labels and title
        # ax.set_xlabel(metric_name.upper(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = f'{metric_name.upper()} Distribution'
        if metric_name in lower_is_better:
            title += ' (lower is better)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Evaluation Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Create a single legend for the entire figure
    # Create proxy artists for the legend (just colored rectangles, no numbers)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', alpha=0.6, 
                            label=model_name) for i, model_name in enumerate(model_names)]
    
    # Place legend at the top, just below the suptitle, centered
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(model_names), 
              bbox_to_anchor=(0.5, 0.965), frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.show()

    return fig
