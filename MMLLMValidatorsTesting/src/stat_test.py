import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
import os

# ⚙️ CONFIGURATION
PATH_SCRAMBLED = "/Users/miriam/projects/multimodal_validators/MMLLMValidatorsTesting/data/scrambled_processed"
PATH_CLEAN     = "/Users/miriam/projects/multimodal_validators/MMLLMValidatorsTesting/data/normal_processed"
OUTPUT_CSV     = "evaluation_analysis.csv"
OUTPUT_PLOT    = "evaluation_plot.png"

def get_dataset_stats(root_path, dataset_type):
    """
    Returns a list of dictionaries with stats for the given dataset.
    dataset_type: 'clean' (Accuracy) or 'scrambled' (Rejection Rate)
    """
    root = Path(root_path)
    stats_list = []
    
    if not root.exists():
        print(f"⚠️ Warning: Path not found for {dataset_type}: {root}")
        return []

    # Target Validity: Clean wants 'True', Scrambled wants 'False' (Error Detection)
    target_validity = 'true' if dataset_type == 'clean' else 'false'
    metric_name = "Accuracy" if dataset_type == 'clean' else "Error Detection Rate"

    # Find directories
    model_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'single' in dirnames and 'combined' in dirnames:
            model_dirs.append(Path(dirpath))

    for model_dir in model_dirs:
        try:
            # Locate CSVs
            path_s = next((model_dir / "single").rglob("clean_evaluation_results.csv"))
            path_c = next((model_dir / "combined").rglob("clean_evaluation_results.csv"))
            
            # Load
            df_s = pd.read_csv(path_s)
            df_c = pd.read_csv(path_c)
            
            # Calculate Success
            df_s['success'] = df_s['validity'].astype(str).str.lower() == target_validity
            df_c['success'] = df_c['validity'].astype(str).str.lower() == target_validity
            
            # Merge
            merged = pd.merge(
                df_s[['question_id', 'trait', 'success']],
                df_c[['question_id', 'trait', 'success']],
                on=['question_id', 'trait'],
                suffixes=('_s', '_c')
            )
            
            if len(merged) == 0: continue

            # McNemar
            s_wins = len(merged[(merged['success_s']) & (~merged['success_c'])])
            c_wins = len(merged[(~merged['success_s']) & (merged['success_c'])])
            table = [[0, s_wins], [c_wins, 0]]
            p_val = mcnemar(table, exact=True).pvalue
            
            # Logic
            single_perf = merged['success_s'].mean()
            combined_perf = merged['success_c'].mean()
            delta = single_perf - combined_perf
            
            if p_val < 0.05:
                winner = "SINGLE" if delta > 0 else "COMBINED"
            else:
                winner = "TIE"

            stats_list.append({
                "Dataset": dataset_type.title(),
                "Model": model_dir.name,
                "Provider": model_dir.parent.name,
                "Metric": metric_name,
                "Single_Score": single_perf,
                "Combined_Score": combined_perf,
                "Delta": delta,
                "P_Value": p_val,
                "Winner": winner
            })

        except Exception:
            continue

    return stats_list

def plot_results(df):
    """Generates a professional side-by-side bar chart."""
    models = df['Model'].unique()
    models.sort()
    
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    plt.subplots_adjust(wspace=0.1)
    
    metrics = {
        'Clean': ('Accuracy (Validating Truth)', axes[1]),
        'Scrambled': ('Rejection Rate (Detecting Errors)', axes[0])
    }
    
    # Color Scheme (Blue vs Orange)
    c_single = '#4C72B0'
    c_combined = '#DD8452'

    for ds_type, (title, ax) in metrics.items():
        subset = df[df['Dataset'] == ds_type].sort_values('Model')
        if subset.empty: continue
        
        x = np.arange(len(subset))
        width = 0.35
        
        # Bars
        rects1 = ax.bar(x - width/2, subset['Single_Score'], width, label='Single', color=c_single)
        rects2 = ax.bar(x + width/2, subset['Combined_Score'], width, label='Combined', color=c_combined)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(subset['Model'], rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)

        # Highlight Winner
        for i, row in enumerate(subset.itertuples()):
            if row.Winner != "TIE":
                # Add star above the winner
                target_x = x[i] - width/2 if row.Winner == "SINGLE" else x[i] + width/2
                target_h = row.Single_Score if row.Winner == "SINGLE" else row.Combined_Score
                ax.text(target_x, target_h + 0.05, "★", ha='center', color='black', fontsize=14)

    # Global Labels
    axes[0].set_ylabel('Score', fontsize=12)
    axes[1].legend(
    loc='upper left',           
    bbox_to_anchor=(1.05, 1),   
    title="Prompt Mode",
    borderaxespad=0.
)
    
    plt.suptitle('LLM Performance: Single vs Combined Prompts', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight', dpi=300)
    print(f"📊 Plot saved to: {OUTPUT_PLOT}")

def main():
    print("🚀 Processing Datasets...")
    
    # 1. Gather Data
    data_scrambled = get_dataset_stats(PATH_SCRAMBLED, 'scrambled')
    data_clean = get_dataset_stats(PATH_CLEAN, 'clean')
    
    full_data = data_scrambled + data_clean
    
    if not full_data:
        print("❌ No data found. Check paths.")
        return

    # 2. Create DataFrame
    df = pd.DataFrame(full_data)
    
    # 3. Save CSV
    # Reorder columns for readability
    cols = ['Dataset', 'Model', 'Provider', 'Metric', 'Single_Score', 'Combined_Score', 'Delta', 'P_Value', 'Winner']
    df[cols].to_csv(OUTPUT_CSV, index=False)
    print(f"💾 CSV saved to:  {OUTPUT_CSV}")

    # 4. Plot
    plot_results(df)

if __name__ == "__main__":
    main()