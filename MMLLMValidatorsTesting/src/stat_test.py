import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
import os
from src.config import PROJECT_ROOT

# ⚙️ CONFIGURATION
PATH_SCRAMBLED = PROJECT_ROOT / "data" / "scrambled_master_results_FINAL"
PATH_CLEAN     = PROJECT_ROOT / "data" / "normal_master_results_FINAL"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "statistical_analysis"
OUTPUT_CSV     = OUTPUT_DIR / "mcnemar_analysis_summary.csv"
OUTPUT_PLOT    = OUTPUT_DIR / "mcnemar_comparison_plot.png"

def get_dataset_stats(root_path, dataset_type):
    """Identifies single/combined pairs in master folder and runs McNemar test."""
    root = Path(root_path)
    stats_list = []
    
    if not root.exists():
        print(f"⚠️ Warning: Path not found: {root}")
        return []

    target_validity = True if dataset_type == 'clean' else False
    metric_name = "Accuracy" if dataset_type == 'clean' else "Error Detection Rate"

    all_files = list(root.glob("*_MASTER.csv"))
    groups = {}
    for f in all_files:
        strategy = "single" if "single" in f.name.lower() else "combined"
        model_key = f.name.lower().replace(f"_{strategy}_master.csv", "")
        if model_key not in groups: groups[model_key] = {}
        groups[model_key][strategy] = f

    for model_key, files in groups.items():
        if "single" not in files or "combined" not in files: continue

        try:
            df_s = pd.read_csv(files["single"])
            df_c = pd.read_csv(files["combined"])
            
            def to_bool(val):
                if isinstance(val, bool): return val
                return str(val).lower() in ['true', '1', 'yes', 't']

            df_s['success'] = df_s['validity'].apply(to_bool) == target_validity
            df_c['success'] = df_c['validity'].apply(to_bool) == target_validity
            
            merged = pd.merge(df_s[['question_id', 'trait', 'success']], 
                              df_c[['question_id', 'trait', 'success']], 
                              on=['question_id', 'trait'], suffixes=('_s', '_c'))
            
            if len(merged) == 0: continue

            s_wins = len(merged[(merged['success_s'] == True) & (merged['success_c'] == False)])
            c_wins = len(merged[(merged['success_s'] == False) & (merged['success_c'] == True)])
            
            p_val = mcnemar([[0, s_wins], [c_wins, 0]], exact=True).pvalue
            delta = merged['success_s'].mean() - merged['success_c'].mean()
            winner = "TIE"
            if p_val < 0.05: winner = "SINGLE" if delta > 0 else "COMBINED"

            stats_list.append({
                "Dataset": dataset_type.title(),
                "Model": model_key.upper().replace("OPENAI_", "").replace("NEBIUS_", ""),
                "Metric": metric_name,
                "Single_Score": merged['success_s'].mean(),
                "Combined_Score": merged['success_c'].mean(),
                "P_Value": p_val,
                "Winner": winner
            })
        except Exception as e:
            print(f"❌ Error {model_key}: {e}")
    return stats_list

def plot_results(df):
    """Academic style comparison with significance brackets and asterisks."""
    if df.empty: return

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    metrics = {
        'Scrambled': ('Scrambled Data: Error Detection', axes[0]),
        'Clean': ('Normal Data: Accuracy', axes[1])
    }

    def get_asterisks(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return None

    for ds_type, (title, ax) in metrics.items():
        subset = df[df['Dataset'] == ds_type].sort_values('Model')
        if subset.empty: continue

        melted = subset.melt(id_vars=['Model', 'Winner', 'P_Value'], 
                             value_vars=['Single_Score', 'Combined_Score'],
                             var_name='Strategy', value_name='Score')
        melted['Strategy'] = melted['Strategy'].str.replace('_Score', '')

        # 1. Plot Bars
        sns.barplot(data=melted, x='Model', y='Score', hue='Strategy', 
                    ax=ax, palette="viridis", edgecolor='white')

        # 2. Add Academic Significance Brackets
        models = sorted(subset['Model'].unique())
        for i, model in enumerate(models):
            row = subset[subset['Model'] == model].iloc[0]
            stars = get_asterisks(row.P_Value)
            
            if stars:
                # Standard x-offsets for Seaborn grouped barplots
                x1, x2 = i - 0.2, i + 0.2
                # Calculate height (3% above the highest bar in the pair)
                y_max = max(row.Single_Score, row.Combined_Score)
                h = 0.02 
                y = y_max + 0.03

                # Draw Bracket: [left_tip, left_top, right_top, right_tip]
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
                
                # Place Asterisks centered above bracket
                ax.text((x1+x2)*0.5, y+h, stars, ha='center', va='bottom', 
                        color='black', fontsize=12, fontweight='bold')

        # 3. Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=30)
        ax.set_ylabel("Mean Score", fontsize=12)
        ax.set_ylim(0, 1.25) # Increase limit to fit brackets
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if ax == axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Prompt Mode", loc='upper left', bbox_to_anchor=(1, 1))

    plt.suptitle('Performance Comparison: Single vs Combined Traits (Paired McNemar Test)', 
                 fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight', dpi=300)
    print(f"✅ Academic-style plot saved: {OUTPUT_PLOT}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Starting Statistical Analysis...")
    data = get_dataset_stats(PATH_SCRAMBLED, 'scrambled') + get_dataset_stats(PATH_CLEAN, 'clean')
    
    if not data:
        print("❌ No data found.")
        return

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 CSV Summary: {OUTPUT_CSV}")
    plot_results(df)
    print("✨ Complete.")

if __name__ == "__main__":
    main()