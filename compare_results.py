"""
Script to compare C180, xiv, and s12 statistics between experimental values, 
MC simulations, and chain calculations.
"""

import numpy as np
import pandas as pd
import pickle
import sys
import os
from functions.data import Data_loader

def load_pickle_data(filepath):
    """Load data from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_statistics(data_dict, stat_keys):
    """
    Extract statistics (C180, s12, and xiv) from the loaded pickle data.
    
    Args:
        data_dict: Dictionary from pickle file
        stat_keys: List of statistic keys to extract
    
    Returns:
        Dictionary with mean, std, and std_err_mean for each statistic
    """
    results = {}
    
    for key, value in data_dict.items():
        df, mean_Cl, std_Cl, mean_Cor, std_Cor = value

        results[key] = {}
        
        for stat_key in stat_keys:
            if stat_key in df.columns:
                values = df[stat_key].values
                num_samples = len(values)
                std_dev = np.std(values, ddof=1) if num_samples > 1 else 0
                std_err_mean = std_dev / np.sqrt(num_samples) if num_samples > 0 else 0
                results[key][stat_key] = {
                    'mean': np.mean(values),
                    'std': std_dev,
                    'std_err_mean': std_err_mean,
                    'values': values
                }
            else:
                results[key][stat_key] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'std_err_mean': np.nan,
                    'values': []
                }
    
    return results

def compare_statistics(exp_values, mc_results, chain_results, intervals):
    """
    Compare statistics between experimental, MC, and chain calculations.
    
    Args:
        exp_values: Dictionary from data_loader.experimental_values()
        mc_results: Extracted results from MC simulations
        chain_results: Extracted results from chain calculations
        intervals: List of tuples defining angular intervals
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    stat_keys = ['C180']
    for a, b in intervals:
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        stat_keys.append(f's12_{theta_upper}_{theta_lower}')
        stat_keys.append(f'xiv_{theta_upper}_{theta_lower}')
    
    for stat_key in stat_keys:
        base_row = {'Statistic': stat_key}
        
        if stat_key in exp_values:
            base_row['Exp_Value'], base_row['Exp_Error'] = exp_values[stat_key]
        else:
            base_row['Exp_Value'], base_row['Exp_Error'] = np.nan, np.nan

        if mc_results and 'Simulation' in mc_results and stat_key in mc_results['Simulation']:
            mc_data = mc_results['Simulation'][stat_key]
            base_row['MC_Mean'] = mc_data['mean']
            base_row['MC_Std'] = mc_data['std']
            base_row['MC_Std_Err_Mean'] = mc_data['std_err_mean']
            base_row['MC_vs_Exp_Diff'] = mc_data['mean'] - base_row['Exp_Value']
            base_row['MC_vs_Exp_Sigma'] = np.sqrt((mc_data['mean'] - base_row['Exp_Value'])**2 / (base_row['Exp_Error']**2 + base_row['MC_Std']**2)) if (base_row['Exp_Error']**2 + base_row['MC_Std']**2) > 0 else np.nan
        else:
            base_row.update({'MC_Mean': np.nan, 'MC_Std': np.nan, 'MC_Std_Err_Mean': np.nan, 'MC_vs_Exp_Diff': np.nan, 'MC_vs_Exp_Sigma': np.nan})

        if chain_results:
            for chain_name, chain_data_dict in chain_results.items():
                row = base_row.copy()
                row['Chain_Name'] = chain_name
                
                if stat_key in chain_data_dict:
                    chain_data = chain_data_dict[stat_key]
                    row['Chain_Mean'] = chain_data['mean']
                    row['Chain_Std'] = chain_data['std']
                    row['Chain_Std_Err_Mean'] = chain_data['std_err_mean']
                    row['Chain_vs_Exp_Diff'] = chain_data['mean'] - row['Exp_Value']
                    row['Chain_vs_Exp_Sigma'] = np.sqrt((chain_data['mean'] - row['Exp_Value'])**2 / (row['Exp_Error']**2 + row['Chain_Std']**2)) if (row['Exp_Error']**2 + row['Chain_Std']**2) > 0 else np.nan
                    row['Chain_vs_MC_Diff'] = chain_data['mean'] - row['MC_Mean']
                else:
                    row.update({'Chain_Mean': np.nan, 'Chain_Std': np.nan, 'Chain_Std_Err_Mean': np.nan, 'Chain_vs_Exp_Diff': np.nan, 'Chain_vs_Exp_Sigma': np.nan, 'Chain_vs_MC_Diff': np.nan})
                
                comparison_data.append(row)
        else:
            row = base_row.copy()
            row.update({'Chain_Name': 'N/A', 'Chain_Mean': np.nan, 'Chain_Std': np.nan, 'Chain_Std_Err_Mean': np.nan, 'Chain_vs_Exp_Diff': np.nan, 'Chain_vs_Exp_Sigma': np.nan, 'Chain_vs_MC_Diff': np.nan})
            comparison_data.append(row)
            
    return pd.DataFrame(comparison_data)

def generate_summary_markdown(df, chain_name):
    """Generate a markdown summary for a single chain."""
    md_string = f"# Comparison Summary for {chain_name}\n\n"
    md_string += "## C180, S12 and XIV Statistics\n\n"
    
    c180_stats = df[df['Statistic'] == 'C180']
    s12_stats = df[df['Statistic'].str.startswith('s12')]
    xiv_stats = df[df['Statistic'].str.startswith('xiv')]
    
    for stat_type, subset in [('C180', c180_stats), ('S12', s12_stats), ('XIV', xiv_stats)]:
        if subset.empty:
            continue
        
        md_string += f"### {stat_type} STATISTICS\n\n"
        
        for _, row in subset.iterrows():
            md_string += f"#### {row['Statistic']}:\n"
            md_string += "```\n"
            md_string += f"  Experimental:  {row['Exp_Value']:12.6e} ± {row['Exp_Error']:10.6e}\n"
            md_string += f"  MC Simulation: {row['MC_Mean']:12.6e} (Std: {row['MC_Std']:10.6e}, Std Err Mean: {row['MC_Std_Err_Mean']:10.6e})\n"
            md_string += f"    └─ Diff from Exp: {row['MC_vs_Exp_Diff']:12.6e} ({row['MC_vs_Exp_Sigma']:+.2f}σ)\n"
            md_string += f"  Chain ({row['Chain_Name']}): {row['Chain_Mean']:12.6e} (Std: {row['Chain_Std']:10.6e}, Std Err Mean: {row['Chain_Std_Err_Mean']:10.6e})\n"
            md_string += f"    └─ Diff from Exp: {row['Chain_vs_Exp_Diff']:12.6e} ({row['Chain_vs_Exp_Sigma']:+.2f}σ)\n"
            md_string += f"    └─ Diff from MC:  {row['Chain_vs_MC_Diff']:12.6e}\n"
            md_string += "```\n\n"
            
    return md_string

def main():
    """Main function to run the comparison."""
    
    intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999), (0.5, 0.866), (0, 0.5), (-0.5, 0), (-0.866, -0.5), (-0.9999999999999999, -0.866)]
    
    mc_file = 'Simulation_1000.pkl'
    chain_file = 'chain_planck_1000.pkl'
    
    print("Loading data...")
    print(f"  - MC Simulation: {mc_file}")
    print(f"  - Chain: {chain_file}")
    
    data_loader = Data_loader()
    exp_values = data_loader.experimental_values(intervals)
    
    mc_data = load_pickle_data(mc_file)
    chain_data = load_pickle_data(chain_file)
    
    if mc_data is None or chain_data is None:
        print("Error: Could not load required data files.")
        return
    
    stat_keys = ['C180']
    for a, b in intervals:
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        stat_keys.append(f's12_{theta_upper}_{theta_lower}')
        stat_keys.append(f'xiv_{theta_upper}_{theta_lower}')
    
    mc_results = extract_statistics(mc_data, stat_keys)
    chain_results = extract_statistics(chain_data, stat_keys)
    
    comparison_df = compare_statistics(exp_values, mc_results, chain_results, intervals)
    
    output_dir = 'comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a separate markdown file for each chain
    for chain_name in comparison_df['Chain_Name'].unique():
        chain_df = comparison_df[comparison_df['Chain_Name'] == chain_name]
        markdown_content = generate_summary_markdown(chain_df, chain_name)
        
        safe_chain_name = chain_name.replace('/', '_') # Sanitize filename
        md_output_file = os.path.join(output_dir, f"{safe_chain_name}.md")
        
        with open(md_output_file, 'w') as f:
            f.write(markdown_content)
        print(f"Summary for {chain_name} saved to: {md_output_file}")
    
    output_file = 'comparison_results.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    print("\nAGGREGATE STATISTICS:")
    print("-" * 100)
    
    s12_rows = comparison_df[comparison_df['Statistic'].str.startswith('s12')]
    xiv_rows = comparison_df[comparison_df['Statistic'].str.startswith('xiv')]
    
    for name, subset in [('S12', s12_rows), ('XIV', xiv_rows)]:
        print(f"\n{name}:")
        mc_avg_sigma = subset['MC_vs_Exp_Sigma'].abs().mean()
        chain_avg_sigma = subset['Chain_vs_Exp_Sigma'].abs().mean()
        print(f"  Average |deviation| from Exp:")
        print(f"    MC: {mc_avg_sigma:.2f}σ")
        print(f"    Chain: {chain_avg_sigma:.2f}σ")

if __name__ == '__main__':
    main()
