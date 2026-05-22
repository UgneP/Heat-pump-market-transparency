import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors

# --- PARAMETERS ---
GAS_BOILER_COSTS = {
    'low': 2500,    # <6kW
    'medium': 3000, # 6-9kW
    'high': 3500    # >9kW
}
GAS_PRICE = 0.11
ELECTRICITY_PRICE = 0.29
GAS_BOILER_EFFICIENCY = 0.85

HEATING_DEMAND_SCENARIOS = {
    '1_bedroom': 8000,
    '2-3_bedroom': 12000,
    '4+_bedroom': 17000
}

# --- DATA LOADING ---
def load_data():
    csv_path = Path("data/processed/filtered_hplib_w_prices_and_decoding.csv")
    df = pd.read_csv(csv_path)
    return df

def assign_gas_boiler_category(power):
    if power < 6:
        return 'low'
    elif power < 9:
        return 'medium'
    else:
        return 'high'

def assign_configuration(df):
    if 'Config' in df.columns and 'Storage' in df.columns:
        return df['Config'].astype(str) + ' - ' + df['Storage'].astype(str)
    return pd.Series(['Unknown']*len(df))

# --- MAIN ANALYSIS ---
def payback_analysis():
    df = load_data()
    # Clean numeric columns
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['SCOP'] = pd.to_numeric(df['SCOP'], errors='coerce')
    df['Rated Power low T [kW]'] = pd.to_numeric(df['Rated Power low T [kW]'], errors='coerce')
    df = df.dropna(subset=['Price', 'SCOP', 'Rated Power low T [kW]'])
    df = df[(df['Price'] > 0) & (df['SCOP'] > 0)]

    # Assign gas boiler category
    df['Gas_Boiler_Category'] = df['Rated Power low T [kW]'].apply(assign_gas_boiler_category)
    df['Gas_Boiler_Cost'] = df['Gas_Boiler_Category'].map(GAS_BOILER_COSTS)
    df['Configuration'] = assign_configuration(df)

    # Calculate total HP investment (dataset price is 60%)
    df['HP_Total_Investment'] = df['Price'] / 0.6
    
    # Calculate grant (35% of total investment, max 4500 EUR)
    df['Grant_Amount'] = np.minimum(df['HP_Total_Investment'] * 0.35, 4500)
    df['HP_Total_Investment_After_Grant'] = df['HP_Total_Investment'] - df['Grant_Amount']
    
    # Prepare results
    results = []
    for scenario, demand in HEATING_DEMAND_SCENARIOS.items():
        for idx, row in df.iterrows():
    # Gas boiler annual cost
            gas_annual_energy = demand / GAS_BOILER_EFFICIENCY
            gas_annual_cost = gas_annual_energy * GAS_PRICE

            # HP annual cost
            hp_annual_energy = demand / row['SCOP']
            hp_annual_cost = hp_annual_energy * ELECTRICITY_PRICE

            # Payback calculation
            upfront_diff = row['HP_Total_Investment_After_Grant'] - row['Gas_Boiler_Cost']
            annual_savings = gas_annual_cost - hp_annual_cost
            if annual_savings > 0:
                payback = upfront_diff / annual_savings
            else:
                payback = np.nan

            results.append({
                'Model': row.get('Model', ''),
                'Manufacturer': row.get('Manufacturer', ''),
                'Configuration': row['Configuration'],
                'Scenario': scenario,
                'Rated_Power_kW': row['Rated Power low T [kW]'],
                'SCOP': row['SCOP'],
                'HP_Total_Investment': row['HP_Total_Investment'],
                'HP_Total_Investment_After_Grant': row['HP_Total_Investment_After_Grant'],
                'Grant_Amount': row['Grant_Amount'],
                'Gas_Boiler_Cost': row['Gas_Boiler_Cost'],
                'Upfront_Diff': upfront_diff,
                'Annual_Savings': annual_savings,
                'Payback_Years': payback
            })
    result_df = pd.DataFrame(results)

    # Output average payback times for every scenario and configuration
    print("Average payback times (years) by scenario and configuration:")
    summary = result_df.groupby(['Scenario', 'Configuration'])['Payback_Years'].mean().unstack()
    print(summary.round(2))
    
    # Show example calculations for each configuration and scenario
    print("\n" + "="*100)
    print("EXAMPLE CALCULATIONS FOR EACH CONFIGURATION AND HEATING SCENARIO")
    print("="*100)
    
    # Define the 5 target configurations
    target_configs = [
        'monoblock - none',
        'monoblock - hydraulic module', 
        'monoblock - warm-water tank',
        'split - hydraulic module',
        'split - warm-water tank'
    ]
    
    scenarios = ['1_bedroom', '2-3_bedroom', '4+_bedroom']
    scenario_names = ['1 Bedroom (8,000 kWh/year)', '2-3 Bedroom (12,000 kWh/year)', '4+ Bedroom (17,000 kWh/year)']
    
    for scenario, scenario_name in zip(scenarios, scenario_names):
        print(f"\n{scenario_name.upper()}")
        print("-" * 80)
        
        for config in target_configs:
            print(f"\n{config.upper()}:")
            
            # Find a representative heat pump for this configuration and scenario
            config_data = result_df[(result_df['Scenario'] == scenario) & 
                                  (result_df['Configuration'] == config)]
            
            if len(config_data) > 0:
                # Get the heat pump with median payback time for this group
                median_payback = config_data['Payback_Years'].median()
                if pd.isna(median_payback):
                    # If median is NaN, get the first available
                    example = config_data.iloc[0]
                else:
                    # Find the heat pump closest to median payback
                    config_data_clean = config_data.dropna(subset=['Payback_Years'])
                    if len(config_data_clean) > 0:
                        closest_idx = (config_data_clean['Payback_Years'] - median_payback).abs().idxmin()
                        example = config_data_clean.loc[closest_idx]
                    else:
                        example = config_data.iloc[0]
                
                print(f"  Example: {example['Manufacturer']} {example['Model']}")
                print(f"  Rated Power: {example['Rated_Power_kW']:.1f} kW")
                print(f"  SCOP: {example['SCOP']:.2f}")
                print(f"  Dataset Price: {example['HP_Total_Investment'] * 0.6:.0f} EUR (60% of total)")
                print(f"  HP Total Investment: {example['HP_Total_Investment']:.0f} EUR (includes 40% installation)")
                print(f"  Grant Amount: {example['Grant_Amount']:.0f} EUR")
                print(f"  HP After Grant: {example['HP_Total_Investment_After_Grant']:.0f} EUR")
                print(f"  Gas Boiler Cost: {example['Gas_Boiler_Cost']:.0f} EUR")
                print(f"  Upfront Diff: {example['Upfront_Diff']:.0f} EUR")
                print(f"  Annual Savings: {example['Annual_Savings']:.0f} EUR")
                print(f"  Payback: {example['Payback_Years']:.1f} years")
                
                # Show group statistics
                valid_paybacks = config_data['Payback_Years'].dropna()
                if len(valid_paybacks) > 0:
                    print(f"  Group stats: n={len(config_data)}, avg={valid_paybacks.mean():.1f}y, min={valid_paybacks.min():.1f}y, max={valid_paybacks.max():.1f}y")
                else:
                    print(f"  Group stats: n={len(config_data)}, no valid payback times")
            else:
                print(f"  No heat pumps available for this configuration")
    
    print("\n" + "="*100)
    
    return result_df, summary

def create_payback_plot(result_df, figures_dir):
    """Create boxplot of payback times by configuration and heating scenario"""
    
    # Set up the plotting style to match descriptive analysis
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Color scheme from descriptive analysis
    color_map = {
        'monoblock': mcolors.to_rgb('#1f77b4'),  # blue
        'split': mcolors.to_rgb('#ff7f0e'),      # orange
    }
    lightness_map = {
        'none': 0.7,
        'hydraulic module': 0.5,
        'warm-water tank': 0.3,
    }
    
    def get_color(config):
        if ' - ' in config:
            base, subtype = config.split(' - ', 1)
            base = base.strip().lower()
            subtype = subtype.strip().lower()
            base_color = color_map.get(base, (0.5, 0.5, 0.5))
            lightness = lightness_map.get(subtype, 0.5)
            color = tuple(lightness + (1 - lightness) * c for c in base_color)
            return color
        return (0.5, 0.5, 0.5)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Payback Time Distribution by Configuration and Heating Scenario', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Get unique configurations and sort them
    configs = result_df['Configuration'].unique()
    monoblock_configs = [c for c in configs if 'monoblock' in c.lower()]
    split_configs = [c for c in configs if 'split' in c.lower()]
    sorted_configs = monoblock_configs + split_configs
    
    # Create plot for each scenario
    scenarios = ['1_bedroom', '2-3_bedroom', '4+_bedroom']
    scenario_labels = ['1 Bedroom\n(8\'000 kWh/year)', '2-3 Bedroom\n(12\'000 kWh/year)', '4+ Bedroom\n(17\'000 kWh/year)']
    
    for i, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
        # Filter data for this scenario
        scenario_data = result_df[result_df['Scenario'] == scenario].copy()
        
        if len(scenario_data) > 0:
            # Create boxplot
            colors = [get_color(config) for config in sorted_configs]
            valid_configs = [config for config in sorted_configs if config in scenario_data['Configuration'].unique()]
            valid_colors = [get_color(config) for config in valid_configs]
            
            if valid_configs:
                box_data = [scenario_data[scenario_data['Configuration'] == config]['Payback_Years'].dropna() 
                           for config in valid_configs]
                
                bp = axes[i].boxplot(box_data, labels=valid_configs, patch_artist=True)
                
                # Apply colors
                for patch, color in zip(bp['boxes'], valid_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add statistics annotations
                for j, config in enumerate(valid_configs):
                    config_data = scenario_data[scenario_data['Configuration'] == config]['Payback_Years'].dropna()
                    if len(config_data) > 0:
                        mean_val = config_data.mean()
                        count = len(config_data)
                        
                        # Add count and mean annotations
                        axes[i].text(j+1, mean_val, 
                                   f"n={count}\n{mean_val:.1f}y", 
                                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            axes[i].set_title(label, fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Payback Time (Years)', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Configuration', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            if valid_configs:
                axes[i].set_xticklabels(valid_configs, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits consistently across all plots
            all_payback = result_df['Payback_Years'].dropna()
            if len(all_payback) > 0:
                y_min = max(0, all_payback.quantile(0.01))
                y_max = all_payback.quantile(0.99)
                axes[i].set_ylim(y_min, y_max)
        else:
            axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(label, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Create output directory
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'payback_distribution_by_configuration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Payback distribution plot saved to: {figures_dir / 'payback_distribution_by_configuration.png'}")

if __name__ == "__main__":
    result_df, summary = payback_analysis()
    
    # Create the plot
    figures_dir = Path("reports/figures/time_to_payback_analysis")
    create_payback_plot(result_df, figures_dir) 