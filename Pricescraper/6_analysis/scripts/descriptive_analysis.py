import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the heat pump data from CSV file"""
    csv_path = Path("data/processed/filtered_hplib_w_prices_and_decoding.csv")
    df = pd.read_csv(csv_path)
    return df

def clean_data(df):
    """Clean and prepare data for analysis"""
    # Convert numeric columns to proper types
    numeric_columns = ['Price', 'SCOP', 'Rated Power low T [kW]', 
                      'SPL outdoor high Power [dBA]', 'Max. water heating temperature [°C]']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create configuration groups
    if 'Config' in df.columns and 'Storage' in df.columns:
        df['Configuration_Group'] = df['Config'].astype(str) + ' - ' + df['Storage'].astype(str)
        
        # Define the 5 specific groups we want to analyze
        target_groups = [
            'monoblock - none',
            'monoblock - hydraulic module', 
            'monoblock - warm-water tank',
            'split - hydraulic module',
            'split - warm-water tank'
        ]
        
        # Filter to only include the target groups
        df = df[df['Configuration_Group'].isin(target_groups)].copy()
    
    return df

def print_descriptive_stats(df):
    """Print detailed descriptive statistics to terminal"""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS FOR HEAT PUMP DATA")
    print("="*80)
    
    # Basic info
    print(f"\nDataset Overview:")
    print(f"Total number of records: {len(df)}")
    print(f"Number of manufacturers: {df['Manufacturer'].nunique()}")
    
    # Price statistics
    if 'Price' in df.columns:
        price_stats = df['Price'].describe()
        price_mode = df['Price'].mode()
        print(f"\nPRICE DISTRIBUTION (EUR):")
        print(f"Count: {price_stats['count']:.0f}")
        print(f"Mean: {price_stats['mean']:.2f}")
        print(f"Median: {price_stats['50%']:.2f}")
        print(f"Mode: {price_mode.iloc[0] if len(price_mode) > 0 else 'No mode'}")
        print(f"Standard Deviation: {price_stats['std']:.2f}")
        print(f"Min: {price_stats['min']:.2f}")
        print(f"Max: {price_stats['max']:.2f}")
        print(f"25th percentile: {price_stats['25%']:.2f}")
        print(f"75th percentile: {price_stats['75%']:.2f}")
        print(f"Interquartile Range: {price_stats['75%'] - price_stats['25%']:.2f}")
    
    # SCOP statistics
    if 'SCOP' in df.columns:
        scop_stats = df['SCOP'].describe()
        scop_mode = df['SCOP'].mode()
        print(f"\nSCOP VALUES:")
        print(f"Count: {scop_stats['count']:.0f}")
        print(f"Mean: {scop_stats['mean']:.2f}")
        print(f"Median: {scop_stats['50%']:.2f}")
        print(f"Mode: {scop_mode.iloc[0] if len(scop_mode) > 0 else 'No mode'}")
        print(f"Standard Deviation: {scop_stats['std']:.2f}")
        print(f"Min: {scop_stats['min']:.2f}")
        print(f"Max: {scop_stats['max']:.2f}")
    
    # Rated Power statistics
    if 'Rated Power low T [kW]' in df.columns:
        power_stats = df['Rated Power low T [kW]'].describe()
        power_mode = df['Rated Power low T [kW]'].mode()
        print(f"\nRATED POWER LOW T (kW):")
        print(f"Count: {power_stats['count']:.0f}")
        print(f"Mean: {power_stats['mean']:.2f}")
        print(f"Median: {power_stats['50%']:.2f}")
        print(f"Mode: {power_mode.iloc[0] if len(power_mode) > 0 else 'No mode'}")
        print(f"Standard Deviation: {power_stats['std']:.2f}")
        print(f"Min: {power_stats['min']:.2f}")
        print(f"Max: {power_stats['max']:.2f}")
    
    # Noise statistics
    if 'SPL outdoor high Power [dBA]' in df.columns:
        noise_stats = df['SPL outdoor high Power [dBA]'].describe()
        noise_mode = df['SPL outdoor high Power [dBA]'].mode()
        print(f"\nNOISE LEVELS - SPL OUTDOOR HIGH POWER (dBA):")
        print(f"Count: {noise_stats['count']:.0f}")
        print(f"Mean: {noise_stats['mean']:.2f}")
        print(f"Median: {noise_stats['50%']:.2f}")
        print(f"Mode: {noise_mode.iloc[0] if len(noise_mode) > 0 else 'No mode'}")
        print(f"Standard Deviation: {noise_stats['std']:.2f}")
        print(f"Min: {noise_stats['min']:.2f}")
        print(f"Max: {noise_stats['max']:.2f}")
    
    # Max water heating temperature statistics
    if 'Max. water heating temperature [°C]' in df.columns:
        temp_stats = df['Max. water heating temperature [°C]'].describe()
        temp_mode = df['Max. water heating temperature [°C]'].mode()
        print(f"\nMAX WATER HEATING TEMPERATURE (°C):")
        print(f"Count: {temp_stats['count']:.0f}")
        print(f"Mean: {temp_stats['mean']:.2f}")
        print(f"Median: {temp_stats['50%']:.2f}")
        print(f"Mode: {temp_mode.iloc[0] if len(temp_mode) > 0 else 'No mode'}")
        print(f"Standard Deviation: {temp_stats['std']:.2f}")
        print(f"Min: {temp_stats['min']:.2f}")
        print(f"Max: {temp_stats['max']:.2f}")
    
    # Price by manufacturer
    if 'Price' in df.columns and 'Manufacturer' in df.columns:
        print(f"\nPRICE BY MANUFACTURER:")
        price_by_brand = df.groupby('Manufacturer')['Price'].agg(['count', 'mean', 'median', 'std']).round(2)
        price_by_brand = price_by_brand.sort_values('mean', ascending=False)
        print(price_by_brand.to_string())
    
    # Price by refrigerant type
    if 'Price' in df.columns and 'Refrigerant' in df.columns:
        print(f"\nPRICE BY REFRIGERANT TYPE:")
        refrigerant_counts = df['Refrigerant'].value_counts().sort_index()
        print(f"\nRecords per refrigerant type:")
        for refrigerant, count in refrigerant_counts.items():
            print(f"  {refrigerant}: {count}")
        
        price_by_refrigerant = df.groupby('Refrigerant')['Price'].agg(['count', 'mean', 'median', 'std']).round(2)
        price_by_refrigerant = price_by_refrigerant.sort_values('mean', ascending=False)
        print(f"\nPrice statistics by refrigerant type:")
        print(price_by_refrigerant.to_string())
        
        # Additional descriptive statistics for each refrigerant type
        print(f"\nDetailed price analysis by refrigerant:")
        for refrigerant in df['Refrigerant'].unique():
            if pd.notna(refrigerant):
                refrigerant_data = df[df['Refrigerant'] == refrigerant]['Price'].dropna()
                if len(refrigerant_data) > 0:
                    print(f"\n  {refrigerant}:")
                    print(f"    Count: {len(refrigerant_data)}")
                    print(f"    Mean: {refrigerant_data.mean():.2f} EUR")
                    print(f"    Median: {refrigerant_data.median():.2f} EUR")
                    print(f"    Std Dev: {refrigerant_data.std():.2f} EUR")
                    print(f"    Min: {refrigerant_data.min():.2f} EUR")
                    print(f"    Max: {refrigerant_data.max():.2f} EUR")
                    print(f"    25th percentile: {refrigerant_data.quantile(0.25):.2f} EUR")
                    print(f"    75th percentile: {refrigerant_data.quantile(0.75):.2f} EUR")
                    print(f"    IQR: {refrigerant_data.quantile(0.75) - refrigerant_data.quantile(0.25):.2f} EUR")
    
    # Price by configuration and rated power groups
    if 'Price' in df.columns and 'Configuration_Group' in df.columns and 'Rated Power low T [kW]' in df.columns:
        print(f"\nPRICE BY CONFIGURATION AND RATED POWER GROUPS:")
        print("="*100)
        
        # Create power groups
        df['Power_Group'] = pd.cut(df['Rated Power low T [kW]'], 
                                  bins=[0, 6, 9, float('inf')], 
                                  labels=['<6kW', '6-9kW', '>9kW'],
                                  include_lowest=True)
        
        # Get all configurations
        configurations = df['Configuration_Group'].unique()
        power_groups = ['<6kW', '6-9kW', '>9kW']
        
        # Create header
        header = f"{'Configuration':<30} {'<6kW':<15} {'6-9kW':<15} {'>9kW':<15}"
        print(header)
        print("-" * 100)
        
        # For each configuration, show average price and count for each power group
        for config in sorted(configurations):
            config_data = df[df['Configuration_Group'] == config]
            row = f"{config:<30}"
            
            for power_group in power_groups:
                group_data = config_data[config_data['Power_Group'] == power_group]
                if len(group_data) > 0:
                    avg_price = group_data['Price'].mean()
                    count = len(group_data)
                    row += f"{avg_price:.0f}€ (n={count})".ljust(15)
                else:
                    row += "N/A".ljust(15)
            
            print(row)
        
        # Add total row
        print("-" * 100)
        total_row = f"{'TOTAL':<30}"
        for power_group in power_groups:
            group_data = df[df['Power_Group'] == power_group]
            if len(group_data) > 0:
                avg_price = group_data['Price'].mean()
                count = len(group_data)
                total_row += f"{avg_price:.0f}€ (n={count})".ljust(15)
            else:
                total_row += "N/A".ljust(15)
        print(total_row)
        print("="*100)
    
    # SCOP by refrigerant type
    if 'SCOP' in df.columns and 'Refrigerant' in df.columns:
        print(f"\nSCOP BY REFRIGERANT TYPE:")
        
        scop_by_refrigerant = df.groupby('Refrigerant')['SCOP'].agg(['count', 'mean', 'median', 'std']).round(3)
        scop_by_refrigerant = scop_by_refrigerant.sort_values('mean', ascending=False)
        print(f"\nSCOP statistics by refrigerant type:")
        print(scop_by_refrigerant.to_string())
        
        # Additional descriptive statistics for SCOP by each refrigerant type
        print(f"\nDetailed SCOP analysis by refrigerant:")
        for refrigerant in df['Refrigerant'].unique():
            if pd.notna(refrigerant):
                refrigerant_scop_data = df[df['Refrigerant'] == refrigerant]['SCOP'].dropna()
                if len(refrigerant_scop_data) > 0:
                    print(f"\n  {refrigerant}:")
                    print(f"    Count: {len(refrigerant_scop_data)}")
                    print(f"    Mean: {refrigerant_scop_data.mean():.3f}")
                    print(f"    Median: {refrigerant_scop_data.median():.3f}")
                    print(f"    Std Dev: {refrigerant_scop_data.std():.3f}")
                    print(f"    Min: {refrigerant_scop_data.min():.3f}")
                    print(f"    Max: {refrigerant_scop_data.max():.3f}")
                    print(f"    25th percentile: {refrigerant_scop_data.quantile(0.25):.3f}")
                    print(f"    75th percentile: {refrigerant_scop_data.quantile(0.75):.3f}")
                    print(f"    IQR: {refrigerant_scop_data.quantile(0.75) - refrigerant_scop_data.quantile(0.25):.3f}")
    

    
    print("\n" + "="*80)

def create_visualizations(df, figures_dir):
    """Create and save all visualizations"""
    
    # Create figure directory if it doesn't exist
    figures_dir.mkdir(exist_ok=True)
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Price Distribution - Histogram only
    if 'Price' in df.columns:
        plt.figure(figsize=(10, 6))
        df['Price'].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Price Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Price (EUR)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. SCOP Values Distribution - Histogram only
    if 'SCOP' in df.columns:
        plt.figure(figsize=(10, 6))
        df['SCOP'].hist(bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('SCOP Values Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('SCOP')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'scop_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Rated Power Low T Distribution - Histogram only
    if 'Rated Power low T [kW]' in df.columns:
        plt.figure(figsize=(10, 6))
        df['Rated Power low T [kW]'].hist(bins=25, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Rated Power Low T Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Rated Power Low T (kW)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'rated_power_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Price by Brand - Colored Box Plot only, ordered from most to least expensive
    if 'Price' in df.columns and 'Manufacturer' in df.columns:
        # Mapping from long to short manufacturer names (handle variants/typos)
        manufacturer_map = {
            'bosch thermotechnik gmbh': 'Bosch',
            'viessmann climate solutions se': 'Viessmann',
            'ait-deutschland gmbh': 'AIT',
            'daikin europe n.v.': 'Daikin',
            'bosch thermotechnik gmbh (buderus)': 'Buderus',
            'samsung electronics air conditioner europe b.v.': 'Samsung',
            'wolf gmbh': 'Wolf',
            'mitsubishi electric air conditioning systems europe ltd': 'Mitsubishi',
            'panasonic marketing europe gmbh': 'Panasonic',
            'lg electronics inc.': 'LG',
            'johnson controls-hitachi air-conditioning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi airconditionning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi air conditioning spain': 'Johnson / Hitachi',
            'toshiba air conditioning': 'Toshiba',
            'vaillant gmbh': 'Vaillant',
        }
        # Display label mapping for correct capitalization
        display_label_map = {
            'Bosch': 'Bosch',
            'Viessmann': 'Viessmann',
            'AIT': 'AIT',
            'Daikin': 'Daikin',
            'Buderus': 'Buderus',
            'Samsung': 'Samsung',
            'Wolf': 'Wolf',
            'Mitsubishi': 'Mitsubishi',
            'Panasonic': 'Panasonic',
            'LG': 'LG',
            'Johnson / Hitachi': 'Johnson / Hitachi',
            'Toshiba': 'Toshiba',
            'Vaillant': 'Vaillant',
        }
        # Standardize manufacturer names (strip, lower)
        df = df.copy()
        df['Manufacturer_clean'] = df['Manufacturer'].astype(str).str.strip().str.lower()
        df['Manufacturer_short'] = df['Manufacturer_clean'].map(manufacturer_map).fillna(df['Manufacturer_clean'])
        # Force Johnson / Hitachi if both words are present
        df['Manufacturer_short'] = df['Manufacturer_short'].apply(lambda x: 'Johnson / Hitachi' if ('johnson' in x and 'hitachi' in x) else x)
        # Calculate mean prices and counts for all manufacturers (short names)
        mean_prices = df.groupby('Manufacturer_short')['Price'].mean().sort_values(ascending=False)
        counts = df['Manufacturer_short'].value_counts()
        all_manufacturers = mean_prices.index.tolist()  # All manufacturers ordered by mean price
        # Create green color palette (darker for expensive, lighter for cheap)
        import matplotlib.ticker as mticker
        green_palette = sns.light_palette("seagreen", n_colors=len(all_manufacturers), reverse=True)
        # Create the plot
        fig, ax = plt.subplots(figsize=(18, 10))
        box_plot = sns.boxplot(data=df, x='Manufacturer_short', y='Price', 
                              order=all_manufacturers, palette=green_palette, ax=ax)
        # Add mean price and n annotations
        for i, manufacturer in enumerate(all_manufacturers):
            avg_price = mean_prices[manufacturer]
            n = counts[manufacturer]
            # Mean price annotation (smaller font)
            ax.annotate(f"{int(round(avg_price)):,}".replace(",", "'"),
                        xy=(i, avg_price),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center', va='center', fontweight='bold', fontsize=13,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, linewidth=2))
            # n = ... annotation below the box
            y_min = df[df['Manufacturer_short'] == manufacturer]['Price'].min()
            ax.annotate(f"n = {n}",
                        xy=(i, y_min),
                        xytext=(0, -30),
                        textcoords='offset points',
                        ha='center', va='center', fontsize=11, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, linewidth=1))
        # Adjust y-limits to ensure n annotations are visible
        y_min_all = df['Price'].min()
        y_max_all = df['Price'].max()
        y_range = y_max_all - y_min_all
        ax.set_ylim([y_min_all - 0.10 * y_range, y_max_all + 0.05 * y_range])
        ax.set_title('Price Distribution by Manufacturer (Most to Least Expensive)', fontsize=22, fontweight='bold', pad=20)
        ax.set_xlabel('Manufacturer', fontsize=18, fontweight='bold')
        ax.set_ylabel('Price (EUR)', fontsize=18, fontweight='bold', labelpad=20)
        # Use display_label_map for correct capitalization
        ax.set_xticklabels([display_label_map.get(manufacturer_map.get(m, m.title()) if not (('johnson' in m and 'hitachi' in m)) else 'Johnson / Hitachi', m) for m in all_manufacturers], rotation=45, ha='right', fontsize=16, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "'")))
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'price_by_brand.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4b. Price per kW by Brand - Colored Box Plot only, ordered from most to least expensive per kW
    if 'Price' in df.columns and 'Manufacturer' in df.columns and 'Rated Power low T [kW]' in df.columns:
        # Create price per kW column
        df['Price_per_kW'] = df['Price'] / df['Rated Power low T [kW]']
        
        # Use the same manufacturer mapping as above
        manufacturer_map = {
            'bosch thermotechnik gmbh': 'Bosch',
            'viessmann climate solutions se': 'Viessmann',
            'ait-deutschland gmbh': 'AIT',
            'daikin europe n.v.': 'Daikin',
            'bosch thermotechnik gmbh (buderus)': 'Buderus',
            'samsung electronics air conditioner europe b.v.': 'Samsung',
            'wolf gmbh': 'Wolf',
            'mitsubishi electric air conditioning systems europe ltd': 'Mitsubishi',
            'panasonic marketing europe gmbh': 'Panasonic',
            'lg electronics inc.': 'LG',
            'johnson controls-hitachi air-conditioning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi airconditionning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi air conditioning spain': 'Johnson / Hitachi',
            'toshiba air conditioning': 'Toshiba',
            'vaillant gmbh': 'Vaillant',
        }
        # Display label mapping for correct capitalization
        display_label_map = {
            'Bosch': 'Bosch',
            'Viessmann': 'Viessmann',
            'AIT': 'AIT',
            'Daikin': 'Daikin',
            'Buderus': 'Buderus',
            'Samsung': 'Samsung',
            'Wolf': 'Wolf',
            'Mitsubishi': 'Mitsubishi',
            'Panasonic': 'Panasonic',
            'LG': 'LG',
            'Johnson / Hitachi': 'Johnson / Hitachi',
            'Toshiba': 'Toshiba',
            'Vaillant': 'Vaillant',
        }
        # Standardize manufacturer names (strip, lower)
        df = df.copy()
        df['Manufacturer_clean'] = df['Manufacturer'].astype(str).str.strip().str.lower()
        df['Manufacturer_short'] = df['Manufacturer_clean'].map(manufacturer_map).fillna(df['Manufacturer_clean'])
        # Force Johnson / Hitachi if both words are present
        df['Manufacturer_short'] = df['Manufacturer_short'].apply(lambda x: 'Johnson / Hitachi' if ('johnson' in x and 'hitachi' in x) else x)
        # Calculate mean price per kW and counts for all manufacturers (short names)
        mean_price_per_kw = df.groupby('Manufacturer_short')['Price_per_kW'].mean().sort_values(ascending=False)
        counts = df['Manufacturer_short'].value_counts()
        all_manufacturers = mean_price_per_kw.index.tolist()  # All manufacturers ordered by mean price per kW
        # Create green color palette (darker for expensive, lighter for cheap)
        import matplotlib.ticker as mticker
        green_palette = sns.light_palette("seagreen", n_colors=len(all_manufacturers), reverse=True)
        # Create the plot
        fig, ax = plt.subplots(figsize=(18, 10))
        box_plot = sns.boxplot(data=df, x='Manufacturer_short', y='Price_per_kW', 
                              order=all_manufacturers, palette=green_palette, ax=ax)
        # Add mean price per kW and n annotations
        for i, manufacturer in enumerate(all_manufacturers):
            avg_price_per_kw = mean_price_per_kw[manufacturer]
            n = counts[manufacturer]
            # Mean price per kW annotation (smaller font)
            ax.annotate(f"{int(round(avg_price_per_kw)):,}".replace(",", "'"),
                        xy=(i, avg_price_per_kw),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center', va='center', fontweight='bold', fontsize=13,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, linewidth=2))
            # n = ... annotation below the box
            y_min = df[df['Manufacturer_short'] == manufacturer]['Price_per_kW'].min()
            ax.annotate(f"n = {n}",
                        xy=(i, y_min),
                        xytext=(0, -30),
                        textcoords='offset points',
                        ha='center', va='center', fontsize=11, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, linewidth=1))
        # Adjust y-limits to ensure n annotations are visible
        y_min_all = df['Price_per_kW'].min()
        y_max_all = df['Price_per_kW'].max()
        y_range = y_max_all - y_min_all
        ax.set_ylim([y_min_all - 0.10 * y_range, y_max_all + 0.05 * y_range])
        ax.set_title('Price per kW Distribution by Manufacturer (Most to Least Expensive per kW)', fontsize=22, fontweight='bold', pad=20)
        ax.set_xlabel('Manufacturer', fontsize=18, fontweight='bold')
        ax.set_ylabel('Price per kW (EUR/kW)', fontsize=18, fontweight='bold', labelpad=20)
        # Use display_label_map for correct capitalization
        ax.set_xticklabels([display_label_map.get(manufacturer_map.get(m, m.title()) if not (('johnson' in m and 'hitachi' in m)) else 'Johnson / Hitachi', m) for m in all_manufacturers], rotation=45, ha='right', fontsize=16, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "'")))
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'price_per_kw_by_brand.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Noise Distribution - Histogram only
    if 'SPL outdoor high Power [dBA]' in df.columns:
        plt.figure(figsize=(10, 6))
        df['SPL outdoor high Power [dBA]'].hist(bins=25, alpha=0.7, color='red', edgecolor='black')
        plt.title('Noise Level Distribution (SPL Outdoor High Power)', fontsize=14, fontweight='bold')
        plt.xlabel('SPL Outdoor High Power (dBA)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'noise_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Max Water Heating Temperature Distribution - Histogram only
    if 'Max. water heating temperature [°C]' in df.columns:
        plt.figure(figsize=(10, 6))
        df['Max. water heating temperature [°C]'].hist(bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Max Water Heating Temperature Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Max Water Heating Temperature (°C)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'max_water_temp_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Configuration Distribution Pie Chart
    if 'Configuration_Group' in df.columns:
        import matplotlib.colors as mcolors
        config_counts = df['Configuration_Group'].value_counts()
        configs = config_counts.index.tolist()
        # Sort configs: monoblock first, then split
        monoblock_configs = [c for c in configs if c.lower().startswith('monoblock')]
        split_configs = [c for c in configs if c.lower().startswith('split')]
        sorted_configs = monoblock_configs + split_configs
        # Count R290 and R32 for each configuration
        r290_counts = df[df['Refrigerant'] == 'R290']['Configuration_Group'].value_counts()
        r32_counts = df[df['Refrigerant'] == 'R32']['Configuration_Group'].value_counts()
        n_total_list = [config_counts[config] for config in sorted_configs]
        n_r290_list = [r290_counts.get(config, 0) for config in sorted_configs]
        n_r32_list = [r32_counts.get(config, 0) for config in sorted_configs]
        labels = [f"{config}" for config in sorted_configs]
        # Color mapping
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
                # Blend with white for lightness
                color = tuple(lightness + (1 - lightness) * c for c in base_color)
                return color
            return (0.5, 0.5, 0.5)
        colors = [get_color(config) for config in sorted_configs]
        values = n_total_list
        # Custom autopct function to show n, R290, R32 (each on its own line, spaces around '=')
        def make_autopct(n_total_list, n_r290_list, n_r32_list):
            def autopct(pct):
                idx = autopct.idx
                label = f"n = {n_total_list[idx]}\nR290 = {n_r290_list[idx]}\nR32 = {n_r32_list[idx]}"
                autopct.idx += 1
                return label
            autopct.idx = 0
            return autopct
        plt.figure(figsize=(10, 8))
        plt.pie(values, labels=labels, colors=colors, autopct=make_autopct(n_total_list, n_r290_list, n_r32_list), startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Distribution of Configurations', fontsize=18, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(figures_dir / 'configuration_pie.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nAll visualizations saved to: {figures_dir}")

def plot_configuration_comparison(df, y_col, y_label, title, filename, figures_dir, color_map, lightness_map):
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from scipy import stats
    # Prepare data
    config_counts = df['Configuration_Group'].value_counts()
    configs = config_counts.index.tolist()
    # Sort configs: monoblock first, then split, but always start with 'monoblock - warm-water tank'
    monoblock_configs = [c for c in configs if c.lower().startswith('monoblock')]
    split_configs = [c for c in configs if c.lower().startswith('split')]
    sorted_configs = monoblock_configs + split_configs
    mw_tank = 'monoblock - warm-water tank'
    if mw_tank in sorted_configs:
        sorted_configs = [mw_tank] + [c for c in sorted_configs if c != mw_tank]
    # Color assignment
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
    colors = [get_color(config) for config in sorted_configs]
    # Prepare plot
    plt.figure(figsize=(14, 8))
    box = sns.boxplot(data=df, x='Configuration_Group', y=y_col, order=sorted_configs, palette=colors)
    # Annotate n and mean for each box
    y_min_all = df[y_col].min()
    y_max_all = df[y_col].max()
    y_range = y_max_all - y_min_all
    for i, config in enumerate(sorted_configs):
        n = config_counts[config]
        group_data = df[df['Configuration_Group'] == config][y_col].dropna()
        y_min = group_data.min()
        # n annotation
        box.annotate(f"n = {n}",
                     xy=(i, y_min),
                     xytext=(0, -18),
                     textcoords='offset points',
                     ha='center', va='top', fontsize=11, fontweight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, linewidth=1))
        # mean annotation
        if len(group_data) > 0:
            mean_val = group_data.mean()
            if y_col == 'Price':
                mean_str = f"{int(round(mean_val)):,}".replace(",", "'")
            else:
                mean_str = f"{mean_val:.2f}"
            box.annotate(mean_str,
                         xy=(i, mean_val),
                         xytext=(0, 0),
                         textcoords='offset points',
                         ha='center', va='center', fontweight='bold', fontsize=13,
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, linewidth=2))
    # Extend y-limits to ensure n annotation is visible
    box.set_ylim([y_min_all - 0.10 * y_range, y_max_all + 0.05 * y_range])
    # ANOVA
    group_data = [df[df['Configuration_Group'] == config][y_col].dropna() for config in sorted_configs]
    valid_groups = [g for g in group_data if len(g) > 1]
    if len(valid_groups) > 1:
        f_stat, p_value = stats.f_oneway(*valid_groups)
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        box.text(1.05, 1.08, f"ANOVA\nF = {f_stat:.2f}\n{p_str}",
                 transform=box.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, linewidth=1))
    box.set_title(title, fontsize=16, fontweight='bold', pad=20)
    box.set_xlabel('Configuration Group', fontsize=14, fontweight='bold')
    box.set_ylabel(y_label, fontsize=14, fontweight='bold', labelpad=20)
    if y_col == 'Price':
        box.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "'")))
    box.set_xticklabels(sorted_configs, rotation=25, ha='right', fontsize=13, fontweight='bold')
    box.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_configuration_comparisons(df, figures_dir):
    """Create comparative visualizations for different heat pump configurations"""
    if 'Configuration_Group' not in df.columns:
        print("Configuration groups not available for comparison plots")
        return
    import matplotlib.colors as mcolors
    color_map = {
        'monoblock': mcolors.to_rgb('#1f77b4'),  # blue
        'split': mcolors.to_rgb('#ff7f0e'),      # orange
    }
    lightness_map = {
        'none': 0.7,
        'hydraulic module': 0.5,
        'warm-water tank': 0.3,
    }
    # 1. Price Distribution by Configuration
    if 'Price' in df.columns:
        plot_configuration_comparison(
            df, 'Price', 'Price (EUR)', 'Price Distribution by Configuration Group',
            'price_by_configuration.png', figures_dir, color_map, lightness_map)
    # 2. SCOP Distribution by Configuration
    if 'SCOP' in df.columns:
        plot_configuration_comparison(
            df, 'SCOP', 'SCOP', 'SCOP Distribution by Configuration Group',
            'scop_by_configuration.png', figures_dir, color_map, lightness_map)
    # 3. Noise Level Distribution by Configuration
    if 'SPL outdoor high Power [dBA]' in df.columns:
        plot_configuration_comparison(
            df, 'SPL outdoor high Power [dBA]', 'SPL Outdoor High Power (dBA)',
            'Noise Level Distribution by Configuration Group',
            'noise_by_configuration.png', figures_dir, color_map, lightness_map)
    # 4. Max Water Temperature Distribution by Configuration
    if 'Max. water heating temperature [°C]' in df.columns:
        plot_configuration_comparison(
            df, 'Max. water heating temperature [°C]', 'Max Water Heating Temperature (°C)',
            'Max Water Heating Temperature by Configuration Group',
            'max_temp_by_configuration.png', figures_dir, color_map, lightness_map)
    # 5. Rated Power Low T Distribution by Configuration (new)
    if 'Rated Power low T [kW]' in df.columns:
        plot_configuration_comparison(
            df, 'Rated Power low T [kW]', 'Rated Power Low T (kW)',
            'Rated Power Low T Distribution by Configuration Group',
            'power_by_configuration.png', figures_dir, color_map, lightness_map)
    print(f"Configuration comparison plots saved to: {figures_dir}")

def create_refrigerant_analysis_plots(df, figures_dir):
    """Create visualizations specifically for refrigerant type analysis"""
    
    if 'Refrigerant' not in df.columns or 'Price' not in df.columns:
        print("Refrigerant or Price data not available for refrigerant analysis plots")
        return
    
    # Set consistent figure parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Price Distribution by Refrigerant Type - Box Plot
    plt.figure(figsize=(10, 6))
    # Remove any NaN values for cleaner plotting
    df_clean = df.dropna(subset=['Refrigerant', 'Price'])
    if len(df_clean) > 0:
        # Use fixed order and color palette
        refrigerant_order = ['R290', 'R32']
        color_palette = {'R290': 'firebrick', 'R32': 'steelblue'}
        counts = df_clean['Refrigerant'].value_counts()
        xtick_labels = [f"R290 (n={counts.get('R290', 0)})", f"R32 (n={counts.get('R32', 0)})"]
        ax = sns.boxplot(data=df_clean, x='Refrigerant', y='Price', order=refrigerant_order, palette=color_palette)
        # Add mean price annotation
        mean_prices = df_clean.groupby('Refrigerant')['Price'].mean()
        for i, refrigerant in enumerate(refrigerant_order):
            if refrigerant in mean_prices:
                mean_price = mean_prices[refrigerant]
                ax.annotate(f"{int(round(mean_price)):,}".replace(",", "'"),
                            xy=(i, mean_price),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center', va='center', fontweight='bold', fontsize=13,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, linewidth=2))
        # Calculate and display p-value only
        from scipy import stats
        r290_values = df_clean[df_clean['Refrigerant'] == 'R290']['Price'].dropna()
        r32_values = df_clean[df_clean['Refrigerant'] == 'R32']['Price'].dropna()
        if len(r290_values) > 0 and len(r32_values) > 0:
            u_stat, p_value = stats.mannwhitneyu(r290_values, r32_values, alternative='two-sided')
            ax.text(1.05, 1.08, f"Mann-Whitney\nU = {int(u_stat)}\np < 0.001" if p_value < 0.001 else f"Mann-Whitney\nU = {int(u_stat)}\np = {p_value:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, linewidth=1))
        ax.set_title('Price Distribution by Refrigerant Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Refrigerant Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (EUR)', fontsize=14, fontweight='bold', labelpad=20)
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "'")))
        ax.set_xticklabels(xtick_labels, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'price_by_refrigerant.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Refrigerant Type Count Distribution
    plt.figure(figsize=(8, 6))
    refrigerant_counts = df_clean['Refrigerant'].value_counts()
    
    plt.pie(refrigerant_counts.values, labels=refrigerant_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette("Set2", len(refrigerant_counts)))
    plt.title('Distribution of Refrigerant Types in Dataset', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(figures_dir / 'refrigerant_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Price by Refrigerant within Configuration Groups
    if 'Configuration_Group' in df.columns:
        configs = df_clean['Configuration_Group'].unique()
        n_configs = len(configs)
        
        fig, axes = plt.subplots(n_configs, 1, figsize=(12, 4 * n_configs))
        if n_configs == 1:
            axes = [axes]
        
        for i, config in enumerate(configs):
            config_data = df_clean[df_clean['Configuration_Group'] == config]
            
            if len(config_data) > 0 and len(config_data['Refrigerant'].unique()) > 1:
                # Get refrigerant order by mean price for this configuration
                config_refrigerant_means = config_data.groupby('Refrigerant')['Price'].mean().sort_values(ascending=False)
                config_refrigerant_order = config_refrigerant_means.index.tolist()
                
                sns.boxplot(data=config_data, x='Refrigerant', y='Price', 
                           order=config_refrigerant_order, palette='Set3', ax=axes[i])
                
                # Add mean annotations
                for j, refrigerant in enumerate(config_refrigerant_order):
                    mean_price = config_refrigerant_means[refrigerant]
                    axes[i].text(j, mean_price, f'{mean_price:.0f}€', 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                axes[i].set_title(f'Price by Refrigerant: {config}', fontweight='bold')
                axes[i].set_xlabel('Refrigerant Type')
                axes[i].set_ylabel('Price (EUR)')
                axes[i].grid(True, alpha=0.3, axis='y')
            else:
                # If only one refrigerant type, show a bar plot
                refrigerant_means = config_data.groupby('Refrigerant')['Price'].mean()
                refrigerant_means.plot(kind='bar', ax=axes[i], color='lightblue')
                axes[i].set_title(f'Average Price by Refrigerant: {config}', fontweight='bold')
                axes[i].set_xlabel('Refrigerant Type')
                axes[i].set_ylabel('Average Price (EUR)')
                axes[i].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'price_by_refrigerant_per_configuration.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. SCOP Distribution by Refrigerant Type - Box Plot
    if 'SCOP' in df.columns:
        plt.figure(figsize=(10, 6))
        # Use fixed order and color palette
        refrigerant_order = ['R290', 'R32']
        color_palette = {'R290': 'firebrick', 'R32': 'steelblue'}
        counts = df_clean['Refrigerant'].value_counts()
        xtick_labels = [f"R290 (n={counts.get('R290', 0)})", f"R32 (n={counts.get('R32', 0)})"]
        ax = sns.boxplot(data=df_clean, x='Refrigerant', y='SCOP', order=refrigerant_order, palette=color_palette)
        # Add mean SCOP annotation
        mean_scop = df_clean.groupby('Refrigerant')['SCOP'].mean()
        for i, refrigerant in enumerate(refrigerant_order):
            if refrigerant in mean_scop:
                mscop = mean_scop[refrigerant]
                ax.annotate(f"{mscop:.2f}",
                            xy=(i, mscop),
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center', va='center', fontweight='bold', fontsize=13,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, linewidth=2))
        # Calculate and display U and p-value only
        r290_values = df_clean[df_clean['Refrigerant'] == 'R290']['SCOP'].dropna()
        r32_values = df_clean[df_clean['Refrigerant'] == 'R32']['SCOP'].dropna()
        if len(r290_values) > 0 and len(r32_values) > 0:
            u_stat, p_value = stats.mannwhitneyu(r290_values, r32_values, alternative='two-sided')
            ax.text(1.05, 1.08, f"Mann-Whitney\nU = {int(u_stat)}\np < 0.001" if p_value < 0.001 else f"Mann-Whitney\nU = {int(u_stat)}\np = {p_value:.2e}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, linewidth=1))
        ax.set_title('SCOP Distribution by Refrigerant Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Refrigerant Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('SCOP', fontsize=14, fontweight='bold', labelpad=20)
        ax.set_xticklabels(xtick_labels, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'scop_by_refrigerant.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. SCOP by Refrigerant within Configuration Groups
    if 'Configuration_Group' in df.columns and 'SCOP' in df.columns:
        configs = df_clean['Configuration_Group'].unique()
        n_configs = len(configs)
        
        fig, axes = plt.subplots(n_configs, 1, figsize=(12, 4 * n_configs))
        if n_configs == 1:
            axes = [axes]
        
        for i, config in enumerate(configs):
            config_data = df_clean[df_clean['Configuration_Group'] == config]
            
            if len(config_data) > 0 and len(config_data['Refrigerant'].unique()) > 1:
                # Get refrigerant order by mean SCOP for this configuration
                config_refrigerant_scop_means = config_data.groupby('Refrigerant')['SCOP'].mean().sort_values(ascending=False)
                config_refrigerant_order = config_refrigerant_scop_means.index.tolist()
                
                sns.boxplot(data=config_data, x='Refrigerant', y='SCOP', 
                           order=config_refrigerant_order, palette='Set1', ax=axes[i])
                
                # Add mean annotations
                for j, refrigerant in enumerate(config_refrigerant_order):
                    mean_scop = config_refrigerant_scop_means[refrigerant]
                    axes[i].text(j, mean_scop, f'{mean_scop:.2f}', 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                axes[i].set_title(f'SCOP by Refrigerant: {config}', fontweight='bold')
                axes[i].set_xlabel('Refrigerant Type')
                axes[i].set_ylabel('SCOP')
                axes[i].grid(True, alpha=0.3, axis='y')
            else:
                # If only one refrigerant type, show a bar plot
                refrigerant_scop_means = config_data.groupby('Refrigerant')['SCOP'].mean()
                refrigerant_scop_means.plot(kind='bar', ax=axes[i], color='lightgreen')
                axes[i].set_title(f'Average SCOP by Refrigerant: {config}', fontweight='bold')
                axes[i].set_xlabel('Refrigerant Type')
                axes[i].set_ylabel('Average SCOP')
                axes[i].tick_params(axis='x', rotation=0)
                
                # Add value annotations on bars
                for j, v in enumerate(refrigerant_scop_means.values):
                    axes[i].text(j, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'scop_by_refrigerant_per_configuration.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Refrigerant analysis plots saved to: {figures_dir}")

def perform_anova_analysis(df):
    """Perform ANOVA tests to compare configuration groups"""
    
    if 'Configuration_Group' not in df.columns:
        print("Configuration groups not available for ANOVA analysis")
        return
    
    print("\n" + "="*80)
    print("ANOVA (ANALYSIS OF VARIANCE) RESULTS")
    print("="*80)
    print("Testing for significant differences between configuration groups")
    print("Null Hypothesis (H0): All group means are equal")
    print("Alternative Hypothesis (H1): At least one group mean is different")
    print("Significance level: α = 0.05")
    
    # Get the configuration groups
    groups = df['Configuration_Group'].unique()
    print(f"\nConfiguration groups being compared ({len(groups)}):")
    for i, group in enumerate(groups, 1):
        count = len(df[df['Configuration_Group'] == group])
        print(f"  {i}. {group} (n={count})")
    
    # Variables to test
    variables = [
        ('Price', 'EUR'),
        ('SCOP', 'dimensionless'),
        ('SPL outdoor high Power [dBA]', 'dBA'),
        ('Max. water heating temperature [°C]', '°C'),
        ('Rated Power low T [kW]', 'kW')  # <-- Added rated power
    ]
    
    for var_name, unit in variables:
        if var_name not in df.columns:
            continue
            
        print(f"\n{'='*60}")
        print(f"ANOVA TEST: {var_name} ({unit})")
        print(f"{'='*60}")
        
        # Remove missing values
        df_clean = df.dropna(subset=[var_name, 'Configuration_Group'])
        
        if len(df_clean) == 0:
            print("No valid data available for this variable")
            continue
        
        # Prepare data for each group
        group_data = []
        group_stats = []
        
        for group in groups:
            group_values = df_clean[df_clean['Configuration_Group'] == group][var_name]
            if len(group_values) > 0:
                group_data.append(group_values)
                group_stats.append({
                    'group': group,
                    'n': len(group_values),
                    'mean': group_values.mean(),
                    'std': group_values.std(),
                    'min': group_values.min(),
                    'max': group_values.max()
                })
        
        # Display group statistics
        print("\nGroup Statistics:")
        print("-" * 100)
        print(f"{'Group':<30} {'N':<5} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 100)
        for stat in group_stats:
            print(f"{stat['group']:<30} {stat['n']:<5} {stat['mean']:<10.2f} {stat['std']:<10.2f} {stat['min']:<10.2f} {stat['max']:<10.2f}")
        
        # Perform ANOVA only if we have at least 2 groups with data
        if len(group_data) < 2:
            print("Insufficient groups for ANOVA (need at least 2 groups)")
            continue
        
        # Check if all groups have at least 2 observations
        valid_groups = [group for group in group_data if len(group) >= 2]
        if len(valid_groups) < 2:
            print("Insufficient observations per group for ANOVA (need at least 2 observations per group)")
            continue
        
        # Perform one-way ANOVA
        try:
            f_statistic, p_value = stats.f_oneway(*valid_groups)
            
            print(f"\nANOVA Results:")
            print(f"F-statistic: {f_statistic:.4f}")
            print(f"p-value: {p_value:.6f}")
            
            # Interpretation
            alpha = 0.05
            print(f"\nInterpretation (α = {alpha}):")
            if p_value < alpha:
                print(f"SIGNIFICANT DIFFERENCE DETECTED")
                print(f"   p-value ({p_value:.6f}) < α ({alpha})")
                print(f"   Reject H0: There are significant differences between configuration groups")
                print(f"   At least one group has a significantly different mean {var_name.lower()}")
            else:
                print(f"NO SIGNIFICANT DIFFERENCE")
                print(f"   p-value ({p_value:.6f}) ≥ α ({alpha})")
                print(f"   Fail to reject H0: No significant differences between configuration groups")
                print(f"   All groups have statistically similar mean {var_name.lower()}")
            
            # Effect size (eta-squared)
            # Calculate total sum of squares and between-group sum of squares
            all_values = df_clean[var_name]
            grand_mean = all_values.mean()
            
            # Between-group sum of squares
            ss_between = sum([len(group) * (group.mean() - grand_mean)**2 for group in valid_groups])
            
            # Total sum of squares  
            ss_total = sum([(x - grand_mean)**2 for x in all_values])
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            print(f"\nEffect Size:")
            print(f"η² (eta-squared): {eta_squared:.4f}")
            
            if eta_squared < 0.01:
                effect_size = "negligible"
            elif eta_squared < 0.06:
                effect_size = "small"
            elif eta_squared < 0.14:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            print(f"Effect size interpretation: {effect_size}")
            print(f"({eta_squared:.1%} of variance in {var_name.lower()} explained by configuration group)")
            
        except Exception as e:
            print(f"Error performing ANOVA: {e}")
    
    print("\n" + "="*80)
    print("ANOVA ANALYSIS COMPLETE")
    print("="*80)


def perform_refrigerant_statistical_tests(df):
    """Perform statistical tests to compare refrigerant types (R290 vs R32) for price and SCOP"""
    
    if 'Refrigerant' not in df.columns:
        print("Refrigerant data not available for statistical testing")
        return
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS: REFRIGERANT TYPE COMPARISONS")
    print("="*80)
    print("Testing for significant differences between R290 and R32 refrigerants")
    print("Null Hypothesis (H0): No significant difference between refrigerant types")
    print("Alternative Hypothesis (H1): Significant difference exists between refrigerant types")
    print("Significance level: α = 0.05")
    
    # Get data for each refrigerant type
    r290_data = df[df['Refrigerant'] == 'R290']
    r32_data = df[df['Refrigerant'] == 'R32']
    
    print(f"\nSample sizes:")
    print(f"R290: {len(r290_data)} units")
    print(f"R32: {len(r32_data)} units")
    
    # Variables to test
    variables = [
        ('Price', 'EUR'),
        ('SCOP', 'dimensionless')
    ]
    
    for var_name, unit in variables:
        if var_name not in df.columns:
            continue
            
        print(f"\n{'='*70}")
        print(f"STATISTICAL TEST: {var_name} ({unit}) - R290 vs R32")
        print(f"{'='*70}")
        
        # Get clean data for each refrigerant
        r290_values = r290_data[var_name].dropna()
        r32_values = r32_data[var_name].dropna()
        
        if len(r290_values) == 0 or len(r32_values) == 0:
            print("Insufficient data for statistical testing")
            continue
        
        # Descriptive statistics
        print(f"\nDescriptive Statistics:")
        print(f"R290: n={len(r290_values)}, mean={r290_values.mean():.3f}, std={r290_values.std():.3f}")
        print(f"R32:  n={len(r32_values)}, mean={r32_values.mean():.3f}, std={r32_values.std():.3f}")
        print(f"Difference: {r290_values.mean() - r32_values.mean():.3f}")
        
        # Test for normality using Shapiro-Wilk test (for sample sizes < 5000)
        if len(r290_values) <= 5000 and len(r32_values) <= 5000:
            _, p_r290_norm = stats.shapiro(r290_values)
            _, p_r32_norm = stats.shapiro(r32_values)
            
            print(f"\nNormality Tests (Shapiro-Wilk):")
            print(f"R290: p-value = {p_r290_norm:.6f} {'(Normal)' if p_r290_norm > 0.05 else '(Non-normal)'}")
            print(f"R32:  p-value = {p_r32_norm:.6f} {'(Normal)' if p_r32_norm > 0.05 else '(Non-normal)'}")
            
            # Decide which test to use
            use_parametric = p_r290_norm > 0.05 and p_r32_norm > 0.05
        else:
            use_parametric = True  # Assume normality for large samples due to CLT
            print(f"\nLarge sample sizes - assuming normality (Central Limit Theorem)")
        
        # Test for equal variances (Levene's test)
        _, p_levene = stats.levene(r290_values, r32_values)
        equal_variances = p_levene > 0.05
        print(f"Equal variances test (Levene): p-value = {p_levene:.6f} {'(Equal)' if equal_variances else '(Unequal)'}")
        
        if use_parametric:
            # Perform t-test
            print(f"\nUsing: Independent samples t-test")
            if equal_variances:
                t_stat, p_value = stats.ttest_ind(r290_values, r32_values, equal_var=True)
                test_name = "Student's t-test"
            else:
                t_stat, p_value = stats.ttest_ind(r290_values, r32_values, equal_var=False)
                test_name = "Welch's t-test"
            
            print(f"Test: {test_name}")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_value:.6f}")
            
            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(((len(r290_values) - 1) * r290_values.var() + 
                                 (len(r32_values) - 1) * r32_values.var()) / 
                                (len(r290_values) + len(r32_values) - 2))
            cohens_d = (r290_values.mean() - r32_values.mean()) / pooled_std
            
            print(f"Effect size (Cohen's d): {cohens_d:.4f}")
            
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            print(f"Effect size interpretation: {effect_interpretation}")
            
        else:
            # Perform Mann-Whitney U test (non-parametric)
            print(f"\nUsing: Mann-Whitney U test (non-parametric)")
            u_stat, p_value = stats.mannwhitneyu(r290_values, r32_values, alternative='two-sided')
            
            print(f"U-statistic: {u_stat:.4f}")
            print(f"p-value: {p_value:.6f}")
            
            # Calculate effect size (r = Z / sqrt(N))
            z_score = stats.norm.ppf(p_value/2)  # Convert p-value to z-score
            n_total = len(r290_values) + len(r32_values)
            r_effect = abs(z_score) / np.sqrt(n_total)
            
            print(f"Effect size (r): {r_effect:.4f}")
            
            if r_effect < 0.1:
                effect_interpretation = "negligible"
            elif r_effect < 0.3:
                effect_interpretation = "small"
            elif r_effect < 0.5:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            print(f"Effect size interpretation: {effect_interpretation}")
        
        # Calculate F-statistic for variance comparison
        print(f"\nF-statistic Analysis (Variance Comparison):")
        # F-statistic = larger variance / smaller variance
        var_r290 = r290_values.var()
        var_r32 = r32_values.var()
        
        if var_r290 > var_r32:
            f_stat_variance = var_r290 / var_r32
            f_interpretation = "R290 has larger variance than R32"
        else:
            f_stat_variance = var_r32 / var_r290
            f_interpretation = "R32 has larger variance than R290"
        
        print(f"R290 variance: {var_r290:.4f}")
        print(f"R32 variance: {var_r32:.4f}")
        print(f"F-statistic (variance ratio): {f_stat_variance:.4f}")
        print(f"F-statistic interpretation: {f_interpretation}")
        
        # F-critical value for α = 0.05 (two-tailed)
        df1 = len(r290_values) - 1
        df2 = len(r32_values) - 1
        f_critical_lower = stats.f.ppf(0.025, df1, df2)  # Lower critical value
        f_critical_upper = stats.f.ppf(0.975, df1, df2)  # Upper critical value
        
        print(f"F-critical values (α = 0.05): [{f_critical_lower:.4f}, {f_critical_upper:.4f}]")
        
        if f_stat_variance < f_critical_lower or f_stat_variance > f_critical_upper:
            print(f"Variance difference: SIGNIFICANT (F-statistic outside critical range)")
        else:
            print(f"Variance difference: NOT SIGNIFICANT (F-statistic within critical range)")
        
        # Interpretation
        alpha = 0.05
        print(f"\nInterpretation (α = {alpha}):")
        if p_value < alpha:
            print(f"SIGNIFICANT DIFFERENCE DETECTED")
            print(f"   p-value ({p_value:.6f}) < α ({alpha})")
            print(f"   Reject H0: R290 and R32 have significantly different {var_name.lower()}")
            if var_name == 'Price':
                higher_type = 'R290' if r290_values.mean() > r32_values.mean() else 'R32'
                print(f"   {higher_type} systems are significantly more expensive")
            else:  # SCOP
                higher_type = 'R290' if r290_values.mean() > r32_values.mean() else 'R32'
                print(f"   {higher_type} systems have significantly higher energy efficiency")
        else:
            print(f"NO SIGNIFICANT DIFFERENCE")
            print(f"   p-value ({p_value:.6f}) ≥ α ({alpha})")
            print(f"   Fail to reject H0: No significant difference in {var_name.lower()} between R290 and R32")
    
    print("\n" + "="*80)
    print("REFRIGERANT STATISTICAL TESTING COMPLETE")
    print("="*80)


def main():
    """Main function to run the complete analysis"""
    print("Starting Heat Pump Data Analysis...")
    
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    
    print(f"Loaded {len(df)} records for analysis")
    
    # Print descriptive statistics
    print_descriptive_stats(df)
    
    # Create visualizations in a subfolder named after this analysis file
    analysis_name = Path(__file__).stem  # Gets 'descriptive_analysis' from 'descriptive_analysis.py'
    figures_dir = Path("reports/figures") / analysis_name
    create_visualizations(df, figures_dir)
    
    # Create configuration comparison plots
    create_configuration_comparisons(df, figures_dir)
    
    # Create refrigerant analysis plots
    create_refrigerant_analysis_plots(df, figures_dir)
    
    # Perform ANOVA analysis
    perform_anova_analysis(df)
    
    # Perform refrigerant statistical tests
    perform_refrigerant_statistical_tests(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 