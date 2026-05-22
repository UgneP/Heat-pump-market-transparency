import pandas as pd
import os

# List of allowed manufacturers
allowed_manufacturers = [
    " ait-deutschland GmbH",
    " Bosch Thermotechnik GmbH",
    " Bosch Thermotechnik GmbH (Buderus)",
    " DAIKIN Europe N.V.",
    " Johnson Controls-Hitachi AirConditioning Spain",
    " Johnson Controls Industries",
    " Johnson Controls Hitachi Air-Conditioning Europe SAS",
    " LG Electronics Inc.",
    " Mitsubishi Electric Air Conditioning Systems Europe LTD",
    " Panasonic Marketing Europe GmbH",
    " Samsung Electronics Air Conditioner Europe B.V.",
    " TOSHIBA AIR CONDITIONING",
    " Vaillant GmbH",
    " Viessmann Climate Solutions SE",
    " WOLF GmbH",
]

# Johnson/Hitachi group for combining into a single file
johnson_hitachi_group = [
    " Johnson Controls-Hitachi AirConditioning Spain",
    " Johnson Controls Industries",
    " Johnson Controls Hitachi Air-Conditioning Europe SAS"
]

# Columns to remove
columns_to_remove = [
    "p1_P_el_h [1/°C]", "p2_P_el_h [1/°C]", "p3_P_el_h [-]", "p4_P_el_h [1/°C]",
    "p1_COP [-]", "p2_COP [-]", "p3_COP [-]", "p4_COP [-]",
    "p1_P_el_c [1/°C]", "p2_P_el_c [1/°C]", "p3_P_el_c [-]", "p4_P_el_c [1/°C]",
    "p1_EER [-]", "p2_EER [-]", "p3_EER [-]", "p4_EER [-]",
    "MAPE P_th", "MAPE P_el", "MAPE COP", "MAPE Pdc", "MAPE P_el_c", "MAPE EER"
]

def main():
    # Use the correct path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level to reach PricescraperCCDS
    
    # Input file path
    csv_path = os.path.join(project_root, 'data', 'external', 'hplib_database_newfeb25.csv')
    
    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path, sep=',')
    
    # Create a copy of the original dataframe
    removed_entries = pd.DataFrame()
    
    # Remove specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    
    # Filter by manufacturer
    manufacturer_mask = ~df['Manufacturer'].isin(allowed_manufacturers)
    removed_manufacturer = df[manufacturer_mask].copy()
    removed_manufacturer['Removal Reason'] = "Manufacturer not in allowed list"
    removed_entries = pd.concat([removed_entries, removed_manufacturer])
    df = df[~manufacturer_mask]
    
    # Filter by refrigerant
    refrigerant_mask = ~df['Refrigerant'].isin(['R32', 'R290'])
    removed_refrigerant = df[refrigerant_mask].copy()
    removed_refrigerant['Removal Reason'] = "Refrigerant not R32 or R290"
    removed_entries = pd.concat([removed_entries, removed_refrigerant])
    df = df[~refrigerant_mask]
    
    # Filter by power rating
    power_mask = (df['Rated Power low T [kW]'] > 13) | (df['Rated Power medium T [kW]'] > 13)
    removed_power = df[power_mask].copy()
    removed_power['Removal Reason'] = "Power rating exceeds 13kW"
    removed_entries = pd.concat([removed_entries, removed_power])
    df = df[~power_mask]
    
    # Remove duplicates
    duplicates_mask = df.duplicated()
    removed_duplicates = df[duplicates_mask].copy()
    removed_duplicates['Removal Reason'] = "Duplicate entry"
    removed_entries = pd.concat([removed_entries, removed_duplicates])
    df = df.drop_duplicates()
    
    # Create output directory
    output_dir = os.path.join(project_root, 'data', 'interim')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save filtered dataset
    filtered_path = os.path.join(output_dir, 'filtered_hplib.csv')
    df.to_csv(filtered_path, index=False)
    
    # Save removed entries
    removed_path = os.path.join(output_dir, 'removed_entries_hplib.csv')
    removed_entries.to_csv(removed_path, index=False)
    
    # Print summary
    print(f"Original entries: {len(df) + len(removed_entries)}")
    print(f"Filtered entries: {len(df)}")
    print(f"Removed entries: {len(removed_entries)}")
    print("\nBreakdown of removed entries:")
    print(removed_entries['Removal Reason'].value_counts())
    
    # Create directory for manufacturer-specific files
    manufacturer_output_dir = os.path.join(output_dir, 'filtered_hplib_GbM')
    os.makedirs(manufacturer_output_dir, exist_ok=True)
    
    # Add new columns between Date and Type
    cols = list(df.columns)
    date_idx = cols.index('Date')
    type_idx = cols.index('Type')
    
    # Create new columns with empty values
    new_cols = ['Model/Type', 'Price', 'Currency', 'Website', 'Date_scraped']
    for i, col in enumerate(new_cols):
        cols.insert(date_idx + 1 + i, col)
    
    # Reorder DataFrame columns and add empty values for new columns
    df_new = df.reindex(columns=cols)
    
    # Create a collection for all Johnson/Hitachi data
    johnson_hitachi_df = pd.DataFrame(columns=df_new.columns)
    
    # Group by manufacturer and save separate files
    for manufacturer in df_new['Manufacturer'].unique():
        manufacturer_df = df_new[df_new['Manufacturer'] == manufacturer]
        
        # Handle Johnson/Hitachi manufacturers as a special case
        if manufacturer in johnson_hitachi_group:
            johnson_hitachi_df = pd.concat([johnson_hitachi_df, manufacturer_df])
        else:
            # Create a safe filename by replacing problematic characters
            safe_name = manufacturer.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
            output_path = os.path.join(manufacturer_output_dir, f'{safe_name}.csv')
            manufacturer_df.to_csv(output_path, index=False)
    
    # Save Johnson/Hitachi data as a single file without duplicates
    if not johnson_hitachi_df.empty:
        # Remove duplicates from Johnson/Hitachi data
        johnson_hitachi_df = johnson_hitachi_df.drop_duplicates()
        output_path = os.path.join(manufacturer_output_dir, 'Johnson_Hitachi.csv')
        johnson_hitachi_df.to_csv(output_path, index=False)
    
    # Verify total count in manufacturer-specific files
    total_entries_in_gbm = 0
    for filename in os.listdir(manufacturer_output_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(manufacturer_output_dir, filename)
            file_df = pd.read_csv(file_path)
            total_entries_in_gbm += len(file_df)
    
    # Compare counts
    print("\nManufacturer-specific files have been created in the 'filtered_hplib_GbM' folder")
    print(f"Total entries in filtered_hplib.csv: {len(df)}")
    print(f"Total entries in manufacturer-specific files: {total_entries_in_gbm}")
    
    if total_entries_in_gbm != len(df):
        print("WARNING: There is a discrepancy in the number of entries!")
        print("Recounting and checking for duplicates across all manufacturer files...")
        
        # Collect all entries from manufacturer files
        all_entries = pd.DataFrame()
        for filename in os.listdir(manufacturer_output_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(manufacturer_output_dir, filename)
                file_df = pd.read_csv(file_path)
                all_entries = pd.concat([all_entries, file_df])
        
        # Check for duplicates across all manufacturer files
        duplicates = all_entries.duplicated()
        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            print(f"Found {duplicate_count} duplicates across manufacturer files.")
            
            # Remove all files and recreate them properly
            for filename in os.listdir(manufacturer_output_dir):
                if filename.endswith('.csv'):
                    os.remove(os.path.join(manufacturer_output_dir, filename))
            
            # Recreate manufacturer files without duplicates
            # First, extract identifier columns for checking duplicates
            id_columns = [col for col in df_new.columns if col not in new_cols]
            
            # Create a collection for all manufacturer data
            all_manufacturer_data = []
            
            for manufacturer in df_new['Manufacturer'].unique():
                manufacturer_df = df_new[df_new['Manufacturer'] == manufacturer].copy()
                
                # Create a safe filename
                if manufacturer in johnson_hitachi_group:
                    safe_name = 'Johnson_Hitachi'
                else:
                    safe_name = manufacturer.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                # Store data with manufacturer name
                all_manufacturer_data.append({
                    'name': safe_name,
                    'data': manufacturer_df
                })
            
            # Group data by manufacturer name
            grouped_data = {}
            for item in all_manufacturer_data:
                name = item['name']
                if name in grouped_data:
                    grouped_data[name] = pd.concat([grouped_data[name], item['data']])
                else:
                    grouped_data[name] = item['data']
            
            # Save deduplicated data
            for name, data in grouped_data.items():
                # Remove duplicates
                data = data.drop_duplicates(subset=id_columns)
                output_path = os.path.join(manufacturer_output_dir, f'{name}.csv')
                data.to_csv(output_path, index=False)
            
            # Recount
            total_entries_in_gbm = 0
            for filename in os.listdir(manufacturer_output_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(manufacturer_output_dir, filename)
                    file_df = pd.read_csv(file_path)
                    total_entries_in_gbm += len(file_df)
            
            print(f"After deduplication: {total_entries_in_gbm} entries in manufacturer files")
            print(f"Original filtered_hplib.csv: {len(df)} entries")
        else:
            print("No duplicates found. The discrepancy may be due to other reasons.")

if __name__ == "__main__":
    main() 