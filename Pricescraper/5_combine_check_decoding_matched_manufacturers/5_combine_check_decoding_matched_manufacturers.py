import pandas as pd
import os
import glob
import re

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to reach PricescraperCCDS root
matched_dir = os.path.join(project_root, 'data', 'processed', 'matched_manufacturers')
interim_dir = os.path.join(project_root, 'data', 'interim')
processed_dir = os.path.join(project_root, 'data', 'processed')
external_dir = os.path.join(project_root, 'data', 'external')
filtered_hplib_path = os.path.join(interim_dir, 'filtered_hplib.csv')
model_decoding_path = os.path.join(external_dir, 'model_decoding.csv')

def normalize_string(s):
    """Normalize strings for matching by removing extra spaces and converting to lowercase"""
    if pd.isna(s):
        return ""
    # Remove leading/trailing spaces, convert multiple spaces to single space
    normalized = str(s).strip()
    # Replace multiple consecutive spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def main():
    print(f"Script directory: {script_dir}")
    print(f"Matched directory: {matched_dir}")
    print(f"Interim directory: {interim_dir}")
    print(f"Processed directory: {processed_dir}")
    print(f"External directory: {external_dir}")
    print(f"Filtered HPLIB path: {filtered_hplib_path}")
    print(f"Model decoding path: {model_decoding_path}")
    
    # Create output directories if they don't exist
    for directory in [interim_dir, processed_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    # Step 1: Read filtered_hplib.csv first to use for validation
    try:
        filtered_hplib_df = pd.read_csv(filtered_hplib_path)
        print(f"Read filtered_hplib.csv with {len(filtered_hplib_df)} rows")
        
        # Get all valid titles from filtered_hplib.csv
        valid_titles = set(filtered_hplib_df['Titel'].dropna().unique())
        print(f"Found {len(valid_titles)} unique titles in filtered_hplib.csv")
    except Exception as e:
        print(f"Error reading filtered_hplib.csv: {e}")
        return
    
    # Step 2: Combine all _matched.csv files into one DataFrame
    all_matched_files = glob.glob(os.path.join(matched_dir, '*_matched.csv'))
    # Sort files alphabetically by manufacturer name
    all_matched_files.sort()
    print(f"Found {len(all_matched_files)} matched CSV files")
    
    # Create a dictionary to store DataFrames by manufacturer
    manufacturer_dfs = {}
    
    # Read all files and store by manufacturer
    for file in all_matched_files:
        try:
            df = pd.read_csv(file)
            # Use the first manufacturer in the file as the key
            if len(df) > 0 and 'Manufacturer' in df.columns:
                manufacturer = df['Manufacturer'].iloc[0].strip()
                manufacturer_dfs[manufacturer] = df
                print(f"Added {os.path.basename(file)} with {len(df)} rows for manufacturer: {manufacturer}")
            else:
                print(f"Warning: Empty file or missing Manufacturer column in {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {e}")
    
    # Combine DataFrames in alphabetical order by manufacturer
    sorted_manufacturers = sorted(manufacturer_dfs.keys())
    combined_df = pd.DataFrame()
    
    for manufacturer in sorted_manufacturers:
        combined_df = pd.concat([combined_df, manufacturer_dfs[manufacturer]], ignore_index=True)
        print(f"Combined data for manufacturer: {manufacturer}")
    
    # Filter out items without valid prices
    if 'Price' in combined_df.columns:
        # Check for missing or zero prices
        valid_price = combined_df['Price'].notna() & (combined_df['Price'] > 0)
        items_with_prices = combined_df[valid_price]
        items_without_valid_prices = combined_df[~valid_price]
        
        print(f"\nFound {len(items_with_prices)} items with valid prices")
        print(f"Found {len(items_without_valid_prices)} items without valid prices")
        
        # Now check if all items with prices have a Titel that exists in filtered_hplib.csv
        if 'Titel' in items_with_prices.columns:
            # Create a new column to indicate if Titel exists in filtered_hplib
            items_with_prices['valid_title'] = items_with_prices['Titel'].isin(valid_titles)
            
            # Filter items with valid titles
            valid_items = items_with_prices[items_with_prices['valid_title']].drop(columns=['valid_title'])
            invalid_items = items_with_prices[~items_with_prices['valid_title']].drop(columns=['valid_title'])
            
            print(f"\nFound {len(valid_items)} items with valid prices AND titles in filtered_hplib.csv")
            print(f"Found {len(invalid_items)} items with valid prices but titles NOT in filtered_hplib.csv")
            
            # Save valid items to filtered_hplib_w_prices.csv in interim directory
            valid_items.to_csv(os.path.join(interim_dir, 'filtered_hplib_w_prices.csv'), index=False)
            print(f"Saved {len(valid_items)} items with valid prices and valid titles to data/interim/filtered_hplib_w_prices.csv")
            
            # Save invalid items for reference in interim directory
            if len(invalid_items) > 0:
                invalid_items.to_csv(os.path.join(interim_dir, 'items_with_invalid_titles.csv'), index=False)
                print(f"Saved {len(invalid_items)} items with invalid titles to data/interim/items_with_invalid_titles.csv")
        else:
            print("Warning: No 'Titel' column found in the items with prices")
            valid_items = items_with_prices
            valid_items.to_csv(os.path.join(interim_dir, 'filtered_hplib_w_prices.csv'), index=False)
        
        # Save items without valid prices for reference in interim directory
        if len(items_without_valid_prices) > 0:
            items_without_valid_prices.to_csv(os.path.join(interim_dir, 'items_without_valid_prices.csv'), index=False)
            print(f"Saved {len(items_without_valid_prices)} items without valid prices to data/interim/items_without_valid_prices.csv")
    else:
        print("Warning: No 'Price' column found in the combined data")
        valid_items = combined_df
        valid_items.to_csv(os.path.join(interim_dir, 'filtered_hplib_w_prices.csv'), index=False)
    
    # Step 3: Find items in filtered_hplib.csv that are not in the valid items
    # Using Titel as the key column for comparison
    merged_df = filtered_hplib_df.merge(
        valid_items[['Titel']].drop_duplicates(),
        on=['Titel'],
        how='left',
        indicator=True
    )
    
    # Get items only in filtered_hplib.csv
    items_without_prices = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Save to filtered_hplib_wo_prices.csv in interim directory
    items_without_prices.to_csv(os.path.join(interim_dir, 'filtered_hplib_wo_prices.csv'), index=False)
    print(f"Saved {len(items_without_prices)} items without prices to data/interim/filtered_hplib_wo_prices.csv")
    
    # Step 4: Add model decoding (from script 6)
    print(f"\nStarting model decoding process...")
    
    # Read the model decoding file
    print(f"Reading model_decoding.csv...")
    # Try different encodings for the decoding file
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df_decoding = None
    
    for encoding in encodings_to_try:
        try:
            df_decoding = pd.read_csv(model_decoding_path, sep=';', encoding=encoding)
            print(f"Successfully read model_decoding.csv with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"Warning: model_decoding.csv not found at {model_decoding_path}")
            break
    
    if df_decoding is None:
        print("Warning: Could not read model_decoding.csv. Skipping decoding step.")
        # Save the final result without decoding
        final_output_path = os.path.join(processed_dir, 'filtered_hplib_w_prices_and_decoding.csv')
        valid_items.to_csv(final_output_path, index=False)
        print(f"Saved final result without decoding to {final_output_path}")
    else:
        # Find the position where Group column is located to insert new columns after it
        if 'Group' in valid_items.columns:
            group_index = valid_items.columns.get_loc("Group")
        else:
            # If Group column doesn't exist, add new columns at the end
            group_index = len(valid_items.columns) - 1
        
        # Create a copy of the main dataframe
        df_result = valid_items.copy()
        
        # Add the new columns after the Group column
        df_result.insert(group_index + 1, "Config", "")
        df_result.insert(group_index + 2, "Storage", "")
        df_result.insert(group_index + 3, "Storage Size (L)", "")
        
        # Create a mapping dictionary from the decoding file
        # Key: (normalized_Model, normalized_Titel), Value: (Config, Storage, Storage Size (L))
        decoding_map = {}
        for _, row in df_decoding.iterrows():
            normalized_model = normalize_string(row['Model'])
            normalized_titel = normalize_string(row['Titel'])
            key = (normalized_model, normalized_titel)
            value = (row['Config'], row['Storage'], row['Storage Size (L)'])
            decoding_map[key] = value
        
        print(f"Created decoding map with {len(decoding_map)} entries")
        
        # Show first few entries from both files for debugging
        print(f"\nFirst 3 entries from main file:")
        for i in range(min(3, len(df_result))):
            model = normalize_string(df_result.iloc[i]['Model'])
            titel = normalize_string(df_result.iloc[i]['Titel'])
            print(f"  {i+1}. Model: '{model}' | Titel: '{titel}'")
        
        print(f"\nFirst 3 entries from decoding file:")
        decoding_keys = list(decoding_map.keys())
        for i in range(min(3, len(decoding_keys))):
            model, titel = decoding_keys[i]
            print(f"  {i+1}. Model: '{model}' | Titel: '{titel}'")
        
        # Track statistics
        matched_count = 0
        unmatched_count = 0
        unmatched_items = []
        
        # Apply the decoding to each row
        for index, row in df_result.iterrows():
            model = normalize_string(row['Model'])
            titel = normalize_string(row['Titel'])
            key = (model, titel)
            
            if key in decoding_map:
                config, storage, storage_size = decoding_map[key]
                df_result.at[index, 'Config'] = config
                df_result.at[index, 'Storage'] = storage
                df_result.at[index, 'Storage Size (L)'] = storage_size
                matched_count += 1
            else:
                # Leave empty for unmatched items
                unmatched_count += 1
                unmatched_items.append(f"{model} | {titel}")
        
        # Save the final result with decoding to processed directory
        final_output_path = os.path.join(processed_dir, 'filtered_hplib_w_prices_and_decoding.csv')
        df_result.to_csv(final_output_path, index=False)
        
        # Print decoding statistics
        print(f"\nDecoding processing complete. Results saved to {final_output_path}")
        print(f"Total items processed: {len(df_result)}")
        print(f"Successfully matched: {matched_count}")
        print(f"Unmatched items: {unmatched_count}")
        
        if unmatched_count > 0:
            print(f"\nAll {unmatched_count} unmatched items:")
            for i, item in enumerate(unmatched_items):
                print(f"  {i+1}. {item}")
        
        # Print column information
        print(f"\nColumns in output file:")
        for i, col in enumerate(df_result.columns):
            print(f"  {i+1}. {col}")
    
    # Print final summary
    print(f"\nFinal Summary:")
    print(f"Total items in filtered_hplib.csv: {len(filtered_hplib_df)}")
    print(f"Items with valid prices and titles: {len(valid_items)}")
    print(f"Items without prices: {len(items_without_prices)}")
    print(f"Coverage: {len(valid_items) / len(filtered_hplib_df) * 100:.2f}%")

if __name__ == "__main__":
    main() 