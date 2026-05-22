import pandas as pd
import os
from pathlib import Path
import numpy as np

def combine_and_standardize_data():
    # Create the interim folder if it doesn't exist
    output_folder = Path('data/interim')
    output_folder.mkdir(exist_ok=True)
    
    # Create a folder for manufacturer-specific files
    manufacturer_folder = output_folder / 'scraped_data_standardized_GbM'
    manufacturer_folder.mkdir(exist_ok=True)
    
    # Path to the scraped data folder
    scraped_folder = Path('data/raw/scraped_data')
    
    # List to store all dataframes
    all_dfs = []
    
    # Read each CSV file in the scraped_data folder
    for csv_file in scraped_folder.glob('*.csv'):
        try:
            print(f"Processing file: {csv_file.name}")
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract website name from filename (first word before underscore)
            website = csv_file.stem.split('_')[0]
            
            # Check if 'page' column exists and drop it if it does
            if 'page' in df.columns:
                df = df.drop('page', axis=1)
            
            # Add website column
            df['Website'] = website
            
            # Rename columns to match desired output format
            df = df.rename(columns={
                'manufacturer': 'Manufacturer',
                'type': 'Model/Type',
                'price': 'Price',
                'currency': 'Currency',
                'date_scraped': 'Date_scraped'
            })
            
            all_dfs.append(df)
            print(f"Successfully processed {csv_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    if not all_dfs:
        print("No valid data was processed. Please check your CSV files.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Reorder columns to match specified format
    column_order = ['Manufacturer', 'Model/Type', 'Price', 'Currency', 'Website', 'Date_scraped']
    combined_df = combined_df[column_order]
    
    # Define the manufacturer mappings
    manufacturer_mappings = {
        'ABN': 'ABN',
        'ARISTON': 'Ariston Thermo Group',
        'Air-to-Water': 'Air-to-Water',
        'Airwell': 'Airwell Residential S.A.S.',
        'Alpha': 'ait-deutschland GmbH',
        'Alpha Innotec heat pumps': 'ait-deutschland GmbH',
        'Autarkie': 'Autarkie',
        'BERETTA': 'Riello S.p.A.',
        'BOSCH': 'Bosch Thermotechnik GmbH',
        'Baxi': 'BAXI',
        'Berker': 'Berker',
        'Bosch': 'Bosch Thermotechnik GmbH',
        'Bosch Thermotechnik': 'Bosch Thermotechnik GmbH',
        'Brötje': 'August Brötje GmbH, BDR THERMEA FR (BRÖTJE)',
        'Buderus': 'Bosch Thermotechnik GmbH (Buderus)',
        'CLIVET': 'Clivet s.p.a.',
        'Conel': 'Conel',
        'DAIKIN': 'DAIKIN Europe N.V.',
        'Daikin': 'DAIKIN Europe N.V.',
        'Dimplex': 'Glen Dimplex Deutschland GmbH',
        'ETHERMA': 'ETHERMA',
        'Eaton': 'Eaton',
        'Erba': 'Erba',
        'FERROLI': 'Ferroli S.p.A.',
        'Galletti': 'Galletti',
        'Glen Dimplex': 'Glen Dimplex Deutschland GmbH',
        'Grant': 'Grant Engineering (Ireland) ULC, Grant Engineering (UK) Ltd',
        'Grundfos': 'Grundfos',
        'HISENSE': 'Hisense Air Conditioning Co. Ltd.',
        'Hager': 'Hager',
        'Haier': 'Qingdao Haier Air Conditioner Electric Co., Ltd.',
        'Heizung': 'Heizung',
        'Hisense': 'Hisense Air Conditioning Co. Ltd.',
        'Hitachi': 'Johnson Controls Hitachi Air-Conditioning Europe SAS, Johnson Controls Industries, Johnson Controls-Hitachi AirConditioning Spain',
        'Hyundai': 'Hyundai',
        'IMMERGAS': 'Immergas S.p.A.',
        'Johnson': 'Johnson Controls Industries',
        'Kaisai': 'Klima-Therm Sp. z o.o',
        'Kermi': 'Kermi',
        'Krone': 'Krone',
        'Kältemittelverbindungsleitung': 'Kältemittelverbindungsleitung',
        'LG': 'LG Electronics Inc.',
        'LG heat pumps': 'LG Electronics Inc.',
        'Luft-Wasser': 'Luft-Wasser',
        'MITSUBISHI ELECTRIC': 'Mitsubishi Electric Air Conditioning Systems Europe LTD',
        'Midea': 'GD Midea Air-Conditioning Equipment Co., Ltd., GD Midea Heating & Ventilating Equipment Co., Ltd.',
        'Mitsubishi': 'Mitsubishi Electric Air Conditioning Systems Europe LTD',
        'NIBE': 'Nibe AB, NIBE ENERGY SYSTEMS FRANCE',
        'NIBE heat pumps': 'Nibe AB, NIBE ENERGY SYSTEMS FRANCE',
        'Nefit Bosch heat pumps': 'Bosch Thermotechnik GmbH',
        'Novelan': 'ait-deutschland GmbH',
        'OLIMPIA SPLENDID': 'Olimpia Splendid S.p.A.',
        'OekoSolve': 'OekoSolve',
        'PANASONIC': 'Panasonic Marketing Europe GmbH',
        'PST': 'PST',
        'Panasonic': 'Panasonic Marketing Europe GmbH',
        'Reflex Winkelmann': 'Reflex Winkelmann',
        'Remeha': 'BDR THERMEA FR (REMEHA), Remeha',
        'Remeha heat pumps': 'BDR THERMEA FR (REMEHA), Remeha',
        'Remko': 'Remko',
        'Riello': 'Riello S.p.A.',
        'Roth': 'Roth',
        'S-Klima': 'S-Klima',
        'SAMSUNG': 'Samsung Electronics Air Conditioner Europe B.V.',
        'Samsung': 'Samsung Electronics Air Conditioner Europe B.V.',
        'Sinclair': 'SINCLAIR Global Group s.r.o.',
        'Stiebel Eltron': 'STIEBEL ELTRON GmbH & Co KG',
        'Stiebel Eltron heat pumps': 'STIEBEL ELTRON GmbH & Co KG',
        'Testo': 'Testo',
        'Tesy': 'Tesy',
        'ThermCube': 'ThermCube',
        'Toshiba': 'TOSHIBA AIR CONDITIONING',
        'Trianco': 'Trianco',
        'Triatherm': 'Triatherm',
        'VAILLANT': 'Vaillant GmbH',
        'Vaillant': 'Vaillant GmbH',
        'Vaillant heat pumps': 'Vaillant GmbH',
        'Viessmann': 'Viessmann Climate Solutions SE',
        'Warmflow': 'Warmflow',
        'Water': 'Water',
        'Wolf': 'WOLF GmbH',
        'Worcester': 'Bosch Thermotechnik GmbH (Buderus)',
        'Wärmepumpenpaket': 'Wärmepumpenpaket',
        'alira': 'alira',
        'alpha': 'ait-deutschland GmbH'
    }
    
    # Get unique manufacturers before standardization
    original_manufacturers = sorted(combined_df['Manufacturer'].unique())
    print("\nOriginal Manufacturers:")
    for man in original_manufacturers:
        print(f"- {man}")
    
    # Apply the manufacturer mappings
    for old_name, new_name in manufacturer_mappings.items():
        mask = combined_df['Manufacturer'].str.lower() == old_name.lower()
        combined_df.loc[mask, 'Manufacturer'] = new_name
    
    # Sort the dataframe by Manufacturer
    combined_df = combined_df.sort_values('Manufacturer')
    
    # Clean and convert Date_scraped to datetime, handling empty values
    print("\nCleaning date formats...")
    # First replace empty strings and whitespace with NaN
    combined_df['Date_scraped'] = combined_df['Date_scraped'].replace(r'^\s*$', np.nan, regex=True)
    
    # Convert to datetime with errors='coerce' to handle invalid dates by converting them to NaT
    combined_df['Date_scraped'] = pd.to_datetime(combined_df['Date_scraped'], errors='coerce')
    
    # Fill NaT values with current date
    missing_dates = combined_df['Date_scraped'].isna().sum()
    if missing_dates > 0:
        print(f"Found {missing_dates} entries with missing or invalid dates. Filling with current date.")
        combined_df['Date_scraped'] = combined_df['Date_scraped'].fillna(pd.Timestamp.now().normalize())
    
    # Remove duplicates based on Model/Type, keeping the most recent entry
    print("\nChecking for duplicate Model/Type entries...")
    before_dedup = len(combined_df)
    combined_df = combined_df.sort_values('Date_scraped', ascending=False).drop_duplicates(subset=['Model/Type'], keep='first')
    after_dedup = len(combined_df)
    duplicates_removed = before_dedup - after_dedup
    print(f"Removed {duplicates_removed} duplicate entries, keeping the most recent version of each Model/Type")
    
    # Save excluded items (prices < 1000) to a separate file
    excluded_df = combined_df[combined_df['Price'] < 1000]
    excluded_path = output_folder / 'scraped_data_excluded_low_price_items.csv'
    excluded_df.to_csv(excluded_path, index=False)
    print(f"\nExcluded items (prices < 1000) saved to: {excluded_path}")
    print(f"Number of excluded items: {len(excluded_df)}")
    
    # Filter out items with prices less than 1000
    combined_df = combined_df[combined_df['Price'] >= 1000]
    
    # Save the combined and standardized data
    output_path = output_folder / 'scraped_data_standardized.csv'
    combined_df.to_csv(output_path, index=False)
    
    # Save separate files for each manufacturer
    print("\nSaving manufacturer-specific files:")
    for manufacturer in combined_df['Manufacturer'].unique():
        # Create a safe filename from the manufacturer name
        safe_filename = "".join(c for c in manufacturer if c.isalnum() or c in (' ', '-', '_')).rstrip()
        manufacturer_file = manufacturer_folder / f"{safe_filename}.csv"
        
        # Get data for this manufacturer
        manufacturer_data = combined_df[combined_df['Manufacturer'] == manufacturer]
        
        # Save to CSV
        manufacturer_data.to_csv(manufacturer_file, index=False)
        print(f"- {manufacturer}: {len(manufacturer_data)} entries saved to {manufacturer_file.name}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY OF DATA PROCESSING")
    print("="*50)
    print(f"Total items found in scraped data: {len(all_dfs[0]) if len(all_dfs) == 1 else sum(len(df) for df in all_dfs)}")
    print(f"Items removed due to duplicates: {duplicates_removed}")
    print(f"Items removed due to low price (< 1000): {len(excluded_df)}")
    print(f"Final number of items in dataset: {len(combined_df)}")
    print("="*50)
    
    # Print detailed mapping report
    print("\nManufacturer Name Changes:")
    print("-" * 50)
    for old_name, new_name in manufacturer_mappings.items():
        count = len(combined_df[combined_df['Manufacturer'].str.lower() == old_name.lower()])
        if count > 0:
            print(f"{old_name} → {new_name} ({count} entries changed)")
    
    print(f"\nCombined and standardized data saved to: {output_path}")
    print(f"Manufacturer-specific files saved to: {manufacturer_folder}")

if __name__ == "__main__":
    combine_and_standardize_data() 