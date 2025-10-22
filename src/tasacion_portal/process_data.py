"""
Data Processing Script
Cleans and processes the scraped property data from data.csv
"""

import pandas as pd
import re
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def process_price(price_str: str) -> float:
    """Clean price column: remove 'DesdeUF' and dots"""
    if pd.isna(price_str) or price_str == '':
        return None
    # Remove 'Desde', 'UF', dots, and extra spaces
    cleaned = str(price_str).replace('Desde', '').replace('UF', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(cleaned)
    except:
        return None


def process_range_with_mean(text: str) -> float:
    """Extract mean from ranges like '2 a 4' or '2 - 4'"""
    if pd.isna(text) or text == '':
        return None

    text = str(text)

    # Look for pattern like "2 a 4" or "2-4"
    # Try to find numbers with 'a' separator
    match = re.search(r'(\d+)\s*a\s*(\d+)', text)
    if match:
        num1 = float(match.group(1))
        num2 = float(match.group(2))
        return (num1 + num2) / 2

    # Try to find numbers with '-' separator
    match = re.search(r'(\d+)\s*-\s*(\d+)', text)
    if match:
        num1 = float(match.group(1))
        num2 = float(match.group(2))
        return (num1 + num2) / 2

    # Try to extract single number
    match = re.search(r'(\d+)', text)
    if match:
        return float(match.group(1))

    return None


def process_surface(surface_str: str) -> float:
    """Clean surface column: handle ranges and remove units"""
    if pd.isna(surface_str) or surface_str == '':
        return None

    surface_str = str(surface_str)

    # Remove text like 'm²', 'útiles', 'util', 'm2', etc.
    cleaned = surface_str.replace('m²', '').replace('m2', '').replace('útiles', '').replace('útil', '').replace('util', '').strip()

    # Now process as range
    return process_range_with_mean(cleaned)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the dataframe"""
    print("\nProcessing data...")
    print(f"Original shape: {df.shape}")

    # Process price column
    if 'price' in df.columns:
        print("- Cleaning price column")
        df['price'] = df['price'].apply(process_price)

    # Process bedrooms column
    if 'bedrooms' in df.columns:
        print("- Processing bedrooms column")
        df['bedrooms'] = df['bedrooms'].apply(process_range_with_mean)

    # Process bathrooms column
    if 'bathrooms' in df.columns:
        print("- Processing bathrooms column")
        df['bathrooms'] = df['bathrooms'].apply(process_range_with_mean)

    # Process surface_total column
    if 'surface_total' in df.columns:
        print("- Processing surface_total column")
        df['surface_total'] = df['surface_total'].apply(process_surface)

    # Process surface_useful column
    if 'surface_useful' in df.columns:
        print("- Processing surface_useful column")
        df['surface_useful'] = df['surface_useful'].apply(process_surface)

    print("Processing complete!")
    print(f"Final shape: {df.shape}")

    return df


def main():
    # Read the raw data
    raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
    print(f"Reading {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Show sample of original data
    print("\n--- Sample of original data ---")
    print(df[['price', 'bedrooms', 'bathrooms', 'surface_useful']].head())

    # Process the data
    df_processed = process_dataframe(df)

    # Show sample of processed data
    print("\n--- Sample of processed data ---")
    print(df_processed[['price', 'bedrooms', 'bathrooms', 'surface_useful']].head())

    # Show data types
    print("\n--- Data types ---")
    print(df_processed.dtypes)

    # Show summary statistics
    print("\n--- Summary statistics ---")
    print(df_processed[['price', 'bedrooms', 'bathrooms', 'surface_total', 'surface_useful']].describe())

    # Save processed data
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
