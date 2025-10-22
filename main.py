"""
Main script - chains scraping and data processing
"""

from scraper import PortalInmobiliarioScraper
import process_data
import pandas as pd


def main():
    print("=" * 60)
    print("Portal Inmobiliario - Scraper & Processor")
    print("=" * 60)

    # Step 1: Scrape the data
    print("\n[STEP 1/2] SCRAPING DATA")
    print("-" * 60)

    url = "https://www.portalinmobiliario.com/venta/departamento"
    scraper = PortalInmobiliarioScraper(url)

    # Scrape multiple pages
    scraper.scrape(max_pages=50)

    # Save raw data
    scraper.save_to_csv('data.csv')

    # Step 2: Process the data
    print("\n[STEP 2/2] PROCESSING DATA")
    print("-" * 60)

    # Read the raw data
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} rows")

    # Process the data
    df_processed = process_data.process_dataframe(df)

    # Save processed data
    df_processed.to_csv('data.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ Final data saved to data.csv")

    # Show summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total properties: {len(df_processed)}")
    print(f"\nData types:")
    print(df_processed.dtypes)
    print(f"\nSample data:")
    print(df_processed[['price', 'bedrooms', 'bathrooms', 'surface_useful']].head())
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
