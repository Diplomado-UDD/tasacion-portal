"""
Complete Analysis Pipeline - Main Entry Point
Runs the entire workflow from data scraping to final PDF report
"""

import sys
import os
from pathlib import Path
from .scraper import PortalInmobiliarioScraper
from . import process_data
import pandas as pd
from datetime import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def print_step(step_num, total_steps, description):
    """Print formatted step header"""
    print("\n" + "="*70)
    print(f"STEP {step_num}/{total_steps}: {description}")
    print("="*70)


def step1_scrape_data():
    """Step 1: Scrape property data from Portal Inmobiliario"""
    print_step(1, 5, "DATA COLLECTION")

    url = "https://www.portalinmobiliario.com/venta/departamento"
    print(f"\nScraping data from: {url}")

    scraper = PortalInmobiliarioScraper(url)
    scraper.scrape(max_pages=50)

    # Save to data/raw directory
    raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
    scraper.save_to_csv(str(raw_data_path))

    print(f"\n✓ Step 1 complete: {len(scraper.properties)} properties scraped")
    return len(scraper.properties)


def step2_process_data():
    """Step 2: Clean and process the scraped data"""
    print_step(2, 5, "DATA PROCESSING")

    print("\nReading raw data...")
    raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows")

    print("\nProcessing data...")
    df_processed = process_data.process_dataframe(df)

    print("\nSaving processed data...")
    processed_data_path = PROJECT_ROOT / 'data' / 'processed' / 'data.csv'
    df_processed.to_csv(processed_data_path, index=False, encoding='utf-8-sig')

    print(f"\n✓ Step 2 complete: Data cleaned and processed")
    return len(df_processed)


def step3_train_models():
    """Step 3: Train and evaluate regression models"""
    print_step(3, 5, "MODEL TRAINING")

    print("\nTraining models (this may take a few minutes)...")

    # Import and run train_models
    from . import train_models
    train_models.main()

    print(f"\n✓ Step 3 complete: Models trained and evaluated")


def step4_explain_model():
    """Step 4: Generate SHAP and LIME explanations"""
    print_step(4, 5, "MODEL INTERPRETABILITY")

    print("\nGenerating SHAP and LIME explanations...")

    # Import and run explain_model
    from . import explain_model
    explain_model.main()

    print(f"\n✓ Step 4 complete: Model explanations generated")


def step5_generate_report():
    """Step 5: Create comprehensive PDF report"""
    print_step(5, 5, "REPORT GENERATION")

    print("\nGenerating comprehensive PDF report...")

    # Import and run generate_report
    from . import generate_report
    generate_report.main()

    print(f"\n✓ Step 5 complete: PDF report created")


def main():
    """Run the complete analysis pipeline"""
    start_time = datetime.now()

    print("\n" + "="*70)
    print("PROPERTY PRICE ANALYSIS - COMPLETE PIPELINE")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will execute 5 steps:")
    print("  1. Data Collection (web scraping)")
    print("  2. Data Processing (cleaning & transformation)")
    print("  3. Model Training (5 regression models)")
    print("  4. Model Interpretability (SHAP & LIME)")
    print("  5. Report Generation (PDF with plots)")
    print("\nEstimated time: 5-10 minutes")

    input("\nPress ENTER to start the pipeline...")

    try:
        # Step 1: Scrape data (skip if already exists)
        raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
        if raw_data_path.exists():
            print_step(1, 5, "DATA COLLECTION")
            print(f"\n⚠️  Raw data already exists at {raw_data_path}")
            print("Skipping Step 1 (scraping). Delete the file to re-scrape.")
            properties_count = len(pd.read_csv(raw_data_path))
            print(f"\n✓ Step 1 skipped: Using existing {properties_count} properties")
        else:
            properties_count = step1_scrape_data()

        # Step 2: Process data
        clean_count = step2_process_data()

        # Step 3: Train models
        step3_train_models()

        # Step 4: Explain model
        step4_explain_model()

        # Step 5: Generate report
        step5_generate_report()

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nExecution time: {duration/60:.1f} minutes")
        print(f"\nFinal outputs:")
        print(f"  • data/raw/data.csv (raw data)")
        print(f"  • data/processed/data.csv ({clean_count} clean records from {properties_count} scraped)")
        print(f"  • outputs/data/model_results.csv (5 models compared)")
        print(f"  • outputs/plots/shap_*.png, lime_*.png (visualizations)")
        print(f"  • outputs/reports/property_price_analysis_report_*.pdf (final report)")
        print(f"\n🎉 All done! Check the PDF report in outputs/reports/ for complete findings.")

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
