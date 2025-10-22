# Portal Inmobiliario - Property Price Analysis

A complete machine learning pipeline for scraping, analyzing, and predicting apartment prices from [Portal Inmobiliario](https://www.portalinmobiliario.com), Chile's leading real estate platform.

## Overview

This project provides an end-to-end solution for:
- **Data Collection**: Automated web scraping of property listings
- **Data Processing**: Intelligent cleaning and transformation
- **Machine Learning**: Training and comparing 5 regression models
- **Interpretability**: SHAP and LIME explanations
- **Reporting**: Comprehensive PDF report with visualizations

## Quick Start

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

#### 1. Install uv

**On Linux/macOS/GitHub Codespaces:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

**Verify installation:**
```bash
uv --version
```

#### 2. Install Project Dependencies

```bash
# Navigate to project directory
cd tasacion-portal

# Install all dependencies
uv sync
```

### Run Complete Pipeline

Execute the entire analysis with a single command:

**Option 1: Using the run script (Simple)**
```bash
python run.py
```

**Option 2: As a Python module**
```bash
python -m tasacion_portal
```

**Option 3: Install and use CLI command**
```bash
pip install -e .
tasacion
```

This will run all 5 steps automatically (takes ~5-10 minutes).

## Pipeline Steps

The complete workflow consists of 5 sequential steps:

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `scraper.py` | Scrapes property listings from Portal Inmobiliario | `data/raw/data.csv` |
| 2 | `process_data.py` | Cleans and transforms data | `data/processed/data.csv` |
| 3 | `train_models.py` | Trains 5 regression models | `outputs/data/model_results.csv` |
| 4 | `explain_model.py` | Generates SHAP & LIME explanations | `outputs/data/*.csv`, `outputs/plots/*.png` |
| 5 | `generate_report.py` | Creates comprehensive PDF report | `outputs/reports/*.pdf` |

### Run Individual Steps (Optional)

If needed, you can run specific steps separately:

```bash
# Step 1: Data collection only
uv run python -m tasacion_portal.scraper

# Step 2: Data processing only
uv run python -m tasacion_portal.process_data

# Step 3: Model training only
uv run python -m tasacion_portal.train_models

# Step 4: Model interpretability only
uv run python -m tasacion_portal.explain_model

# Step 5: PDF report generation only
uv run python -m tasacion_portal.generate_report
```

**Note**: Steps must be run in order as each depends on outputs from previous steps.

## Features

### Data Collection & Processing
- Automated pagination handling (up to 50 pages)
- Extracts: price, location, bedrooms, bathrooms, surface area, URL
- Handles range values (e.g., "2 a 4 dormitorios" → 3.0)
- Cleans text-based numeric fields
- Respectful scraping with 2-second delays

### Machine Learning Models

Five regression models trained and compared:

1. **Linear Regression** - Baseline with interpretable coefficients
2. **Lasso Regression** - L1 regularization for feature selection
3. **Ridge Regression** - L2 regularization to prevent overfitting
4. **Random Forest** - Ensemble of decision trees
5. **XGBoost** - Gradient boosting with regularization

**Evaluation Metrics**: RMSE, MAE, R², MAPE

**Data Split**: 70% train, 15% validation, 15% test (reproducible with seed=42)

### Model Interpretability

**SHAP (SHapley Additive exPlanations)**
- Global feature importance
- Individual prediction explanations
- Summary plots and waterfall charts

**LIME (Local Interpretable Model-agnostic Explanations)**
- Local explanations for individual predictions
- Feature contribution visualizations

### PDF Report

Comprehensive report including:
- Executive summary with key metrics
- Data exploration visualizations
- Model performance comparison
- SHAP and LIME analysis
- Key findings and recommendations
- Technical appendix

## Output Files

After running the complete pipeline, outputs are organized in the following structure:

```
data/
├── raw/data.csv                              # Raw scraped data
└── processed/data.csv                        # Clean property data

outputs/
├── data/
│   ├── model_results.csv                     # Model performance metrics
│   ├── shap_values.csv                       # SHAP values for test set
│   ├── shap_feature_importance.csv           # Feature importance ranking
│   └── lime_explanations.csv                 # LIME explanations
├── plots/
│   ├── data_summary_plot.png                 # Data exploration
│   ├── model_comparison_plot.png             # Model performance
│   ├── feature_importance_comparison.png     # Feature importance
│   ├── shap_summary_plot.png                 # SHAP summary
│   ├── shap_bar_plot.png                     # Feature importance bar chart
│   ├── shap_waterfall_sample_*.png           # Individual predictions
│   └── lime_explanation_sample_*.png         # LIME visualizations
└── reports/
    ├── interpretability_report.txt           # Text summary
    └── property_price_analysis_report_YYYYMMDD.pdf  # Final PDF report
```

## Project Structure

```
tasacion-portal/
├── .python-version                    # Python version (3.12)
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Project configuration & dependencies
├── README.md                          # This file
├── STRUCTURE.md                       # Structure documentation
├── run.py                             # Simple run script (recommended)
│
├── src/tasacion_portal/               # Source code (package)
│   ├── __init__.py                    # Package initialization
│   ├── __main__.py                    # Module entry point
│   ├── main.py                        # Complete pipeline orchestration
│   ├── scraper.py                     # Web scraper (Step 1)
│   ├── process_data.py                # Data processing (Step 2)
│   ├── train_models.py                # Model training (Step 3)
│   ├── explain_model.py               # SHAP & LIME (Step 4)
│   └── generate_report.py             # PDF generator (Step 5)
│
├── data/                              # Data directory
│   ├── raw/                           # Raw scraped data
│   │   └── data.csv
│   └── processed/                     # Cleaned data
│       └── data.csv
│
├── outputs/                           # Generated outputs
│   ├── data/                          # CSVs (model results, etc.)
│   ├── plots/                         # PNG visualizations
│   ├── reports/                       # PDF reports
│   └── models/                        # Saved models
│
└── tests/                             # Unit tests
    └── .gitkeep
```

## Dependencies

- **Web Scraping**: requests, beautifulsoup4, lxml
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Interpretability**: shap, lime
- **Visualization**: matplotlib, seaborn
- **Reporting**: reportlab

All dependencies are automatically installed with `uv sync`.

## Customization

### Change Scraping Parameters

Edit `src/tasacion_portal/main.py` or `src/tasacion_portal/scraper.py`:

```python
# Scrape more pages
scraper.scrape(max_pages=100)

# Change target URL (e.g., houses instead of apartments)
url = "https://www.portalinmobiliario.com/venta/casa"

# Change output filename
scraper.save_to_csv('my_data.csv')
```

### Adjust Model Parameters

Edit `src/tasacion_portal/train_models.py`:

```python
# Change random seed
RANDOM_SEED = 123

# Adjust data split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, train_size=0.8, val_size=0.1, test_size=0.1
)

# Modify model hyperparameters
RandomForestRegressor(n_estimators=200, max_depth=15, ...)
```

## Results Summary

Based on the default configuration:

- **Best Model**: Linear Regression
- **R² Score**: 0.62 (explains 62% of price variance)
- **RMSE**: ~4,858 UF
- **MAPE**: ~19.1%
- **Most Important Feature**: Surface area (70% importance)

## Notes

- Data is scraped from public listings on Portal Inmobiliario
- The scraper respects rate limits with 2-second delays between requests
- The pipeline automatically stops when reaching the last available page
- All outputs use UTF-8 encoding to handle Spanish characters
- Models use standardized features (StandardScaler)

## License

This project is for educational purposes only. Please respect Portal Inmobiliario's terms of service and robots.txt when using this scraper.

## Contributing

This is an educational project. Feel free to fork and adapt for your own learning purposes.

---

**Generated with**: Python 3.12, scikit-learn, XGBoost, SHAP, LIME
