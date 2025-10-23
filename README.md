# Portal Inmobiliario - Property Price Analysis

A complete machine learning pipeline for scraping, analyzing, and predicting apartment prices from [Portal Inmobiliario](https://www.portalinmobiliario.com), Chile's leading real estate platform.

---

## 🎓 Student Quick Start (GitHub Codespaces)

**New to this project?** Follow these 3 steps:

1. **Open in Codespaces**
   - Click the green "Code" button → "Codespaces" → "Create codespace"
   - Wait for automatic setup (~2 minutes)

2. **Get the data** (choose one):
   ```bash
   # Option A: Scrape fresh data (~5 min)
   python -m tasacion_portal.scraper

   # Option B: Upload your data.csv to data/raw/ folder
   ```

3. **Run the pipeline**:
   ```bash
   python run.py
   ```

**That's it!** 🎉 Results will be in `outputs/reports/` folder.

📖 **Need help?** Jump to [Troubleshooting](#troubleshooting)

---

## Overview

This project provides an end-to-end solution for:
- **Data Collection**: Automated web scraping of property listings
- **Data Processing**: Intelligent cleaning and transformation
- **Machine Learning**: Training and comparing 7 regression models
- **Interpretability**: SHAP and LIME explanations
- **Reporting**: Comprehensive PDF report with visualizations

## Quick Start

### For GitHub Codespaces Users (Recommended for Students) ⭐

**✨ Automatic Setup**: When you open this repository in Codespaces, dependencies will install automatically!

**If auto-setup didn't run**, manually install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # Activate uv in current session
uv sync
```

**You're ready!** Skip to [Run Complete Pipeline](#run-complete-pipeline)

---

### For Local Installation

#### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

#### 1. Install uv

**On Linux/macOS:**
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
# From the project root directory, install all dependencies
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
uv run python -m tasacion_portal
```

**Option 3: Install and use CLI command**
```bash
uv pip install -e .
tasacion
```

This will run all 5 steps automatically (takes ~5-10 minutes).

## Pipeline Steps

The complete workflow consists of 5 sequential steps:

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `scraper.py` | Scrapes property listings from Portal Inmobiliario | `data/raw/data.csv` |
| 2 | `process_data.py` | Cleans and transforms data | `data/processed/data.csv` |
| 3 | `train_models.py` | Trains 7 regression models | `outputs/data/model_results.csv` |
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

**Important Notes**:
- Steps must be run in order as each depends on outputs from previous steps
- **First time?** You must run Step 1 (scraper) OR upload your data file to `data/raw/data.csv` before running other steps
- If you see `FileNotFoundError`, make sure you have the data file from Step 1

## Features

### Data Collection & Processing
- Automated pagination handling (up to 50 pages)
- Extracts: price, location, bedrooms, bathrooms, surface area, URL
- Handles range values (e.g., "2 a 4 dormitorios" → 3.0)
- Cleans text-based numeric fields
- **Outlier removal** using IQR method (configurable)
- Respectful scraping with 2-second delays

### Machine Learning Models

Seven regression models trained and compared:

1. **Linear Regression** - Baseline with interpretable coefficients
2. **Lasso Regression** - L1 regularization for feature selection
3. **Ridge Regression** - L2 regularization to prevent overfitting
4. **Random Forest** - Ensemble of decision trees
5. **XGBoost** - Gradient boosting with regularization
6. **CatBoost** - Gradient boosting optimized for categorical features
7. **LightGBM** - Fast gradient boosting with leaf-wise tree growth

**Evaluation Metrics**: RMSE, MAE, R², MAPE

**Data Split**: 70% train, 15% validation, 15% test (reproducible with seed=42)

**Hyperparameter Tuning**: Automatic optimization using RandomizedSearchCV for tree-based models (Random Forest, XGBoost, CatBoost, LightGBM)

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
- **Machine Learning**: scikit-learn, xgboost, catboost, lightgbm
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

# Enable/disable hyperparameter tuning
ENABLE_TUNING = True  # Set to False for faster training
N_ITER = 20  # Number of parameter combinations to try
CV_FOLDS = 3  # Cross-validation folds

# Adjust data split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, train_size=0.8, val_size=0.1, test_size=0.1
)

# Modify model hyperparameters
RandomForestRegressor(n_estimators=200, max_depth=15, ...)
```

### Adjust Outlier Removal

Edit `src/tasacion_portal/process_data.py`:

```python
# Change outlier removal method and threshold
df = remove_outliers(df, outlier_columns, method='iqr', threshold=1.5)

# Or use Z-score method instead
df = remove_outliers(df, outlier_columns, method='zscore', threshold=3)

# Disable outlier removal by commenting out the line
# df = remove_outliers(df, outlier_columns, method='iqr', threshold=1.5)
```

## Results Summary

Based on the default configuration:

- **Best Model**: CatBoost
- **R² Score**: 0.63 (explains 63% of price variance)
- **RMSE**: ~33,825 UF
- **MAPE**: ~16.8%
- **Most Important Feature**: Surface area (70% importance)

## Notes

- Data is scraped from public listings on Portal Inmobiliario
- The scraper respects rate limits with 2-second delays between requests
- The pipeline automatically stops when reaching the last available page
- All outputs use UTF-8 encoding to handle Spanish characters
- Models use standardized features (StandardScaler)

---

## Troubleshooting

### Common Issues in Codespaces

#### ❌ Error: `ModuleNotFoundError: No module named 'xgboost'`

**Cause**: Dependencies not installed

**Solution**:
```bash
uv sync
```

If `uv` is not found, install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

---

#### ❌ Error: `FileNotFoundError: [Errno 2] No such file or directory: '/workspaces/tasacion-portal/data/raw/data.csv'`

**Cause**: Data file doesn't exist yet

**Solutions**:

**Option 1: Run the scraper** (creates fresh data - takes ~5 minutes):
```bash
# Create directories
mkdir -p data/raw data/processed

# Run scraper
python -m tasacion_portal.scraper
```

**Option 2: Upload your existing data**:
1. Create the directory: `mkdir -p data/raw`
2. Upload your `data.csv` file to `data/raw/` folder in Codespaces
3. Continue with processing

---

#### ❌ Error: `uv: command not found` after installation

**Cause**: Terminal needs to reload environment

**Solution**:
```bash
source $HOME/.local/bin/env
```

Or close and reopen the terminal.

---

#### ❌ Training takes too long (>10 minutes)

**Cause**: Hyperparameter tuning is enabled

**Solution**: Disable tuning in `src/tasacion_portal/train_models.py`:
```python
ENABLE_TUNING = False  # Line 27
```

Then re-run:
```bash
python -m tasacion_portal.train_models
```

---

#### ❌ Error: `Permission denied` when creating directories

**Cause**: Codespaces file permissions

**Solution**:
```bash
# Fix permissions
chmod -R u+w data/ outputs/

# Create directories manually
mkdir -p data/raw data/processed outputs/data outputs/plots outputs/reports
```

---

### Quick Reset

If things get messed up, reset everything:

```bash
# Remove generated files
rm -rf data/ outputs/ .venv/

# Reinstall dependencies
uv sync

# Run scraper to get fresh data
mkdir -p data/raw
python -m tasacion_portal.scraper
```

---

### Getting Help

1. **Check the error message carefully** - it usually tells you what's missing
2. **Review IMPROVEMENTS.md** for configuration options
3. **Make sure you ran `uv sync`** after cloning the repository
4. **Verify you're in the project root** directory when running commands

---

## License

This project is for educational purposes only. Please respect Portal Inmobiliario's terms of service and robots.txt when using this scraper.

## Contributing

This is an educational project. Feel free to fork and adapt for your own learning purposes.

---

**Generated with**: Python 3.12, scikit-learn, XGBoost, CatBoost, LightGBM, SHAP, LIME
