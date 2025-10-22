# Project Structure - Best Practices Guide

## Current Structure (Not Best Practice)

```
tasacion-portal/
├── .python-version
├── pyproject.toml
├── README.md
├── main.py                    ❌ All scripts at root
├── scraper.py
├── process_data.py
├── train_models.py
├── explain_model.py
├── generate_report.py
├── run_pipeline.py
├── data.csv                   ❌ Generated files at root
├── model_results.csv
├── *.png                      ❌ Plots scattered
├── *.pdf                      ❌ Reports at root
└── __pycache__/               ❌ Pycache at root
```

## Recommended Structure (Best Practice)

```
tasacion-portal/
├── .python-version
├── .gitignore                 ✅ Proper gitignore
├── pyproject.toml
├── README.md
├── STRUCTURE.md               ✅ This file
│
├── src/                       ✅ Source code
│   └── tasacion_portal/
│       ├── __init__.py
│       ├── main.py            # Main pipeline
│       ├── scraper.py         # Data collection
│       ├── process.py         # Data processing
│       ├── models.py          # Model training
│       ├── explain.py         # Interpretability
│       ├── report.py          # Report generation
│       └── utils.py           # Shared utilities
│
├── data/                      ✅ Data directory
│   ├── raw/                   # Raw scraped data
│   │   └── .gitkeep
│   └── processed/             # Cleaned data
│       └── .gitkeep
│
├── outputs/                   ✅ Generated outputs
│   ├── plots/                 # All PNG visualizations
│   │   └── .gitkeep
│   ├── reports/               # PDF reports
│   │   └── .gitkeep
│   └── models/                # Saved model files
│       └── .gitkeep
│
├── tests/                     ✅ Unit tests
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_models.py
│   └── .gitkeep
│
└── notebooks/                 ✅ Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

## Benefits of Recommended Structure

### 1. **Clean Separation of Concerns**
- Source code isolated in `src/`
- Data separate from code
- Outputs don't clutter project root

### 2. **Package Structure**
- Can be installed as a Python package
- Easy imports: `from tasacion_portal import scraper`
- Supports `pip install -e .` for development

### 3. **Gitignore Friendly**
- Generated files ignored by default
- Easy to track only source code
- `.gitkeep` preserves empty directories

### 4. **Professional Standard**
- Follows [Python Packaging Authority](https://packaging.python.org/) guidelines
- Similar to major Python projects (pandas, scikit-learn, etc.)
- Easy for collaborators to understand

### 5. **CI/CD Ready**
- Tests in dedicated directory
- Easy to run `pytest tests/`
- Clear build/deploy process

## Migration Steps

### Option 1: Manual Reorganization (Recommended for Learning)

```bash
# 1. Move Python files to src/
mv *.py src/tasacion_portal/

# 2. Create __init__.py
touch src/tasacion_portal/__init__.py

# 3. Move generated files
mv data.csv data/processed/
mv model_results.csv outputs/
mv *.png outputs/plots/
mv *.pdf outputs/reports/
mv *.csv outputs/ 2>/dev/null || true

# 4. Update imports in all Python files
# Change: import scraper
# To: from tasacion_portal import scraper

# 5. Update file paths in scripts
# Change: 'data.csv'
# To: 'data/processed/data.csv'
```

### Option 2: Keep Current Structure (Quick Fix)

If you want to keep the current structure but make it cleaner:

```bash
# Just organize outputs
mkdir -p outputs/plots outputs/reports outputs/data
mv *.png outputs/plots/
mv *.pdf outputs/reports/
mv *.csv outputs/data/
```

## Updated pyproject.toml for Recommended Structure

```toml
[project]
name = "tasacion-portal"
version = "0.1.0"
description = "Property price analysis using machine learning"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.1.0",
    "pandas>=2.2.0",
    "scikit-learn>=1.5.0",
    "xgboost>=2.0.0",
    "shap>=0.44.0",
    "lime>=0.2.0",
    "matplotlib>=3.8.0",
    "reportlab>=4.0.0",
    "seaborn>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
tasacion = "tasacion_portal.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tasacion_portal"]
```

## Running After Reorganization

With the recommended structure:

```bash
# Install in editable mode
pip install -e .

# Run the pipeline
tasacion

# Or directly
python -m tasacion_portal.main
```

## Comparison

| Aspect | Current | Recommended | Winner |
|--------|---------|-------------|--------|
| Organization | ❌ Flat | ✅ Nested | Recommended |
| Imports | ❌ Direct | ✅ Package | Recommended |
| Outputs | ❌ Root | ✅ Organized | Recommended |
| Installable | ❌ No | ✅ Yes | Recommended |
| Professional | ❌ No | ✅ Yes | Recommended |
| Simple | ✅ Yes | ❌ More complex | Current |
| Quick Start | ✅ Easy | ❌ Requires setup | Current |

## Recommendation

**For this educational project**, I recommend:

1. **Keep current structure for now** (it works and is simple)
2. **Add proper .gitignore** (already done)
3. **Create output directories** to organize generated files
4. **Consider migration** if the project grows or needs packaging

The current flat structure is acceptable for:
- Educational purposes
- Quick prototyping
- Single-developer projects
- Scripts that don't need packaging

The recommended structure is better for:
- Production deployments
- Team collaboration
- Package distribution
- Long-term maintenance

## Quick Win: Minimal Organization

Keep scripts at root but organize outputs:

```bash
# Create directories
mkdir -p outputs/data outputs/plots outputs/reports

# Update scripts to use new paths
# In all Python files, change:
# 'data.csv' → 'outputs/data/data.csv'
# '*.png' → 'outputs/plots/*.png'
# '*.pdf' → 'outputs/reports/*.pdf'
```

This gives you 80% of the benefit with 20% of the effort.
