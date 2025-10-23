# GitHub Codespaces Setup Guide

This document explains how the project is configured for GitHub Codespaces and what students need to know.

## Automatic Setup

When students open this repository in Codespaces, the following happens automatically:

1. ✅ **Python 3.12** is pre-installed (from devcontainer)
2. ✅ **uv package manager** is installed
3. ✅ **All dependencies** are installed via `uv sync`
4. ✅ **Data directories** are created (`data/raw`, `data/processed`, `outputs/*`)
5. ✅ **VS Code extensions** are installed (Python, Pylance, Jupyter)

## Files That Make This Work

### `.devcontainer/devcontainer.json`
Configures the Codespaces environment:
- Base image: Python 3.12
- Post-create command: Runs setup script
- VS Code settings and extensions
- Python interpreter configuration

### `.devcontainer/setup.sh`
Automatic setup script that:
- Installs uv package manager
- Runs `uv sync` to install dependencies
- Creates necessary directories
- Prints success message

## Student Workflow

### First Time Setup

Students should:

1. **Open in Codespaces**
   - Go to the GitHub repository
   - Click "Code" → "Codespaces" → "Create codespace"
   - Wait 2-3 minutes for automatic setup

2. **Verify Setup**
   ```bash
   # Check uv is installed
   uv --version

   # Check dependencies
   python -c "import xgboost, catboost, lightgbm; print('✅ All libraries installed')"
   ```

3. **Get Data** (two options)

   **Option A: Scrape fresh data**
   ```bash
   python -m tasacion_portal.scraper
   ```

   **Option B: Upload existing data**
   - Create folder: `mkdir -p data/raw`
   - Upload `data.csv` to `data/raw/` via VS Code file explorer

4. **Run Pipeline**
   ```bash
   python run.py
   ```

### Common Student Issues

#### Issue 1: "uv: command not found"
**Solution**:
```bash
source $HOME/.local/bin/env
# OR close and reopen terminal
```

#### Issue 2: "ModuleNotFoundError: No module named 'xgboost'"
**Solution**:
```bash
uv sync
```

#### Issue 3: "FileNotFoundError: data/raw/data.csv"
**Solution**: Need to run scraper or upload data first
```bash
mkdir -p data/raw
python -m tasacion_portal.scraper
```

#### Issue 4: Setup script didn't run automatically
**Solution**: Run it manually
```bash
bash .devcontainer/setup.sh
```

## Environment Details

### Python Version
- **Required**: Python 3.12
- **Provided by**: Codespaces devcontainer

### Dependencies Installed
- **Core ML**: scikit-learn, xgboost, catboost, lightgbm
- **Interpretability**: shap, lime
- **Data**: pandas, numpy
- **Scraping**: requests, beautifulsoup4, lxml
- **Visualization**: matplotlib, seaborn
- **Reporting**: reportlab

### Directory Structure Created
```
data/
├── raw/              # Raw scraped data
└── processed/        # Cleaned data

outputs/
├── data/            # Model results, SHAP values, LIME explanations
├── plots/           # Visualizations
├── reports/         # PDF reports
└── models/          # Saved models (if enabled)
```

## Customization for Instructors

### Disable Automatic Setup

Remove or comment out in `.devcontainer/devcontainer.json`:
```json
"postCreateCommand": "bash .devcontainer/setup.sh",
```

### Pre-load Data

Add to `.devcontainer/setup.sh`:
```bash
# Download pre-scraped data
curl -o data/raw/data.csv https://your-server.com/data.csv
```

### Change Python Version

Edit `.devcontainer/devcontainer.json`:
```json
"features": {
  "ghcr.io/devcontainers/features/python:1": {
    "version": "3.11"  // or 3.10
  }
}
```

### Add Additional Extensions

Edit `.devcontainer/devcontainer.json`:
```json
"extensions": [
  "ms-python.python",
  "ms-python.vscode-pylance",
  "ms-toolsai.jupyter",
  "ms-azuretools.vscode-docker"  // Add more extensions
]
```

## Testing the Setup

### For Instructors

Before sharing with students, test the setup:

1. **Create a fresh Codespace**
   ```
   Delete any existing codespaces
   Create new one from clean state
   ```

2. **Verify automatic setup**
   ```bash
   # Should see these messages:
   # 🚀 Setting up Property Price Analysis environment...
   # 📦 Installing uv package manager...
   # 📚 Installing project dependencies...
   # 📁 Creating data directories...
   # ✅ Setup complete!
   ```

3. **Test import**
   ```bash
   python -c "import xgboost, catboost, lightgbm; print('Success!')"
   ```

4. **Test scraper**
   ```bash
   python -m tasacion_portal.scraper  # Let it run for 1-2 pages then Ctrl+C
   ls data/raw/  # Should see data.csv
   ```

5. **Test training**
   ```bash
   python -m tasacion_portal.train_models
   ```

## Troubleshooting for Instructors

### Setup script doesn't run

**Possible causes**:
1. Syntax error in `setup.sh`
2. Wrong path in `devcontainer.json`
3. Permissions issue

**Debug**:
```bash
# Check if script exists
ls -la .devcontainer/setup.sh

# Run manually to see errors
bash -x .devcontainer/setup.sh
```

### Dependencies fail to install

**Check**:
```bash
# View uv logs
cat ~/.local/state/uv/cache/*/logs/*.log

# Try manual install
uv sync --verbose
```

### Student workspaces run slow

**Optimize**:
1. Disable hyperparameter tuning by default
2. Use smaller dataset sample
3. Reduce number of models trained

Edit `train_models.py`:
```python
ENABLE_TUNING = False  # Faster for students
```

## Best Practices for Students

1. ✅ **Always work in Codespaces** - consistent environment
2. ✅ **Commit progress regularly** - don't lose work
3. ✅ **Check error messages** - they're usually clear
4. ✅ **Use the troubleshooting guide** - common issues documented
5. ✅ **Don't commit data files** - use .gitignore
6. ✅ **Stop Codespace when done** - save billing hours

## Resources

- [Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Dev Containers Spec](https://containers.dev/)
- [uv Documentation](https://docs.astral.sh/uv/)
- Project README.md (main documentation)
- IMPROVEMENTS.md (optimization features)

---

**Last Updated**: 2025-10-22
