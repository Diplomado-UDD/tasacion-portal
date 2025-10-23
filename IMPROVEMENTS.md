# Model Improvements Summary

This document describes the improvements made to enhance model performance and data quality.

## 1. Outlier Removal (process_data.py)

### What was added:
- `remove_outliers()` function with two methods:
  - **IQR (Interquartile Range)**: Default method, removes values beyond Q1 - 1.5×IQR and Q3 + 1.5×IQR
  - **Z-Score**: Alternative method, removes values beyond ±3 standard deviations

### Implementation:
```python
# Automatically applied to these columns:
outlier_columns = ['price', 'bedrooms', 'bathrooms', 'surface_useful']

# Using IQR method with 1.5 threshold
df = remove_outliers(df, outlier_columns, method='iqr', threshold=1.5)
```

### Benefits:
- **Better model generalization**: Removes extreme values that don't represent typical properties
- **Improved R² scores**: Expected 2-5% improvement by removing noise
- **More reliable predictions**: Models won't be skewed by outliers
- **Transparent**: Prints how many outliers were removed

### Example output:
```
- Removing outliers using IQR method...
  Removed 147 outliers (7.3% of data)
```

---

## 2. Hyperparameter Tuning (train_models.py)

### What was added:
- `RandomizedSearchCV` for automatic hyperparameter optimization
- Configurable tuning parameters at the top of the file:
  ```python
  ENABLE_TUNING = True  # Set to False for faster training
  N_ITER = 20          # Number of combinations to try
  CV_FOLDS = 3         # Cross-validation folds
  ```

### Models tuned:
1. **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
2. **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
3. **CatBoost**: n_estimators, max_depth, learning_rate
4. **LightGBM**: n_estimators, max_depth, learning_rate, num_leaves

### Benefits:
- **Better performance**: Expected 5-15% improvement in RMSE/R²
- **Automatic optimization**: No manual trial-and-error needed
- **Cross-validated**: Uses 3-fold CV to prevent overfitting
- **Fast**: RandomizedSearchCV is faster than GridSearchCV (20 iterations vs 1000s)

### Example output:
```
Training models...
  Hyperparameter tuning enabled: 20 iterations, 3-fold CV
  Training Random Forest... (tuning 20 combinations)
    Best params: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1}
```

---

## Expected Performance Improvements

### Before improvements:
- **Best Model**: Linear Regression
- **R² Score**: ~0.62
- **RMSE**: ~48,566 UF
- **MAPE**: ~19.1%

### After improvements (estimated):
- **Best Model**: XGBoost or LightGBM (with tuned hyperparameters)
- **R² Score**: ~0.68-0.72 (+10-15% improvement)
- **RMSE**: ~42,000-45,000 UF (-10-15% improvement)
- **MAPE**: ~17-18% (-10-15% improvement)

### Why?
1. **Outlier removal** eliminates noise → cleaner patterns
2. **Hyperparameter tuning** finds optimal model settings → better predictions
3. **Combined effect**: Data quality + model optimization = significant gains

---

## Configuration Guide

### Disable tuning for faster training:
```python
# In train_models.py
ENABLE_TUNING = False
```

### Adjust tuning intensity:
```python
# More thorough (slower):
N_ITER = 50
CV_FOLDS = 5

# Faster (less thorough):
N_ITER = 10
CV_FOLDS = 3
```

### Change outlier removal sensitivity:
```python
# More aggressive (removes more):
df = remove_outliers(df, outlier_columns, method='iqr', threshold=1.0)

# Less aggressive (removes fewer):
df = remove_outliers(df, outlier_columns, method='iqr', threshold=2.0)

# Alternative method:
df = remove_outliers(df, outlier_columns, method='zscore', threshold=3)
```

---

## Trade-offs

### Outlier Removal:
- ✅ **Pros**: Better model performance, cleaner data
- ⚠️ **Cons**: Loses 5-10% of data, may remove legitimate extreme-value properties

### Hyperparameter Tuning:
- ✅ **Pros**: Automatic optimization, better performance, uses cross-validation
- ⚠️ **Cons**: Slower training (5-10 minutes instead of 1-2 minutes)

---

## Next Steps to Test

1. **Reprocess data with outlier removal**:
   ```bash
   python -m tasacion_portal.process_data
   ```

2. **Train models with hyperparameter tuning**:
   ```bash
   python -m tasacion_portal.train_models
   ```

3. **Compare results**:
   - Check `outputs/data/model_results.csv`
   - Look for improved R² and lower RMSE
   - Note the best hyperparameters found

4. **Optional: Run full pipeline**:
   ```bash
   python run.py
   ```

---

## Files Modified

1. **src/tasacion_portal/process_data.py**
   - Added `remove_outliers()` function (lines 71-120)
   - Integrated into `process_dataframe()` (lines 153-156)

2. **src/tasacion_portal/train_models.py**
   - Added `get_param_distributions()` (lines 32-59)
   - Modified `train_models()` to use `RandomizedSearchCV` (lines 134-206)
   - Added configuration constants (lines 27-29)

3. **README.md**
   - Updated features section
   - Added configuration examples

4. **IMPROVEMENTS.md** (this file)
   - Complete documentation of changes
