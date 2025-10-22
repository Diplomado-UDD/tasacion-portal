"""
Regression Model Training
Trains multiple regression models to predict property prices
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Fixed random seed for reproducibility
RANDOM_SEED = 42


def load_and_prepare_data(filename='data/processed/data.csv'):
    """Load and prepare the data for modeling"""
    print("Loading data...")
    filepath = PROJECT_ROOT / filename
    df = pd.read_csv(filepath)

    print(f"Initial data shape: {df.shape}")

    # Select only numeric features with sufficient data
    # Excluding surface_total (only 8 values) and description (empty)
    feature_cols = ['bedrooms', 'bathrooms', 'surface_useful']
    target_col = 'price'

    print(f"\nData availability:")
    for col in feature_cols + [target_col]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")

    # Remove rows with missing values in key columns
    df_clean = df[feature_cols + [target_col]].dropna()

    print(f"\nClean data shape: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} rows with missing values")

    # Split features and target
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"\nFeatures used: {list(X.columns)}")
    print(f"Target: {target_col}")
    print(f"\nFeature statistics:")
    print(X.describe())

    return X, y


def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    """Split data into train, validation, and test sets"""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    # Second split: separate train and validation
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED
    )

    print(f"\nData split (seed={RANDOM_SEED}):")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Standardize features using training set statistics"""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeatures scaled using StandardScaler")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_models(X_train, y_train):
    """Train all regression models"""
    print("\nTraining models...")

    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=1.0, random_state=RANDOM_SEED),
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_SEED),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    print("✓ All models trained")

    return trained_models


def evaluate_model(model, X, y, set_name):
    """Evaluate a single model"""
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    return {
        'set': set_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def evaluate_all_models(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate all models on train, validation, and test sets"""
    print("\nEvaluating models...")

    results = []

    for name, model in models.items():
        print(f"\n{name}:")

        # Evaluate on train set
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        print(f"  Train - RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}, MAPE: {train_metrics['mape']:.2f}%")

        # Evaluate on validation set
        val_metrics = evaluate_model(model, X_val, y_val, 'validation')
        print(f"  Val   - RMSE: {val_metrics['rmse']:.2f}, MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.4f}, MAPE: {val_metrics['mape']:.2f}%")

        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test, 'test')
        print(f"  Test  - RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}, MAPE: {test_metrics['mape']:.2f}%")

        # Store results
        for metrics in [train_metrics, val_metrics, test_metrics]:
            results.append({
                'model': name,
                **metrics
            })

    return pd.DataFrame(results)


def print_feature_importance(models, feature_names):
    """Print feature importance for tree-based models"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name}:")
            importances = model.feature_importances_
            for feature, importance in sorted(zip(feature_names, importances),
                                             key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")


def save_results(results_df, filename='outputs/data/model_results.csv'):
    """Save results to CSV"""
    filepath = PROJECT_ROOT / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"\n✓ Results saved to {filename}")


def print_summary(results_df):
    """Print summary comparison of models on test set"""
    print("\n" + "="*60)
    print("MODEL COMPARISON (TEST SET)")
    print("="*60)

    test_results = results_df[results_df['set'] == 'test'].copy()
    test_results = test_results.sort_values('rmse')

    print("\nRanked by RMSE (lower is better):")
    print(test_results[['model', 'rmse', 'mae', 'r2', 'mape']].to_string(index=False))

    best_model = test_results.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']}")
    print(f"   RMSE: {best_model['rmse']:.2f}")
    print(f"   MAE: {best_model['mae']:.2f}")
    print(f"   R²: {best_model['r2']:.4f}")
    print(f"   MAPE: {best_model['mape']:.2f}%")


def main():
    print("="*60)
    print("PROPERTY PRICE PREDICTION MODEL TRAINING")
    print("="*60)

    # Load and prepare data
    X, y = load_and_prepare_data('data.csv')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Train models
    models = train_models(X_train_scaled, y_train)

    # Evaluate models
    results_df = evaluate_all_models(
        models,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )

    # Print feature importance
    print_feature_importance(models, X.columns)

    # Print summary
    print_summary(results_df)

    # Save results
    save_results(results_df, 'outputs/data/model_results.csv')

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
