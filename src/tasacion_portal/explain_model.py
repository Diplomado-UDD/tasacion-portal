"""
Model Explanation with SHAP and LIME
Generates interpretability reports for the best regression model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Fixed random seed for reproducibility
RANDOM_SEED = 42


def load_and_prepare_data(filename='data/processed/data.csv'):
    """Load and prepare the data"""
    print("Loading data...")
    filepath = PROJECT_ROOT / filename
    df = pd.read_csv(filepath)

    feature_cols = ['bedrooms', 'bathrooms', 'surface_useful']
    target_col = 'price'

    df_clean = df[feature_cols + [target_col]].dropna()

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"Data shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    return X, y


def split_data(X, y):
    """Split data into train and test sets"""
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )

    # Second split: separate train (70%) and validation (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/(0.85), random_state=RANDOM_SEED
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_best_model(X_train, y_train, X_test, y_test):
    """Train the best model (Linear Regression)"""
    print("\nTraining Linear Regression (best model)...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")

    return model, scaler, X_train_scaled, X_test_scaled


def generate_shap_explanations(model, X_train_scaled, X_test_scaled, feature_names):
    """Generate SHAP explanations"""
    print("\n" + "="*60)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*60)

    # Create SHAP explainer for linear models
    explainer = shap.LinearExplainer(model, X_train_scaled)

    # Calculate SHAP values for test set
    print("\nCalculating SHAP values...")
    shap_values = explainer.shap_values(X_test_scaled)

    # Convert to DataFrame for better readability
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    print(f"SHAP values shape: {shap_values.shape}")

    # Save SHAP values
    shap_path = PROJECT_ROOT / 'outputs' / 'data' / 'shap_values.csv'
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(shap_path, index=False)
    print("✓ SHAP values saved to outputs/data/shap_values.csv")

    # Calculate and print mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nFeature Importance (mean |SHAP value|):")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.2f}")

    importance_path = PROJECT_ROOT / 'outputs' / 'data' / 'shap_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print("✓ Feature importance saved to outputs/data/shap_feature_importance.csv")

    # Generate SHAP summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plot_path = PROJECT_ROOT / 'outputs' / 'plots' / 'shap_summary_plot.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary plot saved to outputs/plots/shap_summary_plot.png")

    # Generate SHAP bar plot
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plot_path = PROJECT_ROOT / 'outputs' / 'plots' / 'shap_bar_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP bar plot saved to outputs/plots/shap_bar_plot.png")

    # Generate waterfall plot for a few samples
    print("Generating SHAP waterfall plots for sample predictions...")
    for i in range(min(3, len(X_test_scaled))):
        plt.figure(figsize=(10, 6))
        shap_exp = shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_test_scaled[i],
            feature_names=feature_names
        )
        shap.waterfall_plot(shap_exp, show=False)
        plt.tight_layout()
        plot_path = PROJECT_ROOT / 'outputs' / 'plots' / f'shap_waterfall_sample_{i+1}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"✓ SHAP waterfall plots saved (3 samples)")

    return shap_values, explainer


def generate_lime_explanations(model, scaler, X_train, X_test, y_test, feature_names):
    """Generate LIME explanations"""
    print("\n" + "="*60)
    print("GENERATING LIME EXPLANATIONS")
    print("="*60)

    # Create LIME explainer
    print("\nCreating LIME explainer...")
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='regression',
        random_state=RANDOM_SEED
    )

    # Create prediction function that handles scaling
    def predict_fn(X):
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    # Generate explanations for a few test samples
    lime_results = []
    n_samples = min(5, len(X_test))

    print(f"\nGenerating LIME explanations for {n_samples} samples...")

    for i in range(n_samples):
        instance = X_test.iloc[i].values
        actual_price = y_test.iloc[i]
        predicted_price = predict_fn(instance.reshape(1, -1))[0]

        # Generate explanation
        exp = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(feature_names)
        )

        # Extract feature contributions
        contributions = dict(exp.as_list())

        # Store results
        result = {
            'sample': i + 1,
            'actual_price': actual_price,
            'predicted_price': predicted_price,
            'error': predicted_price - actual_price
        }

        # Add feature values and contributions
        for j, feature in enumerate(feature_names):
            result[f'{feature}_value'] = instance[j]
            # Find contribution for this feature
            for contrib_str, contrib_val in contributions.items():
                if feature in contrib_str:
                    result[f'{feature}_contribution'] = contrib_val
                    break

        lime_results.append(result)

        # Save visualization
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title(f'LIME Explanation - Sample {i+1}\nActual: {actual_price:.0f} UF, Predicted: {predicted_price:.0f} UF')
        plt.tight_layout()
        plot_path = PROJECT_ROOT / 'outputs' / 'plots' / f'lime_explanation_sample_{i+1}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Sample {i+1}: Actual={actual_price:.0f} UF, Predicted={predicted_price:.0f} UF, Error={result['error']:.0f} UF")

    # Save LIME results to CSV
    lime_df = pd.DataFrame(lime_results)
    lime_path = PROJECT_ROOT / 'outputs' / 'data' / 'lime_explanations.csv'
    lime_df.to_csv(lime_path, index=False)
    print(f"\n✓ LIME explanations saved to outputs/data/lime_explanations.csv")
    print(f"✓ LIME visualizations saved ({n_samples} samples)")

    return lime_df


def create_summary_report(shap_importance, lime_df, feature_names):
    """Create a summary report combining SHAP and LIME insights"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    report = []
    report.append("="*60)
    report.append("MODEL INTERPRETABILITY REPORT")
    report.append("Best Model: Linear Regression")
    report.append("="*60)

    report.append("\n--- SHAP ANALYSIS ---")
    report.append("\nFeature Importance (based on mean |SHAP value|):")
    shap_path = PROJECT_ROOT / 'outputs' / 'data' / 'shap_feature_importance.csv'
    shap_df = pd.read_csv(shap_path)
    for _, row in shap_df.iterrows():
        report.append(f"  {row['feature']}: {row['mean_abs_shap']:.2f}")

    report.append("\nInterpretation:")
    report.append("  - Higher values indicate greater impact on price predictions")
    report.append("  - SHAP values show the contribution of each feature to the prediction")

    report.append("\n--- LIME ANALYSIS ---")
    report.append(f"\nAnalyzed {len(lime_df)} sample predictions:")
    report.append(f"  Average absolute error: {lime_df['error'].abs().mean():.2f} UF")
    report.append(f"  Average predicted price: {lime_df['predicted_price'].mean():.2f} UF")

    # Calculate average feature contributions from LIME
    report.append("\nAverage Feature Contributions (LIME):")
    for feature in feature_names:
        contrib_col = f'{feature}_contribution'
        if contrib_col in lime_df.columns:
            avg_contrib = lime_df[contrib_col].mean()
            report.append(f"  {feature}: {avg_contrib:.2f}")

    report.append("\n--- KEY INSIGHTS ---")
    report.append("\n1. Most Important Feature:")
    top_feature = shap_df.iloc[0]
    report.append(f"   {top_feature['feature']} (SHAP importance: {top_feature['mean_abs_shap']:.2f})")

    report.append("\n2. Model Behavior:")
    report.append("   - Linear model provides consistent, interpretable predictions")
    report.append("   - Feature contributions are additive and transparent")

    report.append("\n3. Files Generated:")
    report.append("   SHAP:")
    report.append("     - shap_values.csv: All SHAP values for test set")
    report.append("     - shap_feature_importance.csv: Feature importance ranking")
    report.append("     - shap_summary_plot.png: Visual summary of feature impacts")
    report.append("     - shap_bar_plot.png: Feature importance bar chart")
    report.append("     - shap_waterfall_sample_*.png: Individual prediction breakdowns")
    report.append("   LIME:")
    report.append("     - lime_explanations.csv: Detailed explanations for samples")
    report.append("     - lime_explanation_sample_*.png: Visual explanations")

    report.append("\n" + "="*60)

    # Print and save report
    report_text = "\n".join(report)
    print(report_text)

    report_path = PROJECT_ROOT / 'outputs' / 'reports' / 'interpretability_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)

    print("\n✓ Full report saved to outputs/reports/interpretability_report.txt")


def main():
    print("="*60)
    print("MODEL INTERPRETABILITY ANALYSIS")
    print("SHAP + LIME Explanations")
    print("="*60)

    # Load and prepare data
    X, y = load_and_prepare_data()  # Uses default: 'data/processed/data.csv'

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    feature_names = list(X.columns)

    # Train best model
    model, scaler, X_train_scaled, X_test_scaled = train_best_model(
        X_train, y_train, X_test, y_test
    )

    # Generate SHAP explanations
    shap_values, shap_explainer = generate_shap_explanations(
        model, X_train_scaled, X_test_scaled, feature_names
    )

    # Generate LIME explanations
    lime_df = generate_lime_explanations(
        model, scaler, X_train, X_test, y_test, feature_names
    )

    # Create summary report
    create_summary_report(shap_values, lime_df, feature_names)

    print("\n" + "="*60)
    print("✓ Interpretability analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
