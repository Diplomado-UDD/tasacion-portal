"""
Tests for train_models.py module
Tests model training, evaluation, and comparison functions
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tasacion_portal.train_models import (
    load_and_prepare_data,
    split_data,
    scale_features,
    train_models,
    evaluate_model,
    evaluate_all_models,
    save_results,
    RANDOM_SEED
)


class TestLoadAndPrepareData:
    """Test suite for load_and_prepare_data function"""

    def test_load_and_prepare_data(self, temp_test_dir, sample_processed_data):
        """Test loading and preparing data"""
        # Save sample data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        # Mock PROJECT_ROOT
        with patch('tasacion_portal.train_models.PROJECT_ROOT', temp_test_dir):
            X, y = load_and_prepare_data()

            # Check X contains features
            assert 'bedrooms' in X.columns
            assert 'bathrooms' in X.columns
            assert 'surface_useful' in X.columns

            # Check y is price
            assert len(y) == len(X)
            assert all(y > 0)

    def test_load_and_prepare_data_removes_missing(self, temp_test_dir):
        """Test that missing values are removed"""
        df = pd.DataFrame({
            'bedrooms': [2, None, 3, 4],
            'bathrooms': [2, 2, None, 3],
            'surface_useful': [75, 80, 90, 100],
            'price': [5000, 6000, 7000, 8000]
        })

        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)

        with patch('tasacion_portal.train_models.PROJECT_ROOT', temp_test_dir):
            X, y = load_and_prepare_data()

            # Should only have rows without missing values
            assert len(X) == 2  # Only row 0 and 3 are complete
            assert X.isna().sum().sum() == 0


class TestSplitData:
    """Test suite for split_data function"""

    def test_split_data_default_sizes(self, sample_processed_data):
        """Test data splitting with default sizes"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Check sizes (with 5 samples: 70% = 3-4, 15% = 0-1, 15% = 0-1)
        total = len(X)
        train_expected = int(total * 0.7)

        assert len(X_train) + len(X_val) + len(X_test) == total
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)

    def test_split_data_custom_sizes(self, sample_processed_data):
        """Test data splitting with custom sizes"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2
        )

        total = len(X)
        assert len(X_train) + len(X_val) + len(X_test) == total

    def test_split_data_random_seed(self, sample_processed_data):
        """Test that random seed produces consistent splits"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(X, y)
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(X, y)

        # Should be identical due to RANDOM_SEED
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)

    def test_split_data_invalid_sizes(self, sample_processed_data):
        """Test that invalid split sizes raise error"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        with pytest.raises(AssertionError):
            split_data(X, y, train_size=0.5, val_size=0.3, test_size=0.3)  # Sum > 1


class TestScaleFeatures:
    """Test suite for scale_features function"""

    def test_scale_features(self, train_test_data):
        """Test feature scaling"""
        X_train, X_test, y_train, y_test = train_test_data

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_train, X_test  # Use X_train for validation for simplicity
        )

        # Check that scaler is StandardScaler
        assert isinstance(scaler, StandardScaler)

        # Check that scaled data has same shape
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Check that training data has approximately mean=0, std=1
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

    def test_scale_features_transform_consistency(self, train_test_data):
        """Test that scaler transforms consistently"""
        X_train, X_test, y_train, y_test = train_test_data

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_train, X_test
        )

        # Manually transform using the scaler
        X_train_manual = scaler.transform(X_train)

        np.testing.assert_array_almost_equal(X_train_scaled, X_train_manual)


class TestTrainModels:
    """Test suite for train_models function"""

    def test_train_models_returns_dict(self, train_test_data):
        """Test that train_models returns dictionary of models"""
        X_train, X_test, y_train, y_test = train_test_data

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Disable tuning for faster testing
        with patch('tasacion_portal.train_models.ENABLE_TUNING', False):
            models = train_models(X_train_scaled, y_train)

        assert isinstance(models, dict)
        assert len(models) > 0

        # Check that all expected models are present
        expected_models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression',
                          'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM']
        for model_name in expected_models:
            assert model_name in models

    def test_train_models_all_fitted(self, train_test_data):
        """Test that all models are fitted"""
        X_train, X_test, y_train, y_test = train_test_data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        with patch('tasacion_portal.train_models.ENABLE_TUNING', False):
            models = train_models(X_train_scaled, y_train)

        # Check that each model can make predictions
        for name, model in models.items():
            predictions = model.predict(X_train_scaled)
            assert len(predictions) == len(y_train)
            assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    @patch('tasacion_portal.train_models.ENABLE_TUNING', False)
    def test_train_models_with_tuning_disabled(self, train_test_data):
        """Test training with hyperparameter tuning disabled"""
        X_train, X_test, y_train, y_test = train_test_data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        models = train_models(X_train_scaled, y_train)

        # Should still return all models
        assert len(models) == 7


class TestEvaluateModel:
    """Test suite for evaluate_model function"""

    def test_evaluate_model_returns_metrics(self, train_test_data):
        """Test that evaluate_model returns correct metrics"""
        X_train, X_test, y_train, y_test = train_test_data

        # Train simple model
        model = LinearRegression()
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, 'test')

        # Check that all metrics are present
        assert 'set' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics

        assert metrics['set'] == 'test'

    def test_evaluate_model_metrics_valid(self, train_test_data):
        """Test that metrics have valid values"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression()
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, 'test')

        # Check that metrics are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
        assert metrics['mape'] >= 0

    def test_evaluate_model_perfect_predictions(self, train_test_data):
        """Test metrics with perfect predictions"""
        X_train, X_test, y_train, y_test = train_test_data

        # Create mock model with perfect predictions
        model = Mock()
        model.predict.return_value = y_test.values

        metrics = evaluate_model(model, X_test, y_test, 'test')

        # Perfect predictions should have MSE=0, R²=1
        assert metrics['mse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2'] == 1.0
        assert metrics['mape'] == 0.0


class TestEvaluateAllModels:
    """Test suite for evaluate_all_models function"""

    def test_evaluate_all_models_returns_dataframe(self, train_test_data):
        """Test that evaluate_all_models returns DataFrame"""
        X_train, X_test, y_train, y_test = train_test_data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train just linear regression for testing
        models = {'Linear Regression': LinearRegression().fit(X_train_scaled, y_train)}

        results = evaluate_all_models(
            models, X_train_scaled, X_train_scaled, X_test_scaled,
            y_train, y_train, y_test
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_evaluate_all_models_includes_all_sets(self, train_test_data):
        """Test that results include train, validation, and test sets"""
        X_train, X_test, y_train, y_test = train_test_data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {'Linear Regression': LinearRegression().fit(X_train_scaled, y_train)}

        results = evaluate_all_models(
            models, X_train_scaled, X_train_scaled, X_test_scaled,
            y_train, y_train, y_test
        )

        # Should have results for all three sets
        assert 'train' in results['set'].values
        assert 'validation' in results['set'].values
        assert 'test' in results['set'].values

    def test_evaluate_all_models_multiple_models(self, train_test_data):
        """Test evaluation with multiple models"""
        X_train, X_test, y_train, y_test = train_test_data

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'Linear Regression': LinearRegression().fit(X_train_scaled, y_train),
            'Ridge Regression': LinearRegression().fit(X_train_scaled, y_train)  # Using LR for simplicity
        }

        results = evaluate_all_models(
            models, X_train_scaled, X_train_scaled, X_test_scaled,
            y_train, y_train, y_test
        )

        # Should have 2 models × 3 sets = 6 rows
        assert len(results) == 6
        assert set(results['model'].unique()) == {'Linear Regression', 'Ridge Regression'}


class TestSaveResults:
    """Test suite for save_results function"""

    def test_save_results_creates_file(self, temp_test_dir):
        """Test that save_results creates CSV file"""
        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Ridge Regression'],
            'set': ['test', 'test'],
            'rmse': [100.0, 105.0],
            'r2': [0.85, 0.83]
        })

        output_path = temp_test_dir / 'model_results.csv'

        with patch('tasacion_portal.train_models.PROJECT_ROOT', temp_test_dir):
            save_results(results_df, str(output_path))

        assert output_path.exists()

        # Verify contents
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == list(results_df.columns)

    def test_save_results_creates_directory(self, temp_test_dir):
        """Test that save_results creates output directory if needed"""
        results_df = pd.DataFrame({
            'model': ['Linear Regression'],
            'rmse': [100.0]
        })

        output_path = temp_test_dir / 'outputs' / 'data' / 'results.csv'

        with patch('tasacion_portal.train_models.PROJECT_ROOT', temp_test_dir):
            save_results(results_df, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()


class TestGetParamDistributions:
    """Test suite for get_param_distributions function"""

    def test_get_param_distributions_returns_dict(self):
        """Test that parameter distributions are returned"""
        from tasacion_portal.train_models import get_param_distributions

        param_dists = get_param_distributions()

        assert isinstance(param_dists, dict)
        assert 'Random Forest' in param_dists
        assert 'XGBoost' in param_dists

    def test_param_distributions_have_valid_params(self):
        """Test that parameter distributions contain valid hyperparameters"""
        from tasacion_portal.train_models import get_param_distributions

        param_dists = get_param_distributions()

        # Random Forest should have n_estimators, max_depth, etc.
        rf_params = param_dists['Random Forest']
        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        assert isinstance(rf_params['n_estimators'], list)


class TestModelTrainingIntegration:
    """Integration tests for complete model training pipeline"""

    def test_full_training_pipeline(self, temp_test_dir, sample_processed_data):
        """Test complete training pipeline"""
        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        with patch('tasacion_portal.train_models.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.train_models.ENABLE_TUNING', False):
                # Load data
                X, y = load_and_prepare_data()

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
                    models, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test
                )

                # Verify results
                assert len(results_df) > 0
                assert 'model' in results_df.columns
                assert 'rmse' in results_df.columns
                assert 'r2' in results_df.columns

    def test_reproducibility_with_random_seed(self, sample_processed_data):
        """Test that results are reproducible with fixed random seed"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        # First run
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(X, y)

        # Second run
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(X, y)

        # Should produce identical results
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
