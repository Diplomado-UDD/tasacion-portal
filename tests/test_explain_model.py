"""
Tests for explain_model.py module
Tests SHAP and LIME model explanation functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tasacion_portal.explain_model import (
    load_and_prepare_data,
    split_data,
    train_best_model,
    generate_shap_explanations,
    generate_lime_explanations
)


class TestLoadAndPrepareDataExplain:
    """Test suite for load_and_prepare_data in explain_model"""

    def test_load_and_prepare_data_explain(self, temp_test_dir, sample_processed_data):
        """Test loading data for model explanation"""
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            X, y = load_and_prepare_data()

            assert 'bedrooms' in X.columns
            assert 'bathrooms' in X.columns
            assert 'surface_useful' in X.columns
            assert len(X) == len(y)


class TestSplitDataExplain:
    """Test suite for split_data in explain_model"""

    def test_split_data_explain(self, sample_processed_data):
        """Test data splitting for explanations"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        total = len(X)
        assert len(X_train) + len(X_val) + len(X_test) == total
        assert len(y_train) == len(X_train)

    def test_split_reproducibility(self, sample_processed_data):
        """Test that splits are reproducible"""
        X = sample_processed_data[['bedrooms', 'bathrooms', 'surface_useful']]
        y = sample_processed_data['price']

        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(X, y)
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(X, y)

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)


class TestTrainBestModel:
    """Test suite for train_best_model function"""

    def test_train_best_model_returns_components(self, train_test_data):
        """Test that train_best_model returns all components"""
        X_train, X_test, y_train, y_test = train_test_data

        model, scaler, X_train_scaled, X_test_scaled = train_best_model(
            X_train, y_train, X_test, y_test
        )

        assert isinstance(model, LinearRegression)
        assert isinstance(scaler, StandardScaler)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

    def test_train_best_model_fits_correctly(self, train_test_data):
        """Test that model is properly fitted"""
        X_train, X_test, y_train, y_test = train_test_data

        model, scaler, X_train_scaled, X_test_scaled = train_best_model(
            X_train, y_train, X_test, y_test
        )

        # Model should be able to make predictions
        predictions = model.predict(X_test_scaled)
        assert len(predictions) == len(y_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_train_best_model_scaling(self, train_test_data):
        """Test that feature scaling is applied correctly"""
        X_train, X_test, y_train, y_test = train_test_data

        model, scaler, X_train_scaled, X_test_scaled = train_best_model(
            X_train, y_train, X_test, y_test
        )

        # Training data should have mean≈0, std≈1
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)


class TestGenerateShapExplanations:
    """Test suite for generate_shap_explanations function"""

    @patch('tasacion_portal.explain_model.shap.LinearExplainer')
    @patch('tasacion_portal.explain_model.plt')
    def test_generate_shap_explanations_creates_explainer(
        self, mock_plt, mock_explainer_class, train_test_data, temp_test_dir
    ):
        """Test SHAP explainer creation"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        feature_names = ['bedrooms', 'bathrooms', 'surface_useful']

        # Mock the explainer
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(len(X_test), len(feature_names))
        mock_explainer.expected_value = 7000.0
        mock_explainer_class.return_value = mock_explainer

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            shap_values, explainer = generate_shap_explanations(
                model, X_train.values, X_test.values, feature_names
            )

            # Check that explainer was created
            mock_explainer_class.assert_called_once()
            mock_explainer.shap_values.assert_called_once()

    @patch('tasacion_portal.explain_model.shap.LinearExplainer')
    @patch('tasacion_portal.explain_model.shap.summary_plot')
    @patch('tasacion_portal.explain_model.plt')
    def test_generate_shap_saves_files(
        self, mock_plt, mock_summary_plot, mock_explainer_class,
        train_test_data, temp_test_dir
    ):
        """Test that SHAP analysis saves output files"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        feature_names = ['bedrooms', 'bathrooms', 'surface_useful']

        # Mock explainer
        mock_explainer = Mock()
        shap_values = np.random.rand(len(X_test), len(feature_names))
        mock_explainer.shap_values.return_value = shap_values
        mock_explainer.expected_value = 7000.0
        mock_explainer_class.return_value = mock_explainer

        # Mock matplotlib
        mock_plt.figure.return_value = Mock()
        mock_plt.savefig.return_value = None

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            returned_shap_values, returned_explainer = generate_shap_explanations(
                model, X_train.values, X_test.values, feature_names
            )

            # Check CSV files are created
            shap_csv = temp_test_dir / 'outputs' / 'data' / 'shap_values.csv'
            importance_csv = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'

            assert shap_csv.exists()
            assert importance_csv.exists()

    def test_generate_shap_feature_importance_calculation(self, train_test_data, temp_test_dir):
        """Test SHAP feature importance calculation"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        feature_names = list(X_train.columns)

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.explain_model.shap.LinearExplainer') as mock_explainer_class:
                with patch('tasacion_portal.explain_model.plt'):
                    with patch('tasacion_portal.explain_model.shap.summary_plot'):
                        # Mock explainer
                        mock_explainer = Mock()
                        shap_values = np.array([[1.0, -0.5, 2.0], [0.5, 0.3, 1.5]])
                        mock_explainer.shap_values.return_value = shap_values
                        mock_explainer.expected_value = 7000.0
                        mock_explainer_class.return_value = mock_explainer

                        generate_shap_explanations(
                            model, X_train.values, X_test.values, feature_names
                        )

                        # Verify importance CSV is created with correct structure
                        importance_csv = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'
                        assert importance_csv.exists()

                        df = pd.read_csv(importance_csv)
                        assert 'feature' in df.columns
                        assert 'mean_abs_shap' in df.columns
                        assert len(df) == len(feature_names)


class TestGenerateLimeExplanations:
    """Test suite for generate_lime_explanations function"""

    @patch('tasacion_portal.explain_model.lime_tabular.LimeTabularExplainer')
    @patch('tasacion_portal.explain_model.plt')
    def test_generate_lime_explanations_creates_explainer(
        self, mock_plt, mock_explainer_class, train_test_data, temp_test_dir
    ):
        """Test LIME explainer creation"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        scaler = StandardScaler().fit(X_train)
        feature_names = list(X_train.columns)

        # Create necessary directories
        (temp_test_dir / 'outputs' / 'data').mkdir(parents=True, exist_ok=True)
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock explainer
        mock_explainer = Mock()
        mock_exp = Mock()
        mock_exp.as_list.return_value = [('bedrooms <= 3', 500), ('bathrooms > 2', 300)]
        mock_exp.as_pyplot_figure.return_value = Mock()
        mock_explainer.explain_instance.return_value = mock_exp
        mock_explainer_class.return_value = mock_explainer

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            lime_df = generate_lime_explanations(
                model, scaler, X_train, X_test, y_test, feature_names
            )

            # Check explainer was created
            mock_explainer_class.assert_called_once()
            assert isinstance(lime_df, pd.DataFrame)

    @patch('tasacion_portal.explain_model.lime_tabular.LimeTabularExplainer')
    @patch('tasacion_portal.explain_model.plt')
    def test_generate_lime_saves_files(
        self, mock_plt, mock_explainer_class, train_test_data, temp_test_dir
    ):
        """Test that LIME analysis saves output files"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        scaler = StandardScaler().fit(X_train)
        feature_names = list(X_train.columns)

        # Create necessary directories
        (temp_test_dir / 'outputs' / 'data').mkdir(parents=True, exist_ok=True)
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock explainer
        mock_explainer = Mock()
        mock_exp = Mock()
        mock_exp.as_list.return_value = [('bedrooms <= 3', 500)]
        mock_fig = Mock()
        mock_fig.set_size_inches = Mock()
        mock_exp.as_pyplot_figure.return_value = mock_fig
        mock_explainer.explain_instance.return_value = mock_exp
        mock_explainer_class.return_value = mock_explainer

        # Mock matplotlib
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.title = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            lime_df = generate_lime_explanations(
                model, scaler, X_train, X_test, y_test, feature_names
            )

            # Check CSV is created
            lime_csv = temp_test_dir / 'outputs' / 'data' / 'lime_explanations.csv'
            assert lime_csv.exists()

            # Check DataFrame has expected columns
            assert 'sample' in lime_df.columns
            assert 'actual_price' in lime_df.columns
            assert 'predicted_price' in lime_df.columns
            assert 'error' in lime_df.columns

    def test_lime_prediction_function(self, train_test_data):
        """Test LIME prediction function handles scaling correctly"""
        X_train, X_test, y_train, y_test = train_test_data

        model = LinearRegression().fit(X_train, y_train)
        scaler = StandardScaler().fit(X_train)

        # Create prediction function (same as in actual code)
        def predict_fn(X):
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)

        # Test prediction
        predictions = predict_fn(X_test.values)

        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestCreateSummaryReport:
    """Test suite for create_summary_report function"""

    @patch('tasacion_portal.explain_model.print')
    def test_create_summary_report(self, mock_print, temp_test_dir):
        """Test summary report creation"""
        from tasacion_portal.explain_model import create_summary_report

        # Create mock data files
        shap_importance_path = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'
        shap_importance_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df = pd.DataFrame({
            'feature': ['surface_useful', 'bedrooms', 'bathrooms'],
            'mean_abs_shap': [500.0, 200.0, 100.0]
        })
        shap_df.to_csv(shap_importance_path, index=False)

        lime_df = pd.DataFrame({
            'sample': [1, 2, 3],
            'actual_price': [5000, 7000, 9000],
            'predicted_price': [5100, 6900, 9100],
            'error': [100, -100, 100]
        })

        feature_names = ['bedrooms', 'bathrooms', 'surface_useful']

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            create_summary_report(None, lime_df, feature_names)

            # Check report file is created
            report_path = temp_test_dir / 'outputs' / 'reports' / 'interpretability_report.txt'
            assert report_path.exists()

            # Check report contents
            with open(report_path, 'r') as f:
                report_text = f.read()
                assert 'SHAP ANALYSIS' in report_text
                assert 'LIME ANALYSIS' in report_text
                assert 'KEY INSIGHTS' in report_text


class TestModelExplainIntegration:
    """Integration tests for model explanation pipeline"""

    def test_full_explanation_pipeline(self, temp_test_dir, sample_processed_data):
        """Test complete explanation pipeline"""
        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            # Load data
            X, y = load_and_prepare_data()

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

            # Train model
            model, scaler, X_train_scaled, X_test_scaled = train_best_model(
                X_train, y_train, X_test, y_test
            )

            # Verify model works
            assert model is not None
            predictions = model.predict(X_test_scaled)
            assert len(predictions) == len(y_test)

    @patch('tasacion_portal.explain_model.shap.LinearExplainer')
    @patch('tasacion_portal.explain_model.lime_tabular.LimeTabularExplainer')
    @patch('tasacion_portal.explain_model.plt')
    def test_explanations_with_mocked_libraries(
        self, mock_plt, mock_lime_class, mock_shap_class,
        temp_test_dir, sample_processed_data
    ):
        """Test explanation generation with mocked SHAP/LIME"""
        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        # Mock SHAP
        mock_shap_explainer = Mock()
        mock_shap_explainer.shap_values.return_value = np.random.rand(2, 3)
        mock_shap_explainer.expected_value = 7000.0
        mock_shap_class.return_value = mock_shap_explainer

        # Mock LIME
        mock_lime_explainer = Mock()
        mock_lime_exp = Mock()
        mock_lime_exp.as_list.return_value = [('feature', 100)]
        mock_lime_exp.as_pyplot_figure.return_value = Mock()
        mock_lime_explainer.explain_instance.return_value = mock_lime_exp
        mock_lime_class.return_value = mock_lime_explainer

        # Mock matplotlib
        mock_plt.figure.return_value = Mock()
        mock_plt.savefig = Mock()

        with patch('tasacion_portal.explain_model.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.explain_model.shap.summary_plot'):
                # Load and process data
                X, y = load_and_prepare_data()
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

                model, scaler, X_train_scaled, X_test_scaled = train_best_model(
                    X_train, y_train, X_test, y_test
                )

                feature_names = list(X.columns)

                # Generate explanations
                shap_values, shap_explainer = generate_shap_explanations(
                    model, X_train_scaled, X_test_scaled, feature_names
                )

                lime_df = generate_lime_explanations(
                    model, scaler, X_train, X_test, y_test, feature_names
                )

                # Verify outputs
                assert shap_values is not None
                assert isinstance(lime_df, pd.DataFrame)
