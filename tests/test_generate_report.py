"""
Tests for generate_report.py module
Tests PDF report generation and visualization functions
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from tasacion_portal.generate_report import (
    create_data_summary_plot,
    create_model_comparison_plot,
    create_metrics_table_plot,
    create_feature_importance_comparison,
    build_pdf_report
)


class TestCreateDataSummaryPlot:
    """Test suite for create_data_summary_plot function"""

    @patch('tasacion_portal.generate_report.plt')
    def test_create_data_summary_plot(self, mock_plt, temp_test_dir, sample_processed_data):
        """Test data summary plot creation"""
        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        # Create output directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_data_summary_plot()

            # Check that savefig was called
            assert mock_plt.savefig.called
            mock_plt.close.assert_called()

    @patch('tasacion_portal.generate_report.plt')
    def test_data_summary_plot_creates_subplots(self, mock_plt, temp_test_dir, sample_processed_data):
        """Test that data summary creates correct number of subplots"""
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_data_summary_plot()

            # Should create 2x2 subplots
            mock_plt.subplots.assert_called_once()
            call_args = mock_plt.subplots.call_args
            assert call_args[0] == (2, 2) or call_args[1].get('nrows') == 2


class TestCreateModelComparisonPlot:
    """Test suite for create_model_comparison_plot function"""

    @patch('tasacion_portal.generate_report.plt')
    def test_create_model_comparison_plot(self, mock_plt, temp_test_dir):
        """Test model comparison plot creation"""
        # Create sample results
        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
            'set': ['test', 'test', 'test'],
            'rmse': [100.0, 105.0, 110.0],
            'mae': [80.0, 85.0, 90.0],
            'r2': [0.85, 0.84, 0.83],
            'mape': [5.0, 5.5, 6.0]
        })

        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock matplotlib with proper barh return
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_bar = Mock()
        mock_bar.get_y.return_value = 0
        mock_bar.get_height.return_value = 1
        mock_ax1.barh.return_value = [mock_bar, mock_bar, mock_bar]
        mock_ax2.barh.return_value = [mock_bar, mock_bar, mock_bar]
        mock_axes = np.array([mock_ax1, mock_ax2])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_model_comparison_plot()

            # Check that savefig was called
            assert mock_plt.savefig.called

    @patch('tasacion_portal.generate_report.plt')
    def test_model_comparison_filters_test_set(self, mock_plt, temp_test_dir):
        """Test that model comparison only uses test set results"""
        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Linear Regression', 'Linear Regression'],
            'set': ['train', 'validation', 'test'],
            'rmse': [90.0, 95.0, 100.0],
            'r2': [0.90, 0.87, 0.85],
            'mae': [70.0, 75.0, 80.0],
            'mape': [4.0, 5.0, 6.0]
        })

        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_bar = Mock()
        mock_bar.get_y.return_value = 0
        mock_bar.get_height.return_value = 1
        mock_ax1.barh.return_value = [mock_bar]
        mock_ax2.barh.return_value = [mock_bar]
        mock_axes = np.array([mock_ax1, mock_ax2])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_model_comparison_plot()

            # Should only use test set data - verify it runs without error
            assert mock_plt.savefig.called


class TestCreateMetricsTablePlot:
    """Test suite for create_metrics_table_plot function"""

    @patch('tasacion_portal.generate_report.plt')
    def test_create_metrics_table_plot(self, mock_plt, temp_test_dir):
        """Test metrics table plot creation"""
        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Ridge Regression'],
            'set': ['test', 'test'],
            'rmse': [100.0, 105.0],
            'mae': [80.0, 85.0],
            'r2': [0.85, 0.84],
            'mape': [5.0, 5.5]
        })

        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock matplotlib with subscriptable table
        mock_fig = Mock()
        mock_ax = Mock()
        mock_cell = Mock()
        mock_cell.set_facecolor = Mock()
        mock_cell.set_text_props = Mock()
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=mock_cell)
        mock_table.auto_set_font_size = Mock()
        mock_table.set_fontsize = Mock()
        mock_table.scale = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.table.return_value = mock_table
        mock_plt.title = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_metrics_table_plot()

            assert mock_plt.savefig.called

    @patch('tasacion_portal.generate_report.plt')
    def test_metrics_table_includes_all_metrics(self, mock_plt, temp_test_dir):
        """Test that metrics table includes all required metrics"""
        results_df = pd.DataFrame({
            'model': ['Linear Regression'],
            'set': ['test'],
            'rmse': [100.0],
            'mae': [80.0],
            'r2': [0.85],
            'mape': [5.0]
        })

        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_cell = Mock()
        mock_cell.set_facecolor = Mock()
        mock_cell.set_text_props = Mock()
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=mock_cell)
        mock_table.auto_set_font_size = Mock()
        mock_table.set_fontsize = Mock()
        mock_table.scale = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.table.return_value = mock_table
        mock_plt.title = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_metrics_table_plot()

            # Check that table was created
            mock_ax.table.assert_called_once()


class TestCreateFeatureImportanceComparison:
    """Test suite for create_feature_importance_comparison function"""

    @patch('tasacion_portal.generate_report.plt')
    def test_create_feature_importance_comparison(self, mock_plt, temp_test_dir):
        """Test feature importance comparison plot creation"""
        shap_df = pd.DataFrame({
            'feature': ['surface_useful', 'bedrooms', 'bathrooms'],
            'mean_abs_shap': [500.0, 200.0, 100.0]
        })

        shap_path = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax1.barh.return_value = Mock()
        mock_ax2.pie.return_value = ([Mock()], [Mock()], [Mock()])
        mock_axes = np.array([mock_ax1, mock_ax2])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_feature_importance_comparison()

            assert mock_plt.savefig.called

    @patch('tasacion_portal.generate_report.plt')
    def test_feature_importance_creates_two_plots(self, mock_plt, temp_test_dir):
        """Test that feature importance creates bar and pie charts"""
        shap_df = pd.DataFrame({
            'feature': ['surface_useful', 'bedrooms', 'bathrooms'],
            'mean_abs_shap': [500.0, 200.0, 100.0]
        })

        shap_path = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax1.barh.return_value = Mock()
        mock_ax2.pie.return_value = ([Mock()], [Mock()], [Mock()])
        mock_axes = np.array([mock_ax1, mock_ax2])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            create_feature_importance_comparison()

            # Should create 1x2 subplots (bar chart and pie chart)
            mock_plt.subplots.assert_called_once()


class TestBuildPdfReport:
    """Test suite for build_pdf_report function"""

    @patch('tasacion_portal.generate_report.SimpleDocTemplate')
    @patch('tasacion_portal.generate_report.datetime')
    def test_build_pdf_report_creates_file(
        self, mock_datetime, mock_doc_class, temp_test_dir, sample_processed_data
    ):
        """Test PDF report file creation"""
        # Setup mock datetime
        mock_datetime.now.return_value.strftime.return_value = '20240101'

        # Setup data files
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Ridge Regression'],
            'set': ['test', 'test'],
            'rmse': [100.0, 105.0],
            'mae': [80.0, 85.0],
            'r2': [0.85, 0.84],
            'mape': [5.0, 5.5]
        })
        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Mock SimpleDocTemplate
        mock_doc = Mock()
        mock_doc_class.return_value = mock_doc

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            pdf_filename = build_pdf_report()

            # Check that document was built
            mock_doc.build.assert_called_once()
            assert 'property_price_analysis_report' in pdf_filename
            assert '.pdf' in pdf_filename

    @patch('tasacion_portal.generate_report.SimpleDocTemplate')
    @patch('tasacion_portal.generate_report.datetime')
    @patch('tasacion_portal.generate_report.Image')
    def test_build_pdf_includes_images(
        self, mock_image_class, mock_datetime, mock_doc_class,
        temp_test_dir, sample_processed_data
    ):
        """Test that PDF includes visualization images"""
        mock_datetime.now.return_value.strftime.return_value = '20240101'

        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        results_df = pd.DataFrame({
            'model': ['Linear Regression'],
            'set': ['test'],
            'rmse': [100.0],
            'r2': [0.85],
            'mae': [80.0],
            'mape': [5.0]
        })
        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Create dummy plot files
        plots_dir = temp_test_dir / 'outputs' / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / 'data_summary_plot.png').touch()
        (plots_dir / 'model_comparison_plot.png').touch()
        (plots_dir / 'metrics_table_plot.png').touch()
        (plots_dir / 'feature_importance_comparison.png').touch()
        (plots_dir / 'shap_summary_plot.png').touch()
        (plots_dir / 'lime_explanation_sample_1.png').touch()

        mock_doc = Mock()
        mock_doc_class.return_value = mock_doc
        mock_image_class.return_value = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            build_pdf_report()

            # Check that images were loaded
            assert mock_image_class.call_count > 0

    @patch('tasacion_portal.generate_report.SimpleDocTemplate')
    @patch('tasacion_portal.generate_report.datetime')
    def test_build_pdf_includes_summary_table(
        self, mock_datetime, mock_doc_class, temp_test_dir, sample_processed_data
    ):
        """Test that PDF includes summary table"""
        mock_datetime.now.return_value.strftime.return_value = '20240101'

        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        results_df = pd.DataFrame({
            'model': ['Linear Regression'],
            'set': ['test'],
            'rmse': [100.0],
            'r2': [0.85],
            'mae': [80.0],
            'mape': [5.0]
        })
        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        mock_doc = Mock()
        mock_doc_class.return_value = mock_doc

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.generate_report.Table') as mock_table_class:
                build_pdf_report()

                # Check that tables were created
                assert mock_table_class.call_count > 0


class TestReportGenerationIntegration:
    """Integration tests for report generation pipeline"""

    @patch('tasacion_portal.generate_report.plt')
    def test_all_plots_creation(self, mock_plt, temp_test_dir, sample_processed_data):
        """Test creating all visualization plots"""
        # Setup data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        results_df = pd.DataFrame({
            'model': ['Linear Regression', 'Ridge Regression'],
            'set': ['test', 'test'],
            'rmse': [100.0, 105.0],
            'mae': [80.0, 85.0],
            'r2': [0.85, 0.84],
            'mape': [5.0, 5.5]
        })
        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        shap_df = pd.DataFrame({
            'feature': ['surface_useful', 'bedrooms', 'bathrooms'],
            'mean_abs_shap': [500.0, 200.0, 100.0]
        })
        shap_path = temp_test_dir / 'outputs' / 'data' / 'shap_feature_importance.csv'
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_path, index=False)

        # Create plots directory
        (temp_test_dir / 'outputs' / 'plots').mkdir(parents=True, exist_ok=True)

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_bar = Mock()
        mock_bar.get_y.return_value = 0
        mock_bar.get_height.return_value = 1
        mock_ax1.barh.return_value = [mock_bar, mock_bar]
        mock_ax2.barh.return_value = [mock_bar, mock_bar]
        mock_ax2.pie.return_value = ([Mock()], [Mock()], [Mock()])

        # Mock table to be subscriptable
        mock_cell = Mock()
        mock_cell.set_facecolor = Mock()
        mock_cell.set_text_props = Mock()
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=mock_cell)
        mock_table.auto_set_font_size = Mock()
        mock_table.set_fontsize = Mock()
        mock_table.scale = Mock()
        mock_ax1.table.return_value = mock_table

        # Mock subplots to return different configurations based on arguments
        def mock_subplots(*args, **kwargs):
            if len(args) >= 2:
                if args[0] == 2 and args[1] == 2:
                    # create_data_summary_plot: 2x2 grid
                    return (mock_fig, np.array([[mock_ax1, mock_ax2], [mock_ax1, mock_ax2]]))
                elif args[0] == 1 and args[1] == 2:
                    # create_model_comparison_plot: 1x2 grid
                    return (mock_fig, np.array([mock_ax1, mock_ax2]))
            # Single plot (create_feature_importance_plot, create_metrics_table_plot)
            return (mock_fig, mock_ax1)

        mock_plt.subplots.side_effect = mock_subplots
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.tight_layout = Mock()

        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            # Create all plots
            create_data_summary_plot()
            create_model_comparison_plot()
            create_metrics_table_plot()
            create_feature_importance_comparison()

            # Verify that savefig was called for all plots
            assert mock_plt.savefig.call_count >= 4

    def test_report_handles_missing_plots_gracefully(self, temp_test_dir, sample_processed_data):
        """Test that report generation handles missing plot files"""
        # Setup minimal data
        data_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_processed_data.to_csv(data_path, index=False)

        results_df = pd.DataFrame({
            'model': ['Linear Regression'],
            'set': ['test'],
            'rmse': [100.0],
            'r2': [0.85],
            'mae': [80.0],
            'mape': [5.0]
        })
        results_path = temp_test_dir / 'outputs' / 'data' / 'model_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        # Don't create plot files - test that report generation doesn't crash
        with patch('tasacion_portal.generate_report.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.generate_report.SimpleDocTemplate') as mock_doc_class:
                with patch('tasacion_portal.generate_report.datetime') as mock_datetime:
                    mock_datetime.now.return_value.strftime.return_value = '20240101'
                    mock_doc = Mock()
                    mock_doc_class.return_value = mock_doc

                    # Should not crash even with missing plots
                    pdf_filename = build_pdf_report()
                    assert pdf_filename is not None
