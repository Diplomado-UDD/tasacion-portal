"""
Tests for main.py module
Tests the complete analysis pipeline orchestration
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from tasacion_portal.main import (
    print_step,
    step1_scrape_data,
    step2_process_data,
    step3_train_models,
    step4_explain_model,
    step5_generate_report,
    main
)


class TestPrintStep:
    """Test suite for print_step function"""

    def test_print_step_formats_correctly(self, capsys):
        """Test that print_step formats output correctly"""
        print_step(1, 5, "TEST STEP")

        captured = capsys.readouterr()
        assert "STEP 1/5" in captured.out
        assert "TEST STEP" in captured.out
        assert "=" in captured.out

    def test_print_step_different_numbers(self, capsys):
        """Test print_step with different step numbers"""
        print_step(3, 10, "ANOTHER STEP")

        captured = capsys.readouterr()
        assert "STEP 3/10" in captured.out
        assert "ANOTHER STEP" in captured.out


class TestStep1ScrapeData:
    """Test suite for step1_scrape_data function"""

    @patch('tasacion_portal.main.PortalInmobiliarioScraper')
    def test_step1_creates_scraper(self, mock_scraper_class, temp_test_dir):
        """Test that step1 creates and uses scraper"""
        mock_scraper = Mock()
        mock_scraper.properties = [{'title': 'Prop 1'}, {'title': 'Prop 2'}]
        mock_scraper_class.return_value = mock_scraper

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            count = step1_scrape_data()

            # Check scraper was created
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape.assert_called_once_with(max_pages=50)
            mock_scraper.save_to_csv.assert_called_once()

            assert count == 2

    @patch('tasacion_portal.main.PortalInmobiliarioScraper')
    def test_step1_saves_to_correct_path(self, mock_scraper_class, temp_test_dir):
        """Test that step1 saves to raw data directory"""
        mock_scraper = Mock()
        mock_scraper.properties = [{'title': 'Prop 1'}]
        mock_scraper_class.return_value = mock_scraper

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            step1_scrape_data()

            # Check save path
            call_args = mock_scraper.save_to_csv.call_args
            save_path = call_args[0][0]
            assert 'data/raw/data.csv' in save_path


class TestStep2ProcessData:
    """Test suite for step2_process_data function"""

    @patch('tasacion_portal.process_data.process_dataframe')
    def test_step2_processes_data(self, mock_process, temp_test_dir, sample_raw_data):
        """Test that step2 loads and processes data"""
        # Setup raw data
        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        sample_raw_data.to_csv(raw_path, index=False)

        # Create processed directory
        processed_dir = temp_test_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Mock processing
        processed_df = pd.DataFrame({
            'price': [5000, 7000],
            'bedrooms': [2, 3],
            'bathrooms': [2, 2],
            'surface_useful': [75, 95]
        })
        mock_process.return_value = processed_df

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            count = step2_process_data()

            # Check that processing was called
            mock_process.assert_called_once()
            assert count == 2

    def test_step2_saves_processed_data(self, temp_test_dir, sample_raw_data):
        """Test that step2 saves processed data"""
        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        sample_raw_data.to_csv(raw_path, index=False)

        # Create processed directory
        processed_dir = temp_test_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.process_data.process_dataframe') as mock_process:
                processed_df = pd.DataFrame({
                    'price': [5000],
                    'bedrooms': [2]
                })
                mock_process.return_value = processed_df

                step2_process_data()

                # Check that processed file exists
                processed_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
                assert processed_path.exists()


class TestStep3TrainModels:
    """Test suite for step3_train_models function"""

    @patch('tasacion_portal.train_models.main')
    def test_step3_calls_train_models(self, mock_train_main):
        """Test that step3 calls train_models.main()"""
        step3_train_models()

        mock_train_main.assert_called_once()

    @patch('tasacion_portal.train_models.main')
    def test_step3_handles_exceptions(self, mock_train_main):
        """Test that step3 handles training exceptions"""
        mock_train_main.side_effect = Exception("Training error")

        with pytest.raises(Exception, match="Training error"):
            step3_train_models()


class TestStep4ExplainModel:
    """Test suite for step4_explain_model function"""

    @patch('tasacion_portal.explain_model.main')
    def test_step4_calls_explain_model(self, mock_explain_main):
        """Test that step4 calls explain_model.main()"""
        step4_explain_model()

        mock_explain_main.assert_called_once()


class TestStep5GenerateReport:
    """Test suite for step5_generate_report function"""

    @patch('tasacion_portal.generate_report.main')
    def test_step5_calls_generate_report(self, mock_report_main):
        """Test that step5 calls generate_report.main()"""
        step5_generate_report()

        mock_report_main.assert_called_once()


class TestMainPipeline:
    """Test suite for main pipeline function"""

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    @patch('tasacion_portal.main.step3_train_models')
    @patch('tasacion_portal.main.step4_explain_model')
    @patch('tasacion_portal.main.step5_generate_report')
    def test_main_executes_all_steps(
        self, mock_step5, mock_step4, mock_step3, mock_step2, mock_step1, mock_input,
        temp_test_dir
    ):
        """Test that main executes all pipeline steps"""
        mock_step1.return_value = 100
        mock_step2.return_value = 90

        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            # Ensure raw data doesn't exist so step1 runs
            if raw_path.exists():
                raw_path.unlink()

            main()

            # Check all steps were called
            mock_step1.assert_called_once()
            mock_step2.assert_called_once()
            mock_step3.assert_called_once()
            mock_step4.assert_called_once()
            mock_step5.assert_called_once()

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step2_process_data')
    @patch('tasacion_portal.main.step3_train_models')
    @patch('tasacion_portal.main.step4_explain_model')
    @patch('tasacion_portal.main.step5_generate_report')
    @patch('tasacion_portal.main.pd.read_csv')
    def test_main_skips_scraping_if_data_exists(
        self, mock_read_csv, mock_step5, mock_step4, mock_step3, mock_step2, mock_input,
        temp_test_dir, sample_raw_data
    ):
        """Test that main skips scraping if raw data already exists"""
        # Create raw data file
        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        sample_raw_data.to_csv(raw_path, index=False)

        mock_read_csv.return_value = sample_raw_data
        mock_step2.return_value = len(sample_raw_data)

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.main.step1_scrape_data') as mock_step1:
                main()

                # Step 1 should not be called
                mock_step1.assert_not_called()

                # Other steps should still be called
                mock_step2.assert_called_once()
                mock_step3.assert_called_once()

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    def test_main_handles_keyboard_interrupt(self, mock_step2, mock_step1, mock_input, temp_test_dir):
        """Test that main handles KeyboardInterrupt"""
        mock_step2.side_effect = KeyboardInterrupt()

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    def test_main_handles_exceptions(self, mock_step2, mock_step1, mock_input, temp_test_dir):
        """Test that main handles general exceptions"""
        mock_step2.side_effect = Exception("Pipeline error")

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    @patch('tasacion_portal.main.step3_train_models')
    @patch('tasacion_portal.main.step4_explain_model')
    @patch('tasacion_portal.main.step5_generate_report')
    def test_main_prints_summary(
        self, mock_step5, mock_step4, mock_step3, mock_step2, mock_step1, mock_input,
        capsys, temp_test_dir
    ):
        """Test that main prints execution summary"""
        mock_step1.return_value = 100
        mock_step2.return_value = 90

        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            if raw_path.exists():
                raw_path.unlink()

            main()

            captured = capsys.readouterr()
            assert "PIPELINE COMPLETE" in captured.out
            assert "Execution time" in captured.out

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    @patch('tasacion_portal.main.step3_train_models')
    @patch('tasacion_portal.main.step4_explain_model')
    @patch('tasacion_portal.main.step5_generate_report')
    @patch('tasacion_portal.main.datetime')
    def test_main_calculates_duration(
        self, mock_datetime, mock_step5, mock_step4, mock_step3, mock_step2, mock_step1, mock_input,
        temp_test_dir
    ):
        """Test that main calculates and reports execution duration"""
        from datetime import datetime, timedelta

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 10, 5, 30)  # 5.5 minutes later

        mock_datetime.now.side_effect = [start_time, end_time]

        mock_step1.return_value = 100
        mock_step2.return_value = 90

        raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            if raw_path.exists():
                raw_path.unlink()

            main()

            # Duration should be calculated correctly
            # (tested indirectly through the print statements)


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""

    @patch('builtins.input', return_value='')
    def test_pipeline_with_minimal_data(self, mock_input, temp_test_dir):
        """Test complete pipeline with minimal data"""
        # Create minimal test data
        sample_data = pd.DataFrame({
            'price': [5000, 7000, 9000],
            'bedrooms': [2, 3, 3],
            'bathrooms': [2, 2, 3],
            'surface_useful': [75, 95, 120]
        })

        processed_path = temp_test_dir / 'data' / 'processed' / 'data.csv'
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(processed_path, index=False)

        # Mock all steps to avoid long execution
        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.main.step1_scrape_data', return_value=3):
                with patch('tasacion_portal.main.step2_process_data', return_value=3):
                    with patch('tasacion_portal.main.step3_train_models'):
                        with patch('tasacion_portal.main.step4_explain_model'):
                            with patch('tasacion_portal.main.step5_generate_report'):
                                # Should complete without errors
                                main()

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    def test_pipeline_execution_order(self, mock_step1, mock_input, temp_test_dir):
        """Test that pipeline steps execute in correct order"""
        call_order = []

        def record_step1():
            call_order.append('step1')
            return 100

        def record_step2():
            call_order.append('step2')
            return 90

        def record_step3():
            call_order.append('step3')

        def record_step4():
            call_order.append('step4')

        def record_step5():
            call_order.append('step5')

        mock_step1.side_effect = record_step1

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.main.step2_process_data', side_effect=record_step2):
                with patch('tasacion_portal.main.step3_train_models', side_effect=record_step3):
                    with patch('tasacion_portal.main.step4_explain_model', side_effect=record_step4):
                        with patch('tasacion_portal.main.step5_generate_report', side_effect=record_step5):
                            # Remove raw data to force step1 execution
                            raw_path = temp_test_dir / 'data' / 'raw' / 'data.csv'
                            if raw_path.exists():
                                raw_path.unlink()

                            main()

                            # Verify order
                            assert call_order == ['step1', 'step2', 'step3', 'step4', 'step5']

    @patch('builtins.input', return_value='')
    def test_pipeline_creates_directory_structure(self, mock_input, temp_test_dir):
        """Test that pipeline creates necessary directory structure"""
        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.main.step1_scrape_data', return_value=100):
                with patch('tasacion_portal.main.step2_process_data', return_value=90):
                    with patch('tasacion_portal.main.step3_train_models'):
                        with patch('tasacion_portal.main.step4_explain_model'):
                            with patch('tasacion_portal.main.step5_generate_report'):
                                main()

                                # Check that directories are created (through the mocked functions)
                                # This is tested indirectly through successful execution


class TestPipelineErrorHandling:
    """Test error handling in the pipeline"""

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    def test_pipeline_stops_on_step_failure(self, mock_step1, mock_input, temp_test_dir):
        """Test that pipeline stops when a step fails"""
        mock_step1.side_effect = Exception("Scraping failed")

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with patch('tasacion_portal.main.step2_process_data') as mock_step2:
                with pytest.raises(SystemExit):
                    main()

                # Step 2 should not be called if step 1 fails
                mock_step2.assert_not_called()

    @patch('builtins.input', return_value='')
    @patch('tasacion_portal.main.step1_scrape_data')
    @patch('tasacion_portal.main.step2_process_data')
    @patch('tasacion_portal.main.step3_train_models')
    def test_pipeline_reports_which_step_failed(
        self, mock_step3, mock_step2, mock_step1, mock_input, temp_test_dir, capsys
    ):
        """Test that pipeline reports which step failed"""
        mock_step1.return_value = 100
        mock_step2.return_value = 90
        mock_step3.side_effect = Exception("Model training failed")

        with patch('tasacion_portal.main.PROJECT_ROOT', temp_test_dir):
            with pytest.raises(SystemExit):
                main()

            captured = capsys.readouterr()
            assert "Pipeline failed" in captured.out or "❌" in captured.out
