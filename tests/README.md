# Tests Documentation

This directory contains comprehensive unit and integration tests for the tasacion-portal project.

## Test Structure

```
tests/
├── __init__.py                  # Package initialization
├── conftest.py                  # Shared pytest fixtures and configuration
├── test_scraper.py              # Tests for web scraping functionality
├── test_process_data.py         # Tests for data processing/cleaning
├── test_train_models.py         # Tests for ML model training
├── test_explain_model.py        # Tests for SHAP/LIME explanations
├── test_generate_report.py      # Tests for PDF report generation
└── test_main.py                 # Tests for pipeline orchestration
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_scraper.py
pytest tests/test_process_data.py
pytest tests/test_train_models.py
```

### Run tests with coverage
```bash
pytest tests/ --cov=src/tasacion_portal --cov-report=html
```

### Run tests with verbose output
```bash
pytest tests/ -v
```

### Run specific test class or function
```bash
pytest tests/test_scraper.py::TestPortalInmobiliarioScraper::test_scraper_initialization
pytest tests/test_process_data.py::TestProcessPrice
```

## Test Coverage

### test_scraper.py
- **TestPortalInmobiliarioScraper**: Tests for web scraping functionality
  - Scraper initialization and configuration
  - Page fetching and parsing
  - Property data extraction
  - Pagination handling
  - CSV file saving
  - Error handling

### test_process_data.py
- **TestProcessPrice**: Tests for price string cleaning
- **TestProcessRangeWithMean**: Tests for range processing (e.g., "2 a 4")
- **TestProcessSurface**: Tests for surface area processing
- **TestRemoveOutliers**: Tests for IQR and Z-score outlier removal
- **TestProcessDataframe**: Tests for complete dataframe processing
- **TestDataProcessingIntegration**: End-to-end data processing tests

### test_train_models.py
- **TestLoadAndPrepareData**: Tests for data loading
- **TestSplitData**: Tests for train/val/test splitting
- **TestScaleFeatures**: Tests for StandardScaler feature scaling
- **TestTrainModels**: Tests for training 7 regression models
- **TestEvaluateModel**: Tests for model evaluation metrics
- **TestEvaluateAllModels**: Tests for multi-model comparison
- **TestSaveResults**: Tests for saving results to CSV
- **TestModelTrainingIntegration**: End-to-end training pipeline tests

### test_explain_model.py
- **TestLoadAndPrepareDataExplain**: Tests for data loading
- **TestTrainBestModel**: Tests for Linear Regression training
- **TestGenerateShapExplanations**: Tests for SHAP value generation
- **TestGenerateLimeExplanations**: Tests for LIME explanations
- **TestCreateSummaryReport**: Tests for interpretability report
- **TestModelExplainIntegration**: End-to-end explanation tests

### test_generate_report.py
- **TestCreateDataSummaryPlot**: Tests for data visualization
- **TestCreateModelComparisonPlot**: Tests for model comparison charts
- **TestCreateMetricsTablePlot**: Tests for metrics table generation
- **TestCreateFeatureImportanceComparison**: Tests for feature importance plots
- **TestBuildPdfReport**: Tests for PDF document generation
- **TestReportGenerationIntegration**: End-to-end report generation tests

### test_main.py
- **TestPrintStep**: Tests for step formatting
- **TestStep1ScrapeData**: Tests for scraping step
- **TestStep2ProcessData**: Tests for processing step
- **TestStep3TrainModels**: Tests for training step
- **TestStep4ExplainModel**: Tests for explanation step
- **TestStep5GenerateReport**: Tests for report generation step
- **TestMainPipeline**: Tests for complete pipeline execution
- **TestPipelineIntegration**: End-to-end integration tests
- **TestPipelineErrorHandling**: Tests for error handling

## Fixtures

The `conftest.py` file provides shared fixtures:

- **sample_raw_data**: Raw scraped property data
- **sample_processed_data**: Clean processed data ready for modeling
- **sample_html_property**: HTML for single property listing
- **sample_html_page**: HTML page with multiple properties
- **temp_test_dir**: Temporary directory for test files
- **temp_csv_file**: Temporary CSV file with sample data
- **mock_sklearn_model**: Mock scikit-learn model
- **train_test_data**: Pre-split train/test datasets
- **mock_requests_response**: Mock HTTP response for web scraping
- **reset_random_seed**: Auto-fixture for reproducible tests
- **suppress_warnings**: Suppress warnings during tests

## Test Patterns

### Mocking External Dependencies
Tests use `unittest.mock` to mock external dependencies like:
- Web requests (`requests` library)
- Matplotlib plots
- File I/O operations
- SHAP/LIME libraries
- User input

### Testing with Temporary Files
Tests use the `temp_test_dir` fixture to create temporary files:
```python
def test_example(temp_test_dir):
    csv_path = temp_test_dir / 'test.csv'
    df.to_csv(csv_path)
    # File automatically cleaned up after test
```

### Parametrized Tests
Use `@pytest.mark.parametrize` for testing multiple inputs:
```python
@pytest.mark.parametrize("input,expected", [
    ("UF 5000", 5000.0),
    ("DesdeUF 7000", 7000.0),
])
def test_example(input, expected):
    assert process_price(input) == expected
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure that:
1. All dependencies are installed (`pytest`, mocking libraries)
2. Tests are isolated and don't depend on external resources
3. Tests clean up after themselves (using fixtures)

## Adding New Tests

When adding new functionality:

1. Create corresponding test file: `test_<module_name>.py`
2. Organize tests into classes by functionality
3. Use descriptive test names: `test_<what>_<condition>_<expected>`
4. Add docstrings explaining what is being tested
5. Use appropriate fixtures from `conftest.py`
6. Mock external dependencies
7. Test both success and failure cases

Example:
```python
class TestNewFeature:
    """Test suite for new feature"""

    def test_feature_with_valid_input(self, sample_data):
        """Test feature works with valid input"""
        result = new_feature(sample_data)
        assert result is not None

    def test_feature_with_invalid_input(self):
        """Test feature handles invalid input gracefully"""
        with pytest.raises(ValueError):
            new_feature(None)
```

## Notes

- Tests use mocking extensively to avoid:
  - Network requests
  - Long-running ML model training
  - File system dependencies
  - External API calls

- Random seeds are fixed (RANDOM_SEED=42) for reproducibility

- Tests are designed to run quickly (<1 second each)

- Integration tests verify end-to-end workflows but use mocked components
