"""
Pytest Configuration and Shared Fixtures
Provides common test fixtures for all test modules
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from bs4 import BeautifulSoup


# Test data fixtures
@pytest.fixture
def sample_raw_data():
    """Sample raw scraped data"""
    return pd.DataFrame({
        'title': ['Departamento en Venta', 'Hermoso Depto', 'Amplio Departamento'],
        'price': ['DesdeUF 5.000', 'UF 7.500', 'DesdeUF 10.000'],
        'location': ['Vitacura, Santiago', 'Las Condes, Santiago', 'Providencia, Santiago'],
        'bedrooms': ['2', '3 a 4', '3'],
        'bathrooms': ['2', '2', '3 a 4'],
        'surface_total': ['80 m²', '100 m²', ''],
        'surface_useful': ['75 m² útiles', '95 m² útiles', '120 m²'],
        'url': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3'],
        'description': ['Hermoso departamento', 'Amplio y luminoso', 'Excelente ubicación']
    })


@pytest.fixture
def sample_processed_data():
    """Sample processed clean data"""
    return pd.DataFrame({
        'price': [5000.0, 7500.0, 10000.0, 6000.0, 8000.0],
        'bedrooms': [2.0, 3.0, 3.0, 2.0, 4.0],
        'bathrooms': [2.0, 2.0, 3.0, 2.0, 3.0],
        'surface_useful': [75.0, 95.0, 120.0, 80.0, 110.0]
    })


@pytest.fixture
def sample_html_property():
    """Sample HTML for a property listing"""
    html = """
    <div class="property-item">
        <h2 class="title">Departamento en Venta</h2>
        <a href="/venta/departamento/12345">Ver detalles</a>
        <span class="price">DesdeUF 5.000</span>
        <span class="location">Vitacura, Santiago</span>
        <span class="attr">2 dorm</span>
        <span class="attr">2 baños</span>
        <span class="attr">75 m² útiles</span>
        <div class="description">Hermoso departamento en Vitacura</div>
    </div>
    """
    return BeautifulSoup(html, 'lxml')


@pytest.fixture
def sample_html_page():
    """Sample HTML page with multiple properties"""
    html = """
    <html>
        <body>
            <div class="property-item">
                <h2 class="title">Departamento 1</h2>
                <a href="/prop1">Ver</a>
                <span class="price">UF 5.000</span>
            </div>
            <div class="property-item">
                <h2 class="title">Departamento 2</h2>
                <a href="/prop2">Ver</a>
                <span class="price">UF 7.000</span>
            </div>
            <div class="property-item">
                <h2 class="title">Departamento 3</h2>
                <a href="/prop3">Ver</a>
                <span class="price">UF 9.000</span>
            </div>
        </body>
    </html>
    """
    return BeautifulSoup(html, 'lxml')


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_csv_file(temp_test_dir, sample_raw_data):
    """Create temporary CSV file with sample data"""
    csv_path = temp_test_dir / 'test_data.csv'
    sample_raw_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_sklearn_model():
    """Mock sklearn model for testing"""
    from unittest.mock import Mock
    model = Mock()
    model.predict.return_value = np.array([5000.0, 7500.0, 10000.0])
    model.score.return_value = 0.85
    model.feature_importances_ = np.array([0.7, 0.2, 0.1])
    return model


@pytest.fixture
def train_test_data(sample_processed_data):
    """Split sample data into train/test sets"""
    df = sample_processed_data
    X = df[['bedrooms', 'bathrooms', 'surface_useful']]
    y = df['price']

    # Simple split: first 3 for train, last 2 for test
    X_train = X.iloc[:3]
    X_test = X.iloc[3:]
    y_train = y.iloc[:3]
    y_test = y.iloc[3:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_requests_response():
    """Mock requests response for web scraping tests"""
    from unittest.mock import Mock
    response = Mock()
    response.status_code = 200
    response.content = b"""
        <html>
            <div class="property-item">
                <h2 class="title">Test Property</h2>
                <span class="price">UF 5000</span>
            </div>
        </html>
    """
    return response


# Test configuration
@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility"""
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings('ignore')
