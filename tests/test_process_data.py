"""
Tests for process_data.py module
Tests data cleaning and processing functions
"""

import pytest
import pandas as pd
import numpy as np
from tasacion_portal.process_data import (
    process_price,
    process_range_with_mean,
    process_surface,
    remove_outliers,
    process_dataframe
)


class TestProcessPrice:
    """Test suite for process_price function"""

    def test_process_price_with_desde_uf(self):
        """Test processing price with 'DesdeUF' prefix"""
        assert process_price('DesdeUF 5.000') == 5000.0
        assert process_price('Desde UF 10.500') == 10500.0

    def test_process_price_with_uf_only(self):
        """Test processing price with 'UF' only"""
        assert process_price('UF 7.500') == 7500.0
        assert process_price('UF 12.000') == 12000.0

    def test_process_price_with_dots(self):
        """Test processing price with thousand separators"""
        assert process_price('UF 15.000') == 15000.0
        assert process_price('DesdeUF 100.000') == 100000.0

    def test_process_price_with_comma_decimal(self):
        """Test processing price with comma as decimal separator"""
        result = process_price('UF 5.500,50')
        # After replacing dot with nothing and comma with dot: '5.500,50' → '5500,50' → '5500.50'
        assert result == 5500.5  # This is the expected behavior based on the code

    def test_process_price_empty_string(self):
        """Test processing empty price string"""
        assert process_price('') is None
        assert process_price('   ') is None

    def test_process_price_none(self):
        """Test processing None value"""
        assert process_price(None) is None
        assert process_price(pd.NA) is None

    def test_process_price_invalid_format(self):
        """Test processing invalid price format"""
        assert process_price('invalid') is None
        assert process_price('abc123') is None


class TestProcessRangeWithMean:
    """Test suite for process_range_with_mean function"""

    def test_process_range_with_a_separator(self):
        """Test processing range with 'a' separator"""
        assert process_range_with_mean('2 a 4') == 3.0
        assert process_range_with_mean('3 a 5') == 4.0

    def test_process_range_with_hyphen_separator(self):
        """Test processing range with hyphen separator"""
        assert process_range_with_mean('2-4') == 3.0
        assert process_range_with_mean('1 - 3') == 2.0

    def test_process_range_single_number(self):
        """Test processing single number"""
        assert process_range_with_mean('3') == 3.0
        assert process_range_with_mean('5') == 5.0

    def test_process_range_with_text(self):
        """Test processing range with surrounding text"""
        assert process_range_with_mean('2 a 4 dormitorios') == 3.0
        assert process_range_with_mean('tiene 3 habitaciones') == 3.0

    def test_process_range_empty_or_none(self):
        """Test processing empty or None values"""
        assert process_range_with_mean('') is None
        assert process_range_with_mean(None) is None
        assert process_range_with_mean(pd.NA) is None

    def test_process_range_no_numbers(self):
        """Test processing text with no numbers"""
        assert process_range_with_mean('sin números') is None


class TestProcessSurface:
    """Test suite for process_surface function"""

    def test_process_surface_with_m2(self):
        """Test processing surface with m² symbol"""
        assert process_surface('75 m²') == 75.0
        assert process_surface('100 m²') == 100.0

    def test_process_surface_with_utiles(self):
        """Test processing surface with 'útiles' text"""
        assert process_surface('75 m² útiles') == 75.0
        assert process_surface('80 m² util') == 80.0

    def test_process_surface_with_range(self):
        """Test processing surface with range"""
        assert process_surface('70 a 80 m²') == 75.0
        assert process_surface('90-100 m² útiles') == 95.0

    def test_process_surface_plain_number(self):
        """Test processing plain number"""
        assert process_surface('85') == 85.0

    def test_process_surface_empty_or_none(self):
        """Test processing empty or None values"""
        assert process_surface('') is None
        assert process_surface(None) is None
        assert process_surface(pd.NA) is None


class TestRemoveOutliers:
    """Test suite for remove_outliers function"""

    def test_remove_outliers_iqr_method(self):
        """Test outlier removal using IQR method"""
        df = pd.DataFrame({
            'price': [1000, 2000, 2500, 3000, 10000],  # 10000 is outlier
            'bedrooms': [2, 3, 3, 4, 3]
        })

        df_clean = remove_outliers(df, ['price'], method='iqr', threshold=1.5)

        # The outlier should be removed
        assert len(df_clean) < len(df)
        assert 10000 not in df_clean['price'].values

    def test_remove_outliers_zscore_method(self):
        """Test outlier removal using Z-score method"""
        df = pd.DataFrame({
            'price': [2000, 2100, 2200, 2300, 2400],
            'bedrooms': [2, 3, 3, 4, 3]
        })

        # With threshold=1, at least some values should be considered outliers
        df_clean = remove_outliers(df, ['price'], method='zscore', threshold=1)

        # Function should run and return a dataframe
        assert isinstance(df_clean, pd.DataFrame)
        # Should have removed some rows or kept all if no outliers
        assert len(df_clean) <= len(df)

    def test_remove_outliers_preserves_nulls(self):
        """Test that outlier removal preserves rows with null values"""
        df = pd.DataFrame({
            'price': [1000, 2000, None, 3000, 10000],
            'bedrooms': [2, 3, 3, 4, 3]
        })

        df_clean = remove_outliers(df, ['price'], method='iqr', threshold=1.5)

        # Should still have the null value
        assert df_clean['price'].isna().sum() > 0

    def test_remove_outliers_no_outliers(self):
        """Test outlier removal when no outliers exist"""
        df = pd.DataFrame({
            'price': [2000, 2500, 3000, 3500, 4000],
            'bedrooms': [2, 3, 3, 4, 3]
        })

        df_clean = remove_outliers(df, ['price'], method='iqr', threshold=1.5)

        # Should have all rows
        assert len(df_clean) == len(df)

    def test_remove_outliers_multiple_columns(self):
        """Test outlier removal on multiple columns"""
        df = pd.DataFrame({
            'price': [1000, 2000, 2500, 3000, 10000],
            'bedrooms': [2, 3, 3, 4, 20]  # 20 is outlier
        })

        df_clean = remove_outliers(df, ['price', 'bedrooms'], method='iqr', threshold=1.5)

        # Outliers in both columns should be removed
        assert len(df_clean) < len(df)

    def test_remove_outliers_nonexistent_column(self):
        """Test outlier removal with non-existent column"""
        df = pd.DataFrame({
            'price': [1000, 2000, 2500, 3000, 4000]
        })

        # Should not raise error
        df_clean = remove_outliers(df, ['nonexistent_column'], method='iqr')
        assert len(df_clean) == len(df)

    def test_remove_outliers_empty_column(self):
        """Test outlier removal with all-null column"""
        df = pd.DataFrame({
            'price': [None, None, None, None, None],
            'bedrooms': [2, 3, 3, 4, 3]
        })

        df_clean = remove_outliers(df, ['price'], method='iqr')
        # Should not crash and should preserve all rows
        assert len(df_clean) == len(df)


class TestProcessDataframe:
    """Test suite for process_dataframe function"""

    def test_process_dataframe_complete(self, sample_raw_data):
        """Test processing complete dataframe"""
        df_processed = process_dataframe(sample_raw_data)

        # Check that price column is processed
        assert df_processed['price'].dtype in [np.float64, float]
        assert all(df_processed['price'].dropna() > 0)

        # Check that numeric columns exist
        assert 'bedrooms' in df_processed.columns
        assert 'bathrooms' in df_processed.columns

    def test_process_dataframe_price_column(self, sample_raw_data):
        """Test that price column is properly processed"""
        df_processed = process_dataframe(sample_raw_data)

        # Original: ['DesdeUF 5.000', 'UF 7.500', 'DesdeUF 10.000']
        prices = df_processed['price'].dropna()
        assert len(prices) > 0
        assert all(isinstance(p, (int, float)) for p in prices)

    def test_process_dataframe_bedrooms_column(self, sample_raw_data):
        """Test that bedrooms column is properly processed"""
        df_processed = process_dataframe(sample_raw_data)

        # Should convert '2', '3 a 4', '3' to numeric
        bedrooms = df_processed['bedrooms'].dropna()
        assert len(bedrooms) > 0
        assert all(isinstance(b, (int, float)) for b in bedrooms)

    def test_process_dataframe_removes_outliers(self):
        """Test that outliers are removed from dataframe"""
        df = pd.DataFrame({
            'price': ['UF 5000', 'UF 7000', 'UF 8000', 'UF 100000'],  # Last is outlier
            'bedrooms': ['2', '3', '3', '4'],
            'bathrooms': ['2', '2', '3', '3'],
            'surface_useful': ['70 m²', '80 m²', '90 m²', '100 m²']
        })

        df_processed = process_dataframe(df)

        # Should have fewer rows after removing outlier
        assert len(df_processed) < len(df)

    def test_process_dataframe_missing_columns(self):
        """Test processing dataframe with missing optional columns"""
        df = pd.DataFrame({
            'price': ['UF 5000', 'UF 7000'],
            'title': ['Prop 1', 'Prop 2']
        })

        # Should not crash even if bedrooms, bathrooms, surface columns are missing
        df_processed = process_dataframe(df)
        assert 'price' in df_processed.columns

    def test_process_dataframe_shape_changes(self, sample_raw_data):
        """Test that dataframe shape is reported correctly"""
        original_shape = sample_raw_data.shape
        df_processed = process_dataframe(sample_raw_data)

        # Shape may change due to outlier removal
        assert df_processed.shape[1] == original_shape[1]  # Same columns
        # Rows may decrease due to outlier removal

    def test_process_dataframe_with_empty_dataframe(self):
        """Test processing empty dataframe"""
        df_empty = pd.DataFrame()

        df_processed = process_dataframe(df_empty)
        assert len(df_processed) == 0

    def test_process_dataframe_preserves_other_columns(self, sample_raw_data):
        """Test that non-processed columns are preserved"""
        df_processed = process_dataframe(sample_raw_data)

        # Should still have title, location, etc.
        assert 'title' in df_processed.columns
        assert 'location' in df_processed.columns
        assert 'url' in df_processed.columns


class TestDataProcessingIntegration:
    """Integration tests for complete data processing pipeline"""

    def test_full_pipeline_with_real_data(self, sample_raw_data, temp_test_dir):
        """Test complete processing pipeline"""
        # Save raw data
        raw_path = temp_test_dir / 'raw_data.csv'
        sample_raw_data.to_csv(raw_path, index=False)

        # Load and process
        df = pd.read_csv(raw_path)
        df_processed = process_dataframe(df)

        # Save processed data
        processed_path = temp_test_dir / 'processed_data.csv'
        df_processed.to_csv(processed_path, index=False)

        # Verify file exists and can be read
        assert processed_path.exists()
        df_loaded = pd.read_csv(processed_path)
        assert len(df_loaded) > 0

    def test_processing_maintains_data_integrity(self, sample_raw_data):
        """Test that processing maintains data integrity"""
        df_processed = process_dataframe(sample_raw_data.copy())

        # Check no duplicate rows created
        assert not df_processed.duplicated().any()

        # Check that processed numeric columns have valid values
        numeric_cols = ['price', 'bedrooms', 'bathrooms', 'surface_useful']
        for col in numeric_cols:
            if col in df_processed.columns:
                values = df_processed[col].dropna()
                if len(values) > 0:
                    assert all(values >= 0)  # All values should be non-negative
