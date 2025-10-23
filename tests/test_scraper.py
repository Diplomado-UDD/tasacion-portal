"""
Tests for scraper.py module
Tests the PortalInmobiliarioScraper class and its methods
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup
import pandas as pd
from tasacion_portal.scraper import PortalInmobiliarioScraper


class TestPortalInmobiliarioScraper:
    """Test suite for PortalInmobiliarioScraper class"""

    def test_scraper_initialization(self):
        """Test scraper initialization with base URL"""
        url = "https://www.portalinmobiliario.com/venta/departamento"
        scraper = PortalInmobiliarioScraper(url)

        assert scraper.base_url == url
        assert scraper.properties == []
        assert scraper.session is not None
        assert 'User-Agent' in scraper.session.headers

    def test_fetch_page_success(self, mock_requests_response):
        """Test successful page fetching"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        with patch.object(scraper.session, 'get', return_value=mock_requests_response):
            soup = scraper.fetch_page("https://example.com/test")

            assert isinstance(soup, BeautifulSoup)
            scraper.session.get.assert_called_once_with("https://example.com/test", timeout=30)

    def test_fetch_page_error(self):
        """Test page fetching with HTTP error"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")

        with patch.object(scraper.session, 'get', return_value=mock_response):
            with pytest.raises(Exception, match="HTTP Error"):
                scraper.fetch_page("https://example.com/test")

    def test_extract_property_data_complete(self, sample_html_property):
        """Test extracting data from a complete property listing"""
        scraper = PortalInmobiliarioScraper("https://example.com")
        property_elem = sample_html_property.find('div', class_='property-item')

        data = scraper.extract_property_data(property_elem)

        assert data['title'] == 'Departamento en Venta'
        assert 'UF' in data['price'] or data['price'] != ''
        assert data['location'] == 'Vitacura, Santiago'
        assert 'dorm' in data['bedrooms'] or data['bedrooms'] != ''
        assert 'baño' in data['bathrooms'] or data['bathrooms'] != ''
        assert data['url'] != ''

    def test_extract_property_data_empty(self):
        """Test extracting data from an empty element"""
        scraper = PortalInmobiliarioScraper("https://example.com")
        empty_html = BeautifulSoup("<div></div>", 'lxml')
        property_elem = empty_html.find('div')

        data = scraper.extract_property_data(property_elem)

        # Should return dict with empty strings
        assert isinstance(data, dict)
        assert all(key in data for key in ['title', 'price', 'location', 'bedrooms', 'bathrooms'])

    def test_scrape_page(self, sample_html_page):
        """Test scraping properties from a page"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        properties = scraper.scrape_page(sample_html_page)

        assert isinstance(properties, list)
        assert len(properties) == 3
        # Check that properties have title or price
        for prop in properties:
            assert prop['title'] or prop['price']

    def test_scrape_page_no_properties(self):
        """Test scraping page with no properties"""
        scraper = PortalInmobiliarioScraper("https://example.com")
        empty_soup = BeautifulSoup("<html><body></body></html>", 'lxml')

        properties = scraper.scrape_page(empty_soup)

        assert isinstance(properties, list)
        assert len(properties) == 0

    def test_get_next_page_url(self):
        """Test pagination URL generation"""
        scraper = PortalInmobiliarioScraper("https://www.portalinmobiliario.com/venta/departamento")
        soup = BeautifulSoup("<html></html>", 'lxml')

        # First page (page 1) should give offset 49 (page 2)
        next_url = scraper.get_next_page_url(soup, current_page=1)
        assert "_Desde_49" in next_url

        # Second page should give offset 97
        next_url = scraper.get_next_page_url(soup, current_page=2)
        assert "_Desde_97" in next_url

    def test_get_next_page_url_removes_existing_pagination(self):
        """Test that existing pagination is removed from base URL"""
        scraper = PortalInmobiliarioScraper(
            "https://www.portalinmobiliario.com/venta/departamento_Desde_49"
        )
        soup = BeautifulSoup("<html></html>", 'lxml')

        next_url = scraper.get_next_page_url(soup, current_page=1)

        # Should have only one _Desde_
        assert next_url.count("_Desde_") == 1
        assert "_Desde_49" in next_url

    @patch('tasacion_portal.scraper.time.sleep')
    def test_scrape_multiple_pages(self, mock_sleep, sample_html_page, mock_requests_response):
        """Test scraping multiple pages"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        mock_requests_response.content = sample_html_page.encode()

        with patch.object(scraper.session, 'get', return_value=mock_requests_response):
            scraper.scrape(max_pages=2)

            # Should have scraped 2 pages with 3 properties each
            assert len(scraper.properties) == 6
            # Sleep should be called for each page after the first
            assert mock_sleep.call_count >= 1

    @patch('tasacion_portal.scraper.time.sleep')
    def test_scrape_stops_on_no_properties(self, mock_sleep, mock_requests_response):
        """Test scraping stops when no properties found"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        # Empty page
        mock_requests_response.content = b"<html><body></body></html>"

        with patch.object(scraper.session, 'get', return_value=mock_requests_response):
            scraper.scrape(max_pages=5)

            # Should stop immediately as no properties found
            assert len(scraper.properties) == 0
            assert scraper.session.get.call_count == 1

    @patch('tasacion_portal.scraper.time.sleep')
    def test_scrape_handles_exception(self, mock_sleep):
        """Test scraping handles exceptions gracefully"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        with patch.object(scraper.session, 'get', side_effect=Exception("Network error")):
            # Should not raise exception
            scraper.scrape(max_pages=1)
            assert len(scraper.properties) == 0

    def test_save_to_csv(self, temp_test_dir):
        """Test saving properties to CSV"""
        scraper = PortalInmobiliarioScraper("https://example.com")
        scraper.properties = [
            {'title': 'Prop 1', 'price': '5000', 'location': 'Santiago'},
            {'title': 'Prop 2', 'price': '7000', 'location': 'Vitacura'}
        ]

        csv_path = temp_test_dir / 'test_output.csv'
        scraper.save_to_csv(str(csv_path))

        assert csv_path.exists()

        # Verify CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ['title', 'price', 'location']

    def test_save_to_csv_empty_properties(self, temp_test_dir, capsys):
        """Test saving empty properties list"""
        scraper = PortalInmobiliarioScraper("https://example.com")
        scraper.properties = []

        csv_path = temp_test_dir / 'test_output.csv'
        scraper.save_to_csv(str(csv_path))

        # Should print warning and not create file
        captured = capsys.readouterr()
        assert "No properties to save" in captured.out
        assert not csv_path.exists()

    def test_extract_property_data_with_ranges(self):
        """Test extracting property with bedroom/bathroom ranges"""
        html = """
        <div class="property-item">
            <h2 class="title">Departamento</h2>
            <span class="attr">2 a 4 dorm</span>
            <span class="attr">2 - 3 baños</span>
        </div>
        """
        soup = BeautifulSoup(html, 'lxml')
        scraper = PortalInmobiliarioScraper("https://example.com")
        property_elem = soup.find('div')

        data = scraper.extract_property_data(property_elem)

        assert 'dorm' in data['bedrooms']
        assert 'baño' in data['bathrooms']

    def test_extract_property_data_url_handling(self):
        """Test URL extraction and handling"""
        scraper = PortalInmobiliarioScraper("https://example.com")

        # Test with relative URL
        html1 = '<div><a href="/venta/prop123">Link</a></div>'
        soup1 = BeautifulSoup(html1, 'lxml')
        data1 = scraper.extract_property_data(soup1.find('div'))
        assert data1['url'].startswith('https://')

        # Test with absolute URL
        html2 = '<div><a href="https://www.example.com/prop123">Link</a></div>'
        soup2 = BeautifulSoup(html2, 'lxml')
        data2 = scraper.extract_property_data(soup2.find('div'))
        assert data2['url'] == 'https://www.example.com/prop123'
