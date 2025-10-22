"""
Portal Inmobiliario Scraper
Scrapes property listings from portalinmobiliario.com and saves to CSV
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import re


class PortalInmobiliarioScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.properties = []

    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page"""
        print(f"Fetching: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'lxml')

    def extract_property_data(self, property_element) -> Dict:
        """Extract data from a single property listing"""
        data = {
            'title': '',
            'price': '',
            'location': '',
            'bedrooms': '',
            'bathrooms': '',
            'surface_total': '',
            'surface_useful': '',
            'url': '',
            'description': ''
        }

        try:
            # Extract title
            title_elem = property_element.find(['h2', 'h3', 'a'], class_=re.compile('title|property-title|heading', re.I))
            if title_elem:
                data['title'] = title_elem.get_text(strip=True)

            # Extract URL
            link_elem = property_element.find('a', href=True)
            if link_elem:
                href = link_elem['href']
                data['url'] = href if href.startswith('http') else f"https://www.portalinmobiliario.com{href}"

            # Extract price
            price_elem = property_element.find(class_=re.compile('price', re.I))
            if price_elem:
                data['price'] = price_elem.get_text(strip=True)

            # Extract location
            location_elem = property_element.find(class_=re.compile('location|address', re.I))
            if location_elem:
                data['location'] = location_elem.get_text(strip=True)

            # Extract attributes (bedrooms, bathrooms, surface)
            attrs = property_element.find_all(class_=re.compile('attr|feature|specs', re.I))
            for attr in attrs:
                text = attr.get_text(strip=True).lower()
                if 'dorm' in text or 'hab' in text:
                    data['bedrooms'] = text
                elif 'baño' in text:
                    data['bathrooms'] = text
                elif 'm²' in text or 'm2' in text:
                    if 'total' in text:
                        data['surface_total'] = text
                    elif 'útil' in text or 'util' in text:
                        data['surface_useful'] = text
                    elif not data['surface_total']:
                        data['surface_total'] = text

            # Extract description
            desc_elem = property_element.find(class_=re.compile('description|summary', re.I))
            if desc_elem:
                data['description'] = desc_elem.get_text(strip=True)

        except Exception as e:
            print(f"Error extracting property data: {e}")

        return data

    def scrape_page(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all properties from a page"""
        properties = []

        # Try different common selectors for property listings
        possible_selectors = [
            {'class_': re.compile('property-item|listing-item|item-card', re.I)},
            {'class_': re.compile('ad-card|advert-card', re.I)},
            {'attrs': {'data-id': True}},
            {'class_': re.compile('ui-search-result', re.I)},
        ]

        property_elements = []
        for selector in possible_selectors:
            property_elements = soup.find_all('div', **selector)
            if property_elements:
                print(f"Found {len(property_elements)} properties using selector: {selector}")
                break

        # Fallback: try to find article tags
        if not property_elements:
            property_elements = soup.find_all('article')
            if property_elements:
                print(f"Found {len(property_elements)} properties using <article> tags")

        for prop_elem in property_elements:
            property_data = self.extract_property_data(prop_elem)
            if property_data['title'] or property_data['price']:  # Only add if we found some data
                properties.append(property_data)

        return properties

    def get_next_page_url(self, soup: BeautifulSoup, current_page: int) -> str:
        """Find the URL for the next page"""
        # Portal Inmobiliario uses _Desde_ pattern for pagination
        # Each page shows 48 items, so page 2 starts at item 49
        items_per_page = 48
        next_offset = (current_page * items_per_page) + 1

        # Remove any existing pagination from base URL
        base = self.base_url.split('_Desde_')[0]

        return f"{base}_Desde_{next_offset}"

    def scrape(self, max_pages: int = 10):
        """Scrape multiple pages of listings"""
        current_url = self.base_url
        page_num = 1

        while current_url and page_num <= max_pages:
            try:
                soup = self.fetch_page(current_url)
                page_properties = self.scrape_page(soup)

                if not page_properties:
                    print(f"No properties found on page {page_num}. Stopping.")
                    break

                self.properties.extend(page_properties)
                print(f"Page {page_num}: Found {len(page_properties)} properties. Total: {len(self.properties)}")

                # Get next page URL
                current_url = self.get_next_page_url(soup, page_num)
                page_num += 1

                # Be respectful - add delay between requests
                if current_url:
                    time.sleep(2)

            except Exception as e:
                print(f"Error on page {page_num}: {e}")
                break

    def save_to_csv(self, filename: str = 'data.csv'):
        """Save scraped properties to CSV"""
        if not self.properties:
            print("No properties to save!")
            return

        df = pd.DataFrame(self.properties)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(self.properties)} properties to {filename}")
        print(f"Columns: {list(df.columns)}")


def main():
    url = "https://www.portalinmobiliario.com/venta/departamento/vitacura-metropolitana"

    print("Starting Portal Inmobiliario scraper...")
    scraper = PortalInmobiliarioScraper(url)

    # Scrape multiple pages (adjust max_pages as needed)
    scraper.scrape(max_pages=50)

    # Save to CSV
    scraper.save_to_csv('data.csv')


if __name__ == "__main__":
    main()