#!/usr/bin/env python3
"""
Tesla OTA Update Scraping Pipeline
Bachelor Thesis - Amine
============================================================
This script automatically scrapes Tesla Model 3 software update
release notes from notateslaapp.com and exports them to Excel/CSV.

Usage:
    Run in Google Colab or locally with Python 3.8+
    pip install requests beautifulsoup4 pandas openpyxl

Output:
    - tesla_scraped_updates.xlsx
    - tesla_scraped_updates.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_URL = "https://www.notateslaapp.com"
UPDATES_URL = f"{BASE_URL}/software-updates/"
OUTPUT_EXCEL = "tesla_scraped_updates.xlsx"
OUTPUT_CSV = "tesla_scraped_updates.csv"
DELAY = 2  # seconds between requests (be polite to the server)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

# ============================================================
# STEP 1: Get list of all versions
# ============================================================
def get_all_versions():
    print("Fetching list of all Tesla software updates...")
    
    try:
        response = requests.get(UPDATES_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching updates list: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    versions = []
    
    # Find all version links
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        if '/software-updates/version/' in href:
            # Extract version number from URL
            version_match = re.search(r'/software-updates/version/([^/]+)/', href)
            if version_match:
                version = version_match.group(1)
                full_url = BASE_URL + href if href.startswith('/') else href
                versions.append({
                    'version': version,
                    'url': full_url
                })
    
    # Remove duplicates
    seen = set()
    unique_versions = []
    for v in versions:
        if v['version'] not in seen:
            seen.add(v['version'])
            unique_versions.append(v)
    
    print(f"Found {len(unique_versions)} unique versions")
    return unique_versions

# ============================================================
# STEP 2: Scrape individual version page
# ============================================================
def scrape_version(version_info):
    version = version_info['version']
    url = version_info['url']
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {version}: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract release date
    release_date = None
    date_elem = soup.find('time') or soup.find(class_=re.compile('date|time', re.I))
    if date_elem:
        release_date = date_elem.get_text(strip=True)
    
    # Extract release notes text
    release_note = None
    
    # Try different selectors for release notes
    note_selectors = [
        {'class': re.compile('release.note|update.note|note', re.I)},
        {'class': re.compile('content|description', re.I)},
        'article',
        'main'
    ]
    
    for selector in note_selectors:
        if isinstance(selector, dict):
            elem = soup.find(attrs=selector)
        else:
            elem = soup.find(selector)
        if elem:
            release_note = elem.get_text(separator='\n', strip=True)
            break
    
    # Fallback: get all paragraph text
    if not release_note:
        paragraphs = soup.find_all('p')
        release_note = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    
    # Parse version format (YYYY.WW.SUB)
    version_parts = version.split('.')
    year = version_parts[0] if len(version_parts) > 0 else None
    
    # Determine quarter
    quarter = None
    if release_date:
        try:
            # Try to parse date
            for fmt in ['%B %d, %Y', '%d %B %Y', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(release_date, fmt)
                    quarter = f"Q{(dt.month - 1) // 3 + 1}"
                    year = str(dt.year)
                    break
                except:
                    continue
        except:
            pass
    
    return {
        'update_name': version,
        'release_date': release_date,
        'year': year,
        'quarter': quarter,
        'source_url': url,
        'release_note': release_note,
        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ============================================================
# STEP 3: Parse YYYY.WW format
# ============================================================
def parse_version_format(version):
    """Extract YYYY.WW from full version string"""
    parts = version.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version

# ============================================================
# STEP 4: Main scraping function
# ============================================================
def scrape_all(max_versions=None):
    """
    Scrape all Tesla software updates
    
    Args:
        max_versions: limit number of versions to scrape (None = all)
    """
    print("=" * 60)
    print("Tesla OTA Update Scraping Pipeline")
    print("=" * 60)
    
    # Get version list
    versions = get_all_versions()
    
    if not versions:
        print("No versions found. Please check the website.")
        return None
    
    if max_versions:
        versions = versions[:max_versions]
        print(f"Limiting to {max_versions} versions")
    
    # Scrape each version
    results = []
    for i, version_info in enumerate(versions, 1):
        print(f"[{i}/{len(versions)}] Scraping {version_info['version']}...")
        
        data = scrape_version(version_info)
        if data:
            data['yyyy_ww'] = parse_version_format(version_info['version'])
            results.append(data)
            print(f"  ✅ Done - {len(data.get('release_note', '') or '')} chars")
        else:
            print(f"  ❌ Failed")
        
        # Be polite - wait between requests
        time.sleep(DELAY)
    
    return results

# ============================================================
# STEP 5: Export results
# ============================================================
def export_results(results):
    if not results:
        print("No results to export")
        return
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['yyyy_ww', 'update_name', 'release_date', 'year', 'quarter',
            'source_url', 'release_note', 'scraped_at']
    df = df[[c for c in cols if c in df.columns]]
    
    # Export to Excel
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"✅ Excel saved: {OUTPUT_EXCEL}")
    
    # Export to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ CSV saved: {OUTPUT_CSV}")
    
    print(f"\nTotal versions scraped: {len(df)}")
    print(f"Versions with release notes: {df['release_note'].notna().sum()}")
    
    return df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Scrape all versions (remove max_versions limit for full scrape)
    results = scrape_all(max_versions=None)
    
    if results:
        df = export_results(results)
        print("\n✅ Scraping complete!")
    else:
        print("\n❌ Scraping failed. Please check your internet connection.")
