import argparse
import json
import requests
from requests.adapters import HTTPAdapter
import sqlite3
from tqdm import tqdm
from urllib.parse import quote_plus
from urllib3.util.retry import Retry
from warcio.archiveiterator import ArchiveIterator

def init_database(target_data):
    """
    Initializes an SQLite database for storing web pages.

    Args:
        target_data (str): Name of the target dataset, used as the database name.

    Creates a table with the following columns:
        - url (TEXT, Primary Key): The URL of the web page.
        - html (TEXT): The HTML content of the web page.
    """
    # Connect to the database
    conn = sqlite3.connect(f"datasets/{target_data}.db")
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {target_data} (
            url TEXT PRIMARY KEY,
            html TEXT
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def search_cc_index(target_url, index_name="2019-51"):
    """
    Searches the Common Crawl index for pages under the specified target URL.

    Args:
        target_url (str): The base URL to search for (e.g., 'medium.com').

    Returns:
        list: A list of index records containing metadata about matching web pages.
    """
    # URL-encode the target URL
    encoded_url = quote_plus(target_url)

    # Construct the index URL
    index_url = f'http://index.commoncrawl.org/CC-MAIN-{index_name}-index?url={encoded_url}&matchType=domain&output=json'

    # Send a GET request to the index URL
    try:
        response = requests.get(index_url, headers={
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        })
    except Exception as e:
        print(f"Error fetching index records for {target_url}: {e}")
        return None

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response into a list of records
        index_records = [json.loads(index_record) 
                         for index_record in response.text.strip().split('\n')]
        
        # Filter the records based on the specified criteria
        index_records = [index_record 
                         for index_record in index_records 
                         if index_record['url'].startswith('https')
                         and index_record['mime'] == 'text/html' 
                         and index_record['status'] == '200' 
                         and index_record.get('languages') == 'eng'
                         and index_record['encoding'] == 'UTF-8']
        
        print(f"Found {len(index_records)} index records for {target_url}")
        print()
        
        return index_records
    else:
        return None
    
def fetch_pages_from_cc(index_records, target_data):
    """
    Fetches web pages from the Common Crawl WARC files based on index records.

    Args:
        index_records (list): Metadata about pages to fetch.
        target_data (str): The name of the SQLite database to store fetched pages.

    Saves the HTML content of each page to the database.
    """
    # Initialize the database connection
    conn = sqlite3.connect(f'datasets/{target_data}.db')
    cursor = conn.cursor()
    print(f"Initialized database datasets/{target_data}.db")
    print()
    # Initialize the session for making HTTP requests with retry logic for robustness
    session = requests.Session()
    retries = Retry(total=4, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Initialize the counter for skipped records
    skipped_record_count = 0

    # Use tqdm to display a progress bar
    with tqdm(total=len(index_records), desc="Fetching WARC records", unit="record") as pbar:

        # Fetch and process WARC records
        for index_record in index_records:
            # Extract the offset and length from the index record
            offset, length = int(index_record['offset']), int(index_record['length'])

            # Construct the S3 URL for the WARC file
            s3_url = f'https://data.commoncrawl.org/{index_record["filename"]}'

            # Construct the byte range for the WARC record
            byte_range = f'bytes={offset}-{offset+length-1}'

            try:
                # Send a GET request to the S3 URL with the byte range
                response = session.get(
                    s3_url,
                    headers={'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36', 'Range': byte_range},
                    stream=True
                )
                response.raise_for_status()

                # Check if the response is a partial content response
                if response.status_code == 206:
                    # Parse the WARC record to extract web page content
                    for warc_record in ArchiveIterator(response.raw):
                        if warc_record.rec_type == 'response':
                            # Extract the URL and HTML content from the record
                            url = warc_record.rec_headers.get_header('WARC-Target-URI')
                            html = warc_record.content_stream().read().decode('utf-8', errors='ignore')

                            # Insert the URL and HTML content into the database
                            cursor = conn.cursor()
                            cursor.execute(f'''
                                INSERT OR IGNORE INTO {target_data} (url, html) VALUES (?, ?)
                            ''', (url, html))   
                            conn.commit()      
            except Exception as e:
                # Increment the skipped record counter if an error occurs
                skipped_record_count += 1
            
            # Update the progress bar
            pbar.update(1)

    # Close the database connection
    conn.close()

    print(f"Saved {len(index_records) - skipped_record_count} WARC records to datasets/{target_data}.db")
    print(f"Skipped {skipped_record_count} WARC records")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Retrieve dataset from Common Crawl")
    parser.add_argument("--target_data", help="Target dataset name (e.g., 'foxnews')", required=True)
    parser.add_argument("--target_url", help="Target URL to fetch data (e.g., 'foxnews.com')", required=True)
    parser.add_argument("--index_name", help="Index name to fetch data (e.g., 'YYYY-WW'), default is 2019-51", default="2019-51")
    args = parser.parse_args()
    print()

    # Search for index records
    index_records = search_cc_index(args.target_url, args.index_name)

    # If index records are found, fetch pages from Common Crawl
    if index_records:
        # Initialize the database
        init_database(args.target_data)

        # Fetch pages from Common Crawl
        fetch_pages_from_cc(index_records, args.target_data)
    else:
        print(f"No index records found for {args.target_url}")
    print()
