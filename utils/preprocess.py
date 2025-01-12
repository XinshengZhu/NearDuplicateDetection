from bs4 import BeautifulSoup
from collections import Counter
from math import log
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import stopwordsiso as stopwords

def load_dataset_from_db(target_data):
    """
    Loads the dataset from the SQLite database.

    Args:
        target_data (str): The name of the SQLite database.

    Returns:
        dict: A dictionary with URLs as keys and HTML content as values.
    """
    # Check if the database exists
    if not os.path.exists(f"datasets/{target_data}.db"):
        raise FileNotFoundError(f"The database file datasets/{target_data}.db does not exist.")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(f"datasets/{target_data}.db")
    cursor = conn.cursor()

    # Execute the SQL query to select all records
    cursor.execute(f'SELECT url, html FROM {target_data} ORDER BY url ASC')
    results = cursor.fetchall()

    # Create a dictionary with URLs as keys and HTML content as values
    url_html_dict = {}
    for url, html in results:
        url_html_dict[url] = html

    # Close the database connection
    conn.close()

    return url_html_dict

def extract_text_from_html(html):
    """
    Extracts the text from the HTML content.

    Args:
        html (str): The HTML content.

    Returns:
        tuple: A tuple containing the heavy text and the light text.
    """
    # Extract the text from the HTML content
    try:
        soup = BeautifulSoup(html, "html.parser")
    except:
        return []
    
    # Extract the heavy text
    heavy_text = ""
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heavy_text += tag.get_text(separator=' ') + " "
        tag.decompose()
    
    # Remove the header, footer, nav, script, style, meta, and link tags
    for tag in soup(['header', 'footer', 'nav', 'script', 'style', 'meta', 'link']):
        tag.decompose()
    
    # Replace the img tags with their alt text
    for img in soup.find_all('img'):
        if 'alt' in img.attrs:
            img.replace_with(img['alt'])
    
    # Extract the light text
    light_text = soup.get_text(separator=' ')

    # Remove non-alphabetic characters
    heavy_text = re.sub(r'[^a-zA-Z]', ' ', heavy_text)
    light_text = re.sub(r'[^a-zA-Z]', ' ', light_text)

    # Split camel case words
    heavy_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', heavy_text)
    light_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', light_text)

    return (heavy_text, light_text)

def tokenize_words_from_text(text, weight=False):
    """
    Tokenizes the words from the text.

    Args:
        text (tuple): The tuple of heavy text and light text.
        weight (bool): Whether to weight the words by schema.

    Returns:
        list: A list of words.
    """
    # Combine the heavy text and light text
    if weight:
        text = text[0] + text[0] + text[1]
    else:
        text = text[0] + text[1]
    
    # Tokenize the text
    words = word_tokenize(text)

    # Convert words to lowercase
    words = [word.lower() for word in words]

    # Remove stopwords
    stop_words = set(stopwords.stopwords("en"))
    words = [word for word in words if word not in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words

def generate_bag_of_words(words):
    """
    Generates the bag of words.

    Args:
        words (list): A list of words.

    Returns:
        dict: A dictionary with words as keys and frequencies as values.
    """
    # Generate the bag of words
    if words:
        bag = Counter(words)
        bag = dict(sorted(bag.items(), key=lambda x: x[1], reverse=True))
        return bag
    else:
        return {}

def compute_idf(bags):
    """
    Computes the inverse document frequency.

    Args:
        bags (dict): A dictionary with URLs as keys and bags of words as values.

    Returns:
        dict: A dictionary with words as keys and IDF values as values.
    """
    # Compute the term document count
    term_document_count = {}
    num_documents = len(bags)
    for bag in bags.values():
        unique_terms = set(bag.keys())
        for term in unique_terms:
            term_document_count[term] = term_document_count.get(term, 0) + 1
    # Compute the IDF
    idf = {term: log(num_documents / (1 + count)) for term, count in term_document_count.items()}
    return idf

def compute_tfidf(bag, idf):
    """
    Computes the TF-IDF.

    Args:
        bag (dict): A dictionary with words as keys and frequencies as values.
        idf (dict): A dictionary with words as keys and IDF values as values.

    Returns:
        dict: A dictionary with words as keys and TF-IDF values as values.
    """
    # Compute the total frequency
    total_freq = sum(bag.values())
    # Compute the TF-IDF
    tf_idf = {word: (freq / total_freq) * idf[word] for word, freq in bag.items() if word in idf}
    return tf_idf

def flatten_bag_of_words(bag):
    """
    Flattens the bag of words.

    Args:
        bag (dict): A dictionary with words as keys and frequencies as values.

    Returns:
        dict: A dictionary with flattened words as keys and 1s as values.
    """
    # Flatten the bag of words
    flattened = {f"{word}{i}": 1 for word, freq in bag.items() for i in range(1, freq + 1)}
    return flattened

def compute_tfidf_matrix(bags):
    """
    Computes the TF-IDF matrix using sklearn's TfidfVectorizer.

    Args:
        bags (dict): A dictionary with URLs as keys and bags of words as values.

    Returns:
        np.ndarray: The TF-IDF matrix.
    """
    # Convert bags of words to documents
    documents = []
    for bag in bags.values():
        doc = " ".join([word for word, count in bag.items() for _ in range(count)])
        documents.append(doc)
    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return tfidf_matrix
