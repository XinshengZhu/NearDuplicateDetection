import argparse
from itertools import combinations
import networkx as nx
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.preprocess import load_dataset_from_db, extract_text_from_html, tokenize_words_from_text

def compute_similarity_matrix(documents):
    """
    Compute the similarity matrix using TF-IDF.

    Args:
        documents (list of str): The documents to compute the similarity matrix for.

    Returns:
        np.ndarray: The similarity matrix.
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Compute the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def build_similarity_graph(similarity_matrix, threshold=0.8, urls=None):
    """
    Build the similarity graph using the similarity matrix.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix.
        threshold (float): The similarity threshold.
        urls (list of str): The URLs of the documents.

    Returns:
        nx.Graph: The similarity graph.
    """
    # Initialize the similarity graph
    G = nx.Graph()
    # Get the number of documents
    num_docs = len(similarity_matrix)
    # Add nodes to the graph
    for i in range(num_docs):
        G.add_node(urls[i])
    # Add edges to the graph
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(urls[i], urls[j], weight=similarity_matrix[i][j])
    
    return G

def extract_connected_components(G):
    """
    Extract connected components from the similarity graph.

    Args:
        G (nx.Graph): The similarity graph.

    Returns:
        list of list of str: The connected components.
    """
    # Extract connected components
    components = list(nx.connected_components(G))
    
    return components

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Forge pseudo labels for dataset")
    parser.add_argument("--target_data", help="Target dataset name (e.g., 'foxnews')", required=True)
    parser.add_argument("--threshold", help="Similarity threshold, default is 0.8", type=float, default=0.8)
    args = parser.parse_args()
    print()

    # Load the dataset from the database based on the target data
    url_html_dict = load_dataset_from_db(args.target_data)
    print(f"Loaded {len(url_html_dict)} HTML pages from datasets/{args.target_data}.db.")

    # Generate the documents
    documents = {}
    for url, html in tqdm(url_html_dict.items(), desc=f"Generating documents"):
        text = extract_text_from_html(html)
        words = tokenize_words_from_text(text)
        document = " ".join(words)
        if document:
            documents[url] = document
    print(f"Generated {len(documents)} documents.")
    print()

    # Compute the similarity matrix
    similarity_matrix = compute_similarity_matrix(list(documents.values()))
    # Build the similarity graph
    G = build_similarity_graph(similarity_matrix, args.threshold, list(documents.keys()))
    # Extract connected components
    components = extract_connected_components(G)

    # Identify near-duplicate pairs as pseudo labels
    similar_pairs = set()
    for component in components:
        for url1, url2 in combinations(component, 2):
            pair = tuple(sorted((url1, url2)))
            similar_pairs.add(pair)
    print(f"Identified {len(similar_pairs)} near-duplicate pairs as pseudo labels.")
    print()

    # Save the pseudo labels to a pkl file
    with open(f"labels/{args.target_data}_label.pkl", "wb") as f:
        pickle.dump(similar_pairs, f)
    print(f"Saved pseudo labels to labels/{args.target_data}_label.pkl.")
    print()
