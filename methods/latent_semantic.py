import numpy as np
from sklearn.decomposition import TruncatedSVD

class LatentSemanticAnalysis:
    """
    Latent Semantic Analysis (LSA) is a technique used to analyze relationships between a set of documents and the terms 
    that occur in them. It is based on the singular value decomposition (SVD) of a term-document matrix.
    """
    def __init__(self, n_components=5000):
        """
        Initialize the LSA model with the number of components to keep.
        """
        self.n_components = n_components
        self.lsa_matrix = None
        self.topic_word_matrix = None
        self.explained_variance = None

    def fit(self, tfidf_matrix):
        """
        Perform SVD on the TF-IDF matrix and compute the LSA matrix.

        Args:
            tfidf_matrix (np.ndarray): The TF-IDF matrix to decompose.

        Returns:
            np.ndarray: The LSA-transformed matrix (U * Sigma from SVD).
        """
        # Perform SVD on the TF-IDF matrix
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.lsa_matrix = svd.fit_transform(tfidf_matrix)
        self.topic_word_matrix = svd.components_
        self.explained_variance = svd.explained_variance_ratio_

    def get_lsa_matrix(self):
        """
        Get the LSA-transformed matrix (U * Sigma from SVD).

        Returns:
            np.ndarray: The LSA-transformed matrix.
        """
        return self.lsa_matrix
    
    def get_topic_word_matrix(self):
        """
        Get the topic-word matrix (VT from SVD).

        Returns:
            np.ndarray: The topic-word matrix.
        """
        return self.topic_word_matrix
    
    def get_explained_variance(self):
        """
        Get the explained variance of the components.

        Returns:
            np.ndarray: Array of explained variance ratios for the components.
        """
        return self.explained_variance

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
