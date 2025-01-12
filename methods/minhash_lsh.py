from collections import defaultdict
import hashlib
import numpy as np

class MinHash:
    """
    MinHash class for computing MinHash signatures.
    """
    def __init__(self, num_hashes=128):
        """
        Initialize the MinHash object with the specified number of hash functions.
        """
        self.num_hashes = num_hashes
        self.hash_functions = self._generate_hash_functions(num_hashes)

    def _generate_hash_functions(self, num_hashes):
        """
        Generate a list of random hash functions.

        Args:
            num_hashes (int): The number of hash functions to generate.

        Returns:
            list: A list of hash functions, where each function maps a word to a hash value.
        """
        # Define the maximum hash value
        max_hash = (1 << 32) - 1
        # Initialize the list of hash functions
        functions = []
        # Generate the specified number of hash functions
        for _ in range(num_hashes):
            # Generate random coefficients for the hash function
            a, b = np.random.randint(1, max_hash, size=2)
            # Append the hash function to the list
            functions.append(lambda x, a=a, b=b: (a * int(hashlib.md5(x.encode()).hexdigest(), 16) + b) % max_hash)

        return functions

    def compute_signature(self, features):
        """
        Compute the MinHash signature for a given set of features.

        Args:
            features (dict): A dictionary of features (e.g., tokens from a document).

        Returns:
            list: The MinHash signature, represented as a list of minimum hash values.
        """
        # Initialize the signature list
        signature = []
        # Apply each hash function to all features and take the minimum hash value
        for h in self.hash_functions:
            # Compute the minimum hash value for each feature
            min_hash = min(h(feature) for feature in features)
            # Append the minimum hash value to the signature
            signature.append(min_hash)

        return signature

def jaccard_similarity(signature1, signature2):
    """
    Compute the Jaccard similarity between two MinHash signatures.

    Args:
        signature1 (list): The first MinHash signature.
        signature2 (list): The second MinHash signature.

    Returns:
        float: The Jaccard similarity between the two signatures.
    """
    # Compute the intersection of the two signatures
    intersection = sum(1 for i in range(len(signature1)) if signature1[i] == signature2[i])
    # Compute the union of the two signatures
    union = len(signature1) + len(signature2) - intersection
    # If the union is zero, return 0
    if union == 0:
        return 0
    # Return the Jaccard similarity
    return intersection / union

class LocalitySensitiveHashing:
    """
    Locality Sensitive Hashing (LSH) class for finding similar items efficiently.
    """
    def __init__(self, num_bands=16, rows_per_band=8):
        """
        Initialize the LSH object with the specified number of bands and rows per band.
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [defaultdict(list) for _ in range(num_bands)]

    def insert(self, identifier, signature):
        """
        Insert a document's signature into the LSH structure.

        Args:
            identifier (str): The identifier (e.g., URL) of the document.
            signature (list): The signature of the document.
        """
        # Iterate over the bands
        for i in range(self.num_bands):
            # Compute the start and end indices for the current band
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            # Extract the band from the signature
            band = tuple(signature[start:end])
            # Compute the bucket ID using an MD5 hash of the band
            bucket_id = hashlib.md5(str(band).encode()).hexdigest()
            # Append the identifier to the bucket
            self.buckets[i][bucket_id].append(identifier)

    def query(self, signature):
        """
        Query for documents with similar signatures.

        Args:
            signature (list): The signature of the query document.

        Returns:
            set: A set of candidate document identifiers (e.g., URLs) that may be similar.
        """
        # Initialize the set of candidate documents
        candidates = set()

        # Iterate over the bands
        for i in range(self.num_bands):
            # Compute the start and end indices for the current band
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            # Extract the band from the signature
            band = tuple(signature[start:end])
            # Compute the bucket ID for the current band
            bucket_id = hashlib.md5(str(band).encode()).hexdigest()
            # Update the set of candidates with all documents in the bucket
            candidates.update(self.buckets[i].get(bucket_id, []))

        return candidates
