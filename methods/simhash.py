import mmh3
import numpy as np

class SimHash:
    """
    SimHash class for computing SimHash fingerprints.
    """
    def __init__(self, num_bits=64):
        """
        Initialize the SimHash object with the number of bits.
        """
        self.num_bits = num_bits
        self.mask = (1 << num_bits) - 1
        
    def compute_fingerprint(self, features):
        """
        Compute the SimHash fingerprint.

        Args:
            features (dict): Feature dictionary with weights.

        Returns:
            int: The computed SimHash fingerprint.
        """
        # Initialize a vector of zeros with the number of bits
        vector = np.zeros(self.num_bits, dtype=np.float64)
    
        # Iterate over the features and weights
        for feature, weight in features.items():
            # Compute the hash value for the feature
            hash_value = mmh3.hash(feature, seed=0, signed=False) & self.mask
            # Update the vector based on the hash value
            for i in range(self.num_bits):
                bit = (hash_value >> i) & 1
                if bit:
                    vector[i] += weight
                else:
                    vector[i] -= weight
            
        # Initialize the fingerprint
        fingerprint = 0
        # Iterate over the bits
        for i in range(self.num_bits):
            # If the bit is positive, set the corresponding bit in the fingerprint
            if vector[i] > 0:
                fingerprint |= (1 << (self.num_bits - 1 - i))
            
        return fingerprint

def hamming_distance(hash1, hash2):
    """
    Compute the Hamming distance between two fingerprints.

    Args:
        hash1 (int): The first fingerprint.
        hash2 (int): The second fingerprint.

    Returns:
        int: The Hamming distance between the two fingerprints.
    """
    # Compute the XOR of the two fingerprints
    xor = hash1 ^ hash2
    # Count the number of 1s in the binary representation of the XOR
    return bin(xor).count('1')
