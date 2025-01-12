# Scalable Near-Duplicate Detection for Web Mining

## Overview

This project is written and tested entirely on macOS Sequoia throughout the development process.
We perform near-duplicate detection using three different approaches:
- **SimHash**: A dimensionality reduction technique that generates bit-wise fingerprints by hashing their features (terms or n-grams) and combining the hash values through weighted bit-voting to identify near-duplicate pairs based on a threshold of Hamming distance.
- **MinHash with Locality Sensitive Hashing**: A two-stage probabilistic approach that first computes $k$ hash functions to estimate Jaccard similarity as signatures between document sets (shingles or terms), then uses LSH with $b$ bands of $r$ rows to efficiently find candidate pairs above a threshold.
- **Latent Semantic Analysis**: A linear algebra technique that decomposes the TF-IDF weighted term-document matrix using singular value decomposition (SVD), reduces dimensionality by keeping only k largest singular values, and measures the cosine similarity between document vectors in the reduced semantic space.

## Files

Here's a brief description of all code files of the project:

```
NearDuplicateDetection/
├── datasets/                       # Dictionary for retrieved datasets in .db format (SQLite databases)
│
├── labels/                         # Dictionary for forged labels of datasets in .pkl format (pickle files)
│
├── methods/                        # Dictionary for implementation of different near-duplicate detection methods
│    ├── latent_semantic.py         # Implementation of Latent Semantic Analysis
│    ├── minhash_lsh.py             # Implementation of MinHash with Locality Sensitive Hashing
│    └── simhash.py                 # Implementation of SimHash
│
├── results/                        # Dictionary for generated results of the detection methods in .pkl format (pickle files)
│
├── utils/                          # Dictionary for utils Python files
│    └── preprocess.py              # Multiple functions for dataset preprocessing
│
├── detect_duplicate.py             # Script for detecting duplicates using selected methods on selected datasets
├── evaluate_result.py              # Script for evaluating result performance of selected methods on selected datasets
├── forge_label.py                  # Script for forging pseudo-labels for datasets
├── README.md                       # You're looking at it
├── requirements.txt                # List of all required Python packages for the project
└── retrieve_dataset.py             # Script for retrieving datasets from Common Crawl sources
```

## Requirements

Several Python packages are required to run our Python scripts, including `beautifulsoup4`, `mmh3`, `networkx`, `nltk`, `numpy`, `requests`, `scikit-learn`, `stopwordsiso`, `tqdm`, and `warcio`. Directly install them using the following commands:

```bash
pip install -r requirements.txt
```

There is also an additional Setup for `nltk`. After installing `nltk`, you need to download the tokenizer data by running:

```python
import nltk
nltk.download('punkt')
```

Before running the Python scripts, please make sure that:
* The `datasets/`, `labels/`, and `results/` directories are created within the root `NearDuplicateDetection/` directory.
* The required Python packages are installed.
* The current working directory is the root `NearDuplicateDetection/` directory.

## Instructions

1. **Retrieve a dataset from Common Crawl**:
    ```bash
    # Help for retrieve_dataset.py
    $ python3 retrieve_dataset.py -h
    usage: retrieve_dataset.py [-h] --target_data TARGET_DATA --target_url TARGET_URL [--index_name INDEX_NAME]

    Retrieve dataset from Common Crawl

    optional arguments:
    -h, --help            show this help message and exit
    --target_data TARGET_DATA
                            Target dataset name (e.g., 'foxnews')
    --target_url TARGET_URL
                            Target URL to fetch data (e.g., 'foxnews.com')
    --index_name INDEX_NAME
                            Index name to fetch data (e.g., 'YYYY-WW'), default is 2019-51

    # Example: A SQLite database file will be generated in NearDuplicateDetection/datasets/wikipedia.db
    $ python3 retrieve_dataset.py --target_data wikipedia --target_url en.wikipedia.org
    ```

2. **Forge pseudo-label for a dataset**:
    ```bash
    # Help for forge_label.py
    $ python3 forge_label.py -h
    usage: forge_label.py [-h] --target_data TARGET_DATA [--threshold THRESHOLD]

    Forge pseudo labels for dataset

    optional arguments:
    -h, --help            show this help message and exit
    --target_data TARGET_DATA
                            Target dataset name (e.g., 'foxnews')
    --threshold THRESHOLD
                            Similarity threshold, default is 0.8

    # Example: A pseudo-label pickle file will be generated in NearDuplicateDetection/labels/wikipedia_label.pkl
    $ python3 forge_label.py --target_data wikipedia
    ```

3. **Detect near-duplicate using a selected method on a selected dataset**:
    ```bash
    # Help for detect_duplicate.py
    $ python3 detect_duplicate.py -h
    usage: detect_duplicate.py [-h] --target_data TARGET_DATA --detection_method {simhash,minhash_lsh,latent_semantic} [--weight] [--num_bits NUM_BITS] [--hamming_threshold HAMMING_THRESHOLD] [--num_hashes NUM_HASHES] [--num_bands NUM_BANDS] [--jaccard_threshold JACCARD_THRESHOLD] [--n_components N_COMPONENTS]
                            [--cosine_threshold COSINE_THRESHOLD]

    Detect near-duplicate web pages using various methods.

    optional arguments:
    -h, --help            show this help message and exit
    --target_data TARGET_DATA
                            The target dataset.
    --detection_method {simhash,minhash_lsh,latent_semantic}
                            The method to use for near-duplicate detection (simhash, minhash_lsh, or latent_semantic).
    --weight              Whether to weight the words by schema, default is False.
    --num_bits NUM_BITS   Number of bits for SimHash (required if method is 'simhash'), default is 64.
    --hamming_threshold HAMMING_THRESHOLD
                            Threshold for Hamming distance (required if method is 'simhash'), default is 3.
    --num_hashes NUM_HASHES
                            Number of hash functions (required if method is 'minhash_lsh'), default is 128.
    --num_bands NUM_BANDS
                            Number of bands (required if method is 'minhash_lsh'), default is 16.
    --jaccard_threshold JACCARD_THRESHOLD
                            Threshold for Jaccard similarity (required if method is 'minhash_lsh'), default is 0.6.
    --n_components N_COMPONENTS
                            Number of components for Latent Semantic Analysis (required if method is 'latent_semantic'), default is 5000.
    --cosine_threshold COSINE_THRESHOLD
                            Threshold for cosine similarity (required if method is 'latent_semantic'), default is 0.8.

    # Example: A near-duplicate result pickle file will be generated in NearDuplicateDetection/results/wikipedia_minhash_lsh_result.pkl
    $ python3 detect_duplicate.py --target_data wikipedia --detection_method minhash_lsh
    ```

4. **Evaluate a result's performance of precision and recall**:
    ```bash
    # Help for evaluate_result.py
    $ python3 evaluate_result.py -h
    usage: evaluate_result.py [-h] --target_data TARGET_DATA --detection_method {simhash,minhash_lsh,latent_semantic}

    Evaluate near-duplicate detection result

    optional arguments:
    -h, --help            show this help message and exit
    --target_data TARGET_DATA
                            Target dataset name (e.g., 'foxnews')
    --detection_method {simhash,minhash_lsh,latent_semantic}
                            The method to use for near-duplicate detection (simhash, minhash_lsh, or latent_semantic).

    # Example: Precision and recall for evaluated near-duplicate detection result will be print out in the command line
    $ python3 evaluate_result.py --target_data wikipedia --detection_method minhash_lsh
    ```

It should be noted that for a specific dataset and a specific detection method, all of the four Python scripts must be run in the order given above to ensure proper functionality.
