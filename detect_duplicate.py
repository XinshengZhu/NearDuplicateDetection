import argparse
from itertools import combinations
from math import comb
import pickle
import time
from tqdm import tqdm

from utils.preprocess import load_dataset_from_db, extract_text_from_html, tokenize_words_from_text, generate_bag_of_words

def parse_arguments():
    """
    Parse and validate command-line arguments for near-duplicate detection in web pages.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Detect near-duplicate web pages using various methods.")

    # General arguments
    parser.add_argument("--target_data", required=True, help="The target dataset.")
    parser.add_argument(
        "--detection_method", 
        required=True, 
        choices=["simhash", "minhash_lsh", "latent_semantic"],
        help="The method to use for near-duplicate detection (simhash, minhash_lsh, or latent_semantic)."
    )
    parser.add_argument(
        "--weight",
        action="store_true",
        default=False,
        help="Whether to weight the words by schema, default is False."
    )

    # SimHash-specific arguments
    parser.add_argument(
        "--num_bits", 
        type=int, 
        default=64, 
        help="Number of bits for SimHash (required if method is 'simhash'), default is 64."
    )
    parser.add_argument(
        "--hamming_threshold", 
        type=int, 
        default=3, 
        help="Threshold for Hamming distance (required if method is 'simhash'), default is 3."
    )

    # MinHash with Locality Sensitive Hashing-specific arguments
    parser.add_argument(
        "--num_hashes", 
        type=int, 
        default=128, 
        help="Number of hash functions (required if method is 'minhash_lsh'), default is 128."
    )
    parser.add_argument(
        "--num_bands", 
        type=int, 
        default=16, 
        help="Number of bands (required if method is 'minhash_lsh'), default is 16."
    )
    parser.add_argument(
        "--jaccard_threshold", 
        type=float, 
        default=0.6, 
        help="Threshold for Jaccard similarity (required if method is 'minhash_lsh'), default is 0.6."
    )

    # Latent Semantic Analysis-specific arguments
    parser.add_argument(
        "--n_components", 
        type=int, 
        default=5000, 
        help="Number of components for Latent Semantic Analysis (required if method is 'latent_semantic'), default is 5000."
    )
    parser.add_argument(
        "--cosine_threshold", 
        type=float, 
        default=0.8, 
        help="Threshold for cosine similarity (required if method is 'latent_semantic'), default is 0.8."
    )

    args = parser.parse_args()

    # Validate arguments based on the selected method
    if args.detection_method == "simhash":
        if args.num_bits <= 0:
            parser.error("--num_bits must be a positive integer when using SimHash.")
        if args.hamming_threshold < 0:
            parser.error("--hamming_threshold must be non-negative when using SimHash.")

    elif args.detection_method == "minhash_lsh":
        if args.num_hashes <= 0:
            parser.error("--num_hashes must be a positive integer when using MinHash with Locality Sensitive Hashing.")
        if args.num_bands <= 0:
            parser.error("--num_bands must be a positive integer when using MinHash with Locality Sensitive Hashing.")
        if not (0 <= args.jaccard_threshold <= 1):
            parser.error("--jaccard_threshold must be between 0 and 1 when using MinHash with Locality Sensitive Hashing.")

    elif args.detection_method == "latent_semantic":
        if args.n_components <= 0:
            parser.error("--n_components must be a positive integer when using Latent Semantic Analysis.")
        if not (0 <= args.cosine_threshold <= 1):
            parser.error("--cosine_threshold must be between 0 and 1 when using Latent Semantic Analysis.")

    return args

if __name__ == '__main__':
    # Parse the arguments
    args = parse_arguments()
    print()

    # Load the dataset from the database based on the target data
    url_html_dict = load_dataset_from_db(args.target_data)
    print(f"Loaded {len(url_html_dict)} HTML pages from datasets/{args.target_data}.db.")

    # Generate the bag of words for each HTML page
    generate_start_time = time.time()
    bags = {}
    for url, html in tqdm(url_html_dict.items(), desc=f"Generating bags of words"):
        text = extract_text_from_html(html)
        words = tokenize_words_from_text(text, args.weight)
        bag = generate_bag_of_words(words)
        if bag:
            bags[url] = bag
    generate_end_time = time.time()
    print(f"Generated {len(bags)} bags of words in {generate_end_time - generate_start_time:.2f} seconds.")
    print()

    # Start the near duplicate detection based on the selected method
    print(f"Starting near-duplicate detection:")
    if args.detection_method == "simhash":
        print(f"Using SimHash with {args.num_bits} bits and a Hamming distance threshold of {args.hamming_threshold}.")
        print()
        from utils.preprocess import compute_idf, compute_tfidf
        from methods.simhash import SimHash, hamming_distance

        # Normalize frequency of terms in each bag of words using TF-IDF
        normalize_start_time = time.time()
        normalized_bags = {}
        idf = compute_idf(bags)
        for url, bag in bags.items():
            tfidf = compute_tfidf(bag, idf)
            normalized_bags[url] = tfidf
        normalize_end_time = time.time()
        print(f"Normalized (TF-IDF) {len(normalized_bags)} bags of words in {normalize_end_time - normalize_start_time:.2f} seconds.")
        print()

        # Compute the SimHash fingerprints for each normalized bag of words
        simhash_start_time = time.time()
        simhash_detector = SimHash(args.num_bits)
        sh_fingerprints = {}
        for url, normalized_bag in tqdm(normalized_bags.items(), desc="Computing SimHash fingerprints"):
            sh_fingerprints[url] = simhash_detector.compute_fingerprint(normalized_bag)
        simhash_end_time = time.time()
        print(f"Computed {len(sh_fingerprints)} SimHash fingerprints in {simhash_end_time - simhash_start_time:.2f} seconds.")
        print()

        # Detect near duplicates based on the Hamming distance
        detect_start_time = time.time()
        result_pairs = set()
        for url1, url2 in tqdm(combinations(sh_fingerprints.keys(), 2), desc="Detecting near duplicates", total=comb(len(sh_fingerprints), 2)):
            distance = hamming_distance(sh_fingerprints[url1], sh_fingerprints[url2])
            if distance <= args.hamming_threshold:
                pair = tuple(sorted((url1, url2)))
                result_pairs.add((pair, distance))
        detect_end_time = time.time()
        print(f"Detected {len(result_pairs)} near-duplicate pairs in {detect_end_time - detect_start_time:.2f} seconds.")
        print()

        # Save the result pairs to a pkl file
        result_pairs = sorted(result_pairs, key=lambda x: x[1])
        with open(f"results/{args.target_data}_{args.detection_method}_result.pkl", "wb") as f:
            pickle.dump(result_pairs, f)
        print(f"Saved result pairs to results/{args.target_data}_{args.detection_method}_result.pkl.")
        print()

    elif args.detection_method == "minhash_lsh":
        print(f"Using MinHash with Locality Sensitive Hashing with {args.num_hashes} hash functions and {args.num_bands} LSH bands and a Jaccard similarity threshold of {args.jaccard_threshold}.")
        print()
        from utils.preprocess import flatten_bag_of_words
        from methods.minhash_lsh import MinHash, LocalitySensitiveHashing, jaccard_similarity
        
        # Flatten the bags of words
        flatten_start_time = time.time()
        flattened_bags = {}
        for url, bag in bags.items():
            flattened_bags[url] = flatten_bag_of_words(bag)
        flatten_end_time = time.time()
        print(f"Flattened {len(flattened_bags)} bags of words in {flatten_end_time - flatten_start_time:.2f} seconds.")
        print()

        # Compute the MinHash signatures for each flattened bag of words
        minhash_start_time = time.time()
        minhash_detector = MinHash(args.num_hashes)
        mh_signatures = {}
        for url, flattened_bag in tqdm(flattened_bags.items(), desc="Computing MinHash signatures"):
            mh_signatures[url] = minhash_detector.compute_signature(flattened_bag)
        minhash_end_time = time.time()
        print(f"Computed {len(mh_signatures)} MinHash signatures in {minhash_end_time - minhash_start_time:.2f} seconds.")
        print()

        # Insert the MinHash signatures into the Locality Sensitive Hashing index
        lsh_start_time = time.time()
        lsh_index = LocalitySensitiveHashing(args.num_bands, args.num_hashes // args.num_bands)
        for url, signature in mh_signatures.items():
            lsh_index.insert(url, signature)
        lsh_end_time = time.time()
        print(f"Inserted {len(mh_signatures)} MinHash signatures into the LSH index in {lsh_end_time - lsh_start_time:.2f} seconds.")
        print()

        # Detect near duplicates using the LSH index based on the Jaccard similarity
        detect_start_time = time.time()
        result_pairs = set()
        for url, signature in tqdm(mh_signatures.items(), desc="Detecting near duplicates"):
            candidates = lsh_index.query(signature)
            for candidate_url in candidates:
                similarity = jaccard_similarity(signature, mh_signatures[candidate_url])
                if candidate_url != url and similarity >= args.jaccard_threshold:
                    pair = tuple(sorted((url, candidate_url)))
                    result_pairs.add((pair, similarity))
        detect_end_time = time.time()
        print(f"Detected {len(result_pairs)} near-duplicate pairs in {detect_end_time - detect_start_time:.2f} seconds.")
        print()

        # Save the result pairs to a pkl file
        result_pairs = sorted(result_pairs, key=lambda x: x[1], reverse=True)
        with open(f"results/{args.target_data}_{args.detection_method}_result.pkl", "wb") as f:
            pickle.dump(result_pairs, f)
        print(f"Saved result pairs to results/{args.target_data}_{args.detection_method}_result.pkl.")
        print()

    elif args.detection_method == "latent_semantic":
        print(f"Using Latent Semantic Analysis with {args.n_components} components and a cosine similarity threshold of {args.cosine_threshold}.")
        print()
        from utils.preprocess import compute_tfidf_matrix
        from methods.latent_semantic import LatentSemanticAnalysis, cosine_similarity

        # Compute the TF-IDF matrix
        tfidf_start_time = time.time()
        tfidf_matrix = compute_tfidf_matrix(bags)
        tfidf_end_time = time.time()
        print(f"Computed {tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]} TF-IDF matrix in {tfidf_end_time - tfidf_start_time:.2f} seconds.")
        print()

        # Perform Latent Semantic Analysis
        lsa_start_time = time.time()
        lsa = LatentSemanticAnalysis(args.n_components)
        lsa.fit(tfidf_matrix)
        lsa_matrix = lsa.get_lsa_matrix()
        lsa_end_time = time.time()
        print(f"Performed Latent Semantic Analysis on the TF-IDF matrix in {lsa_end_time - lsa_start_time:.2f} seconds.")
        print()

        # Detect near duplicates based on cosine similarity
        detect_start_time = time.time()
        result_pairs = set()
        url_list = list(bags.keys())
        for i, j in tqdm(combinations(range(len(url_list)), 2), desc="Detecting near duplicates", total=comb(len(url_list), 2)):
            similarity = cosine_similarity(lsa_matrix[i], lsa_matrix[j])
            if similarity >= args.cosine_threshold:
                pair = tuple(sorted((url_list[i], url_list[j])))
                result_pairs.add((pair, similarity))
        detect_end_time = time.time()
        print(f"Detected {len(result_pairs)} near-duplicate pairs in {detect_end_time - detect_start_time:.2f} seconds.")
        print()

        # Save the result pairs to a pkl file
        result_pairs = sorted(result_pairs, key=lambda x: x[1], reverse=True)
        with open(f"results/{args.target_data}_{args.detection_method}_result.pkl", "wb") as f:
            pickle.dump(result_pairs, f)
        print(f"Saved result pairs to results/{args.target_data}_{args.detection_method}_result.pkl.")
        print()
        