import argparse
import pickle

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate near-duplicate detection result")
    parser.add_argument("--target_data", help="Target dataset name (e.g., 'foxnews')", required=True)
    parser.add_argument(
        "--detection_method", 
        required=True, 
        choices=["simhash", "minhash_lsh", "latent_semantic"],
        help="The method to use for near-duplicate detection (simhash, minhash_lsh, or latent_semantic)."
    )
    args = parser.parse_args()
    print()

    # Load the pseudo label
    with open(f"labels/{args.target_data}_label.pkl", "rb") as f:
        pseudo_labels = pickle.load(f)
    print(f"Loaded {len(pseudo_labels)} pseudo labels from labels/{args.target_data}_label.pkl.")

    # Load the result
    with open(f"results/{args.target_data}_{args.detection_method}_result.pkl", "rb") as f:
        result_pairs = pickle.load(f)
    print(f"Loaded {len(result_pairs)} result pairs from results/{args.target_data}_{args.detection_method}_result.pkl.")
    print()

    # Evaluate the result
    count = 0
    for (pair, value) in result_pairs:
        if value >= 0.6 and pair in pseudo_labels:
            count += 1
    precision = count / len(result_pairs)
    recall = count / len(pseudo_labels)
    print(f"Precision of using {args.detection_method} for {args.target_data}: {precision:.2f}")
    print(f"Recall of using {args.detection_method} for {args.target_data}: {recall:.2f}")
    print()
