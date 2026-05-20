import argparse
import time
from cluster_duplicate_signals.pipeline import run_quality_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Quality Quantification Pipeline Prototype using FAISS"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="openeurollm/propella-annotations",
        help="Hugging Face repository ID (default: openeurollm/propella-annotations)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="data/propella-1-4b/finepdfs/swe_Latn/shard000000.parquet",
        help="Target Parquet filename in Hugging Face repository"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=10000,
        help="Size of each consecutive block for systematic sampling (default: 10,000)"
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=10,
        help="Number of blocks spaced across the dataset (default: 10)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Lightweight sentence transformer model for dense text embeddings (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--embedding_backend",
        type=str,
        choices=["sentence-transformers", "fastembed"],
        default="sentence-transformers",
        help="Embedding engine backend to use (default: sentence-transformers)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generating embeddings (default: 256)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for semantic duplicates (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    # Run the quality pipeline
    results = run_quality_pipeline(
        repo_id=args.repo_id,
        filename=args.filename,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        embedding_model=args.embedding_model,
        embedding_backend=args.embedding_backend,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold
    )
    
    # Print the core quality signal report
    print("\n" + "=" * 70)
    print("                CORE DATASET QUALITY SIGNALS REPORT")
    print("=" * 70)
    print(f"  Length Shannon Entropy     : {results['length_entropy']:.6f} (Normalized: {results['length_entropy_normalized']:.6f})")
    print(f"  Content Type Shannon Ent.  : {results['type_entropy']:.6f} (Normalized: {results['type_entropy_normalized']:.6f})")
    print(f"  Business Sector Shannon Ent: {results['sector_entropy']:.6f} (Normalized: {results['sector_entropy_normalized']:.6f})")
    print(f"  Tag Bucket Density         : {results['bucket_density']:.6f}")
    print(f"  Semantic Heterogeneity     : {results['semantic_heterogeneity']:.6f}")
    print("-" * 70)
    print("  METADATA STATISTICS:")
    print(f"    Sample Size (Pooled)      : {results['total_documents']:,} documents")
    print(f"    Semantic Duplicate Pairs  : {results['num_semantic_duplicates']:,} pairs (similarity > {args.similarity_threshold})")
    print("-" * 70)
    print("  PHASE TIMINGS:")
    for phase, seconds in results["timings"].items():
        print(f"    {phase.capitalize():<25} : {seconds:>7.2f} seconds")
    print("=" * 70 + "\n")
    
    # Optional print of a few duplicates if found
    if results["num_semantic_duplicates"] > 0:
        print(f"Sample of Detected Semantic Duplicates (showing up to 5):")
        for i, (idx_a, idx_b, sim) in enumerate(results["duplicate_pairs"][:5]):
            print(f"  [{i+1}] Doc #{idx_a} & Doc #{idx_b} -> Cosine Similarity: {sim:.4f}")
        print()

if __name__ == "__main__":
    main()
