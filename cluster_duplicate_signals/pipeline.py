import time
import pandas as pd
import numpy as np
from cluster_duplicate_signals.data.loader import load_annotations
from cluster_duplicate_signals.data.sampler import systematic_block_sample
from cluster_duplicate_signals.signals.entropy import calculate_frequencies_and_entropy
from cluster_duplicate_signals.signals.representations import (
    generate_document_tag_sets,
    calculate_bucket_density
)
from cluster_duplicate_signals.signals.embeddings import generate_description_embeddings
from cluster_duplicate_signals.signals.heterogeneity import calculate_semantic_heterogeneity

def run_quality_pipeline(
    repo_id: str = "openeurollm/propella-annotations",
    filename: str = "data/propella-1-4b/finepdfs/swe_Latn/shard000000.parquet",
    block_size: int = 10000,
    num_blocks: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_backend: str = "sentence-transformers",
    batch_size: int = 256,
    similarity_threshold: float = 0.9
) -> dict:
    """
    Orchestrates the entire systematic dataset quality evaluation pipeline.
    Uses FAISS IndexFlatIP range search to detect semantic duplicates.
    
    Returns:
        dict: A dictionary containing all quality signals and intermediate analysis metrics.
    """
    pipeline_t0 = time.time()
    print("=" * 70)
    print("STARTING DATASET QUALITY QUANTIFICATION PIPELINE")
    print("=" * 70)
    
    # Step 1: Load selected columns from Parquet file (using local HF caching)
    t0 = time.time()
    columns = ["id", "content_length", "content_type", "business_sector", "one_sentence_description"]
    df = load_annotations(repo_id=repo_id, filename=filename, columns=columns)
    load_time = time.time() - t0
    print(f"-> Loader phase completed in {load_time:.2f} seconds.\n")
    
    # Step 2: Perform systematic block sampling (Option B) and pool them (Option A)
    t0 = time.time()
    pooled_df = systematic_block_sample(df, block_size=block_size, num_blocks=num_blocks)
    sampling_time = time.time() - t0
    print(f"-> Systematic block sampling completed in {sampling_time:.2f} seconds.\n")
    
    # Step 3: Calculate tag frequencies and Shannon entropies
    t0 = time.time()
    print("Calculating Shannon Entropies:")
    length_freqs, length_entropy = calculate_frequencies_and_entropy(pooled_df, "content_length")
    type_freqs, type_entropy = calculate_frequencies_and_entropy(pooled_df, "content_type")
    sector_freqs, sector_entropy = calculate_frequencies_and_entropy(pooled_df, "business_sector")
    entropy_time = time.time() - t0
    
    # Normalization (using maximum possible values based on property_descriptions.md)
    import math
    length_entropy_norm = length_entropy / math.log2(4) if length_entropy > 0 else 0.0
    type_entropy_norm = type_entropy / math.log2(18) if type_entropy > 0 else 0.0
    sector_entropy_norm = sector_entropy / math.log2(37) if sector_entropy > 0 else 0.0
    
    print(f"  Length Entropy (raw)      : {length_entropy:.6f} | Normalized: {length_entropy_norm:.6f}")
    print(f"  Content Type Entropy (raw): {type_entropy:.6f} | Normalized: {type_entropy_norm:.6f}")
    print(f"  Business Sector Ent. (raw): {sector_entropy:.6f} | Normalized: {sector_entropy_norm:.6f}")
    print(f"-> Entropy calculations completed in {entropy_time:.2f} seconds.\n")
    
    # Step 4: Generate tag string sets for each document
    t0 = time.time()
    print("Generating document tag representations...")
    tag_sets = generate_document_tag_sets(pooled_df)
    representations_time = time.time() - t0
    print(f"-> Tag representation generation completed in {representations_time:.2f} seconds.\n")
    
    # Step 5: Calculate bucket density
    t0 = time.time()
    print("Calculating bucket density...")
    bucket_density = calculate_bucket_density(tag_sets)
    density_time = time.time() - t0
    print(f"-> Bucket density calculation completed in {density_time:.2f} seconds.\n")
    
    # Step 6: Generate dense text embeddings from descriptions for ALL documents
    t0 = time.time()
    all_descriptions = [str(desc) for desc in pooled_df["one_sentence_description"].tolist()]
    
    # Generate embeddings for all descriptions
    embeddings = generate_description_embeddings(
        descriptions=all_descriptions,
        model_name=embedding_model,
        backend=embedding_backend,
        batch_size=batch_size
    )
    embedding_time = time.time() - t0
    print(f"-> Dense embedding generation completed in {embedding_time:.2f} seconds.\n")
    
    # Step 7: Calculate candidate cosine similarities and semantic heterogeneity using FAISS
    t0 = time.time()
    print("Calculating semantic heterogeneity metrics using FAISS...")
    semantic_heterogeneity, duplicate_pairs, num_duplicates = calculate_semantic_heterogeneity(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )
    heterogeneity_time = time.time() - t0
    print(f"-> Semantic heterogeneity quantification completed in {heterogeneity_time:.2f} seconds.\n")
    
    total_time = time.time() - pipeline_t0
    print("=" * 70)
    print("DATASET QUALITY QUANTIFICATION PIPELINE COMPLETED")
    print(f"Total Pipeline Execution Time: {total_time:.2f} seconds")
    print("=" * 70)
    
    # Consolidate results
    results = {
        # Core Output Quality Signals
        "length_entropy": length_entropy,
        "length_entropy_normalized": length_entropy_norm,
        "type_entropy": type_entropy,
        "type_entropy_normalized": type_entropy_norm,
        "sector_entropy": sector_entropy,
        "sector_entropy_normalized": sector_entropy_norm,
        "bucket_density": bucket_density,
        "semantic_heterogeneity": semantic_heterogeneity,
        
        # Supporting Debugging Information
        "total_documents": len(pooled_df),
        "num_semantic_duplicates": len(duplicate_pairs),
        "duplicate_pairs": duplicate_pairs,
        
        # Categorical frequencies
        "frequencies": {
            "content_length": length_freqs,
            "content_type": type_freqs,
            "business_sector": sector_freqs
        },
        
        # Execution timings (for scaling assessments)
        "timings": {
            "loader": load_time,
            "sampler": sampling_time,
            "entropy": entropy_time,
            "representations": representations_time,
            "bucket_density": density_time,
            "embeddings": embedding_time,
            "heterogeneity": heterogeneity_time,
            "total": total_time
        }
    }
    
    return results
