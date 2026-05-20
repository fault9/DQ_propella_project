import torch
import numpy as np
import pandas as pd
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

def generate_description_embeddings(
    descriptions: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    backend: str = "sentence-transformers",
    batch_size: int = 256
) -> np.ndarray:
    """
    Generates dense text embeddings from the one_sentence_description field of each document.
    Supports 'sentence-transformers' (highly optimized via PyTorch on CPU) and 'fastembed' (ONNX Runtime).
    
    Args:
        descriptions (list[str]): List of descriptions to embed.
        model_name (str): Embedding model name. Default 'all-MiniLM-L6-v2'.
        backend (str): Engine to use ('sentence-transformers' or 'fastembed'). Default 'sentence-transformers'.
        batch_size (int): Batch size for embedding generation. Default 256.
        
    Returns:
        np.ndarray: A 2D numpy array of shape (num_documents, embedding_dim).
    """
    import os
    
    # Clean and preprocess descriptions, filling NaN/None with empty string
    cleaned_descriptions = [
        str(desc).strip() if (pd.notna(desc) and desc is not None) else ""
        for desc in descriptions
    ]
    
    # Use fastembed if selected and installed
    if backend == "fastembed" and HAS_FASTEMBED:
        fastembed_model_name = model_name
        if model_name == "all-MiniLM-L6-v2":
            fastembed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        elif model_name == "paraphrase-multilingual-MiniLM-L12-v2":
            fastembed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
        # Optimize threads: ONNX Runtime default is logical cores, but on CPU physical cores is faster
        cpu_count = os.cpu_count()
        optimal_threads = max(1, cpu_count // 2) if cpu_count else None
        
        print(f"Using fastembed for CPU inference. Loading model '{fastembed_model_name}' (threads={optimal_threads})...")
        try:
            model = TextEmbedding(model_name=fastembed_model_name, threads=optimal_threads)
            # TextEmbedding.embed returns a generator of numpy arrays, we convert it to a 2D numpy array
            embeddings_gen = model.embed(cleaned_descriptions, batch_size=batch_size)
            embeddings = np.vstack(list(embeddings_gen)).astype(np.float32)
            print(f"Successfully generated embeddings matrix of shape {embeddings.shape} using fastembed.")
            return embeddings
        except Exception as e:
            print(f"Warning: fastembed failed to load/generate: {e}. Falling back to SentenceTransformer...")
            
    # Otherwise, use SentenceTransformer (highly optimized on CPU via PyTorch BLAS/OpenMP)
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer model '{model_name}' on device: {device}...")
    
    model = SentenceTransformer(model_name, device=device)
    print(f"Generating dense embeddings for {len(cleaned_descriptions):,} descriptions using SentenceTransformer...")
    
    embeddings = model.encode(
        cleaned_descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Successfully generated embeddings matrix of shape {embeddings.shape} using SentenceTransformer.")
    return embeddings
