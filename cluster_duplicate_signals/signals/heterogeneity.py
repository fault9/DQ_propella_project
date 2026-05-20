import numpy as np
import faiss

def calculate_semantic_heterogeneity(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.9
) -> tuple[float, list[tuple[int, int, float]], int]:
    """
    Calculates semantic heterogeneity using a FAISS IndexFlatIP range search.
    Finds all document pairs with cosine similarity > similarity_threshold.
    
    Formula:
        semantic_heterogeneity = (N - |U_duplicated|) / N
    where:
        N = total documents
        U_duplicated = unique document IDs involved in at least one duplicate pair
    """
    total_documents = embeddings.shape[0]
    if total_documents == 0:
        return 1.0, [], 0
        
    print(f"  Building FAISS IndexFlatIP and running range search (threshold: {similarity_threshold})...")
    
    # 1. Normalize embeddings to unit length (L2 norm)
    # Convert embeddings to float32 as required by FAISS
    embeddings_f32 = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_f32)
    
    # 2. Build index
    dimension = embeddings_f32.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_f32)
    
    # 3. Range search (finds all matches with inner product >= similarity_threshold)
    # Range search query takes (queries, threshold)
    lims, D, I = index.range_search(embeddings_f32, similarity_threshold)
    
    # 4. Extract unique pairs with neighbor_idx > query_idx to avoid self-matches and double-counting
    duplicate_pairs_list = []
    duplicated_docs = set()
    total_pairs_found = 0
    
    for i in range(total_documents):
        start = lims[i]
        end = lims[i+1]
        for j in range(start, end):
            neighbor_idx = int(I[j])
            similarity = float(D[j])
            if neighbor_idx > i:
                duplicate_pairs_list.append((i, neighbor_idx, similarity))
                duplicated_docs.add(i)
                duplicated_docs.add(neighbor_idx)
                total_pairs_found += 1
                
    num_duplicated = len(duplicated_docs)
    num_non_duplicated = total_documents - num_duplicated
    semantic_heterogeneity = num_non_duplicated / total_documents
    
    # Sort duplicate pairs by similarity descending for nicer reporting
    duplicate_pairs_list.sort(key=lambda x: x[2], reverse=True)
    
    print(f"  Found {total_pairs_found:,} semantic duplicate pairs (similarity > {similarity_threshold}) using FAISS.")
    print(f"  Duplicated documents count: {num_duplicated:,} / {total_documents:,}")
    print(f"  Semantic Heterogeneity (fraction of non-duplicated docs): {semantic_heterogeneity:.6f}")
    
    # Return: semantic_heterogeneity, duplicate_pairs_list, and the total pairs found
    return float(semantic_heterogeneity), duplicate_pairs_list, total_pairs_found
