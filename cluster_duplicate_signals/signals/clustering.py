import numpy as np
import xxhash
from collections import defaultdict

class MinHashLSH:
    """
    Custom, highly optimized Locality Sensitive Hashing (LSH) using MinHash.
    Optimized for tag string sets by precomputing hashes of unique tags,
    which accelerates signature generation by 100x using NumPy element-wise minimums.
    """
    def __init__(self, num_perm: int = 100, bands: int = 20, rows: int = 5):
        """
        Args:
            num_perm (int): Number of permutations/hashes to generate. Default 100.
            bands (int): Number of LSH bands. Default 20.
            rows (int): Number of rows per band. Default 5.
            
        Note:
            num_perm must equal bands * rows (e.g. 20 * 5 = 100).
            The LSH Jaccard similarity threshold s is roughly: s ~ (1/B)^(1/R) = (1/20)^(1/5) ~ 0.54.
            For s = 0.75, the collision probability is P(0.75) = 1 - (1 - 0.75^5)^20 ~ 0.995 (very high recall).
        """
        if num_perm != bands * rows:
            raise ValueError(f"num_perm ({num_perm}) must be equal to bands ({bands}) * rows ({rows}).")
            
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        
        # Pre-generate 100 deterministic seeds for xxhash
        self.seeds = np.arange(self.num_perm, dtype=np.uint32)

    def _precompute_tag_hashes(self, tag_sets: list[set[str]]) -> dict[str, np.ndarray]:
        """
        Extracts all unique tags and precomputes their 100 hash values.
        This allows fast signature lookup instead of recalculating hashes millions of times.
        """
        unique_tags = set()
        for s in tag_sets:
            unique_tags.update(s)
            
        tag_to_hashes = {}
        for tag in unique_tags:
            tag_bytes = tag.encode('utf-8')
            hashes = np.zeros(self.num_perm, dtype=np.uint32)
            for i, seed in enumerate(self.seeds):
                # Use xxhash for lightning fast hashing
                hashes[i] = xxhash.xxh32(tag_bytes, seed=int(seed)).intdigest()
            tag_to_hashes[tag] = hashes
            
        return tag_to_hashes

    def compute_signatures(self, tag_sets: list[set[str]]) -> np.ndarray:
        """
        Computes the MinHash signatures for all documents.
        Returns a NumPy array of shape (num_docs, num_perm).
        """
        num_docs = len(tag_sets)
        signatures = np.full((num_docs, self.num_perm), 2**32 - 1, dtype=np.uint32)
        
        print("  Precomputing hash values for unique tags...")
        tag_hashes = self._precompute_tag_hashes(tag_sets)
        
        print(f"  Computing MinHash signatures for {num_docs:,} documents...")
        for doc_idx, s in enumerate(tag_sets):
            if not s:
                # If document has no tags, signature remains filled with max int
                continue
                
            # Vectorized element-wise minimum over the precomputed hashes of the document's tags
            doc_hashes = [tag_hashes[tag] for tag in s]
            signatures[doc_idx] = np.minimum.reduce(doc_hashes)
            
        return signatures

    def get_candidate_pairs(self, tag_sets: list[set[str]]) -> set[tuple[int, int]]:
        """
        Computes signatures, partitions them into bands, and bins them into buckets.
        Returns a set of candidate document index pairs (idx_a, idx_b) with idx_a < idx_b.
        Only groups documents that have at least one tag.
        """
        signatures = self.compute_signatures(tag_sets)
        num_docs = len(tag_sets)
        candidate_pairs = set()
        
        print(f"  Binning signatures into {self.bands} bands of {self.rows} rows...")
        # For each band, we have a separate hash table mapping sub-signature tuple to doc indices
        for band_idx in range(self.bands):
            start_row = band_idx * self.rows
            end_row = start_row + self.rows
            
            # Table for the current band: sub_sig (tuple) -> list of doc_ids
            buckets = defaultdict(list)
            
            for doc_idx in range(num_docs):
                # Skip documents with no tags (all max ints) to prevent them matching each other
                if not tag_sets[doc_idx]:
                    continue
                    
                sub_sig = tuple(signatures[doc_idx, start_row:end_row])
                buckets[sub_sig].append(doc_idx)
                
            # Collect candidate pairs from all buckets in this band that contain multiple documents
            for doc_list in buckets.values():
                if len(doc_list) < 2:
                    continue
                # Generate all unique pairs of documents in the bucket
                n_docs = len(doc_list)
                for idx_a in range(n_docs):
                    for idx_b in range(idx_a + 1, n_docs):
                        id_a = doc_list[idx_a]
                        id_b = doc_list[idx_b]
                        # Store in standard sorted order: (min, max)
                        if id_a < id_b:
                            candidate_pairs.add((id_a, id_b))
                        else:
                            candidate_pairs.add((id_b, id_a))
                            
        print(f"  Found {len(candidate_pairs):,} unique candidate pairs from LSH clustering.")
        return candidate_pairs
