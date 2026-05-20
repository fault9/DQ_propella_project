import unittest
import numpy as np
import pandas as pd
import math

from cluster_duplicate_signals.data.sampler import systematic_block_sample
from cluster_duplicate_signals.signals.entropy import calculate_frequencies_and_entropy
from cluster_duplicate_signals.signals.representations import (
    generate_document_tag_sets,
    calculate_bucket_density
)
from cluster_duplicate_signals.signals.heterogeneity import calculate_semantic_heterogeneity

class TestDatasetQualityPipeline(unittest.TestCase):
    
    def test_systematic_block_sampling(self):
        """
        Verifies decile systematic block sampling starts (Option B).
        With N=100, block_size=5, num_blocks=10, the start indices should be:
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90.
        """
        # Create dummy df of size 100
        df = pd.DataFrame({"id": [f"doc_{i}" for i in range(100)]})
        
        pooled = systematic_block_sample(df, block_size=5, num_blocks=10)
        
        # Total pooled size should be 5 * 10 = 50 rows
        self.assertEqual(len(pooled), 50)
        
        # Verify first row of each block:
        # Block 1 starts at 0 -> doc_0
        self.assertEqual(pooled.iloc[0]["id"], "doc_0")
        # Block 2 starts at 10 -> doc_10 (which is index 5 in pooled)
        self.assertEqual(pooled.iloc[5]["id"], "doc_10")
        # Block 10 starts at 90 -> doc_90 (which is index 45 in pooled)
        self.assertEqual(pooled.iloc[45]["id"], "doc_90")

    def test_shannon_entropy(self):
        """
        Verifies Shannon entropy calculation H(X) = -sum(P(x_i)*log_2(P(x_i)))
        Let's construct a column with 4 items: ['brief', 'brief', 'medium', 'long']
        Total tags = 4.
        Frequencies: brief: 2 (P=0.5), medium: 1 (P=0.25), long: 1 (P=0.25)
        Expected Entropy: - (0.5 * log2(0.5) + 0.25 * log2(0.25) + 0.25 * log2(0.25))
                          = - (0.5 * (-1) + 0.25 * (-2) + 0.25 * (-2))
                          = 0.5 + 0.5 + 0.5 = 1.5
        """
        df = pd.DataFrame({"content_length": ["brief", "brief", "medium", "long"]})
        freqs, entropy = calculate_frequencies_and_entropy(df, "content_length")
        
        self.assertEqual(freqs["brief"], 2)
        self.assertEqual(freqs["medium"], 1)
        self.assertEqual(freqs["long"], 1)
        self.assertAlmostEqual(entropy, 1.5, places=6)

    def test_shannon_entropy_multi_select(self):
        """
        Verifies multi-select entropy.
        df with 3 rows:
          row 0: ['a', 'b']
          row 1: ['a']
          row 2: ['c']
        Total tags = 4. Frequencies: a: 2 (P=0.5), b: 1 (P=0.25), c: 1 (P=0.25)
        Expected Entropy: 1.5
        """
        df = pd.DataFrame({"content_type": [["a", "b"], ["a"], ["c"]]})
        freqs, entropy = calculate_frequencies_and_entropy(df, "content_type")
        
        self.assertEqual(freqs["a"], 2)
        self.assertEqual(freqs["b"], 1)
        self.assertEqual(freqs["c"], 1)
        self.assertAlmostEqual(entropy, 1.5, places=6)

    def test_document_tag_sets_and_bucket_density(self):
        """
        Verifies unified tag string sets and bucket density calculation.
        """
        df = pd.DataFrame({
            "content_length": ["brief", "medium"],
            "content_type": [["news"], ["blog", "news"]],
            "business_sector": [["tech"], ["finance"]]
        })
        
        tag_sets = generate_document_tag_sets(df)
        
        # Verify tag structure
        self.assertEqual(tag_sets[0], {"len:brief", "type:news", "sector:tech"})
        self.assertEqual(tag_sets[1], {"len:medium", "type:blog", "type:news", "sector:finance"})
        
        # Both sets are unique, so bucket density is 2 / 2 = 1.0
        density = calculate_bucket_density(tag_sets)
        self.assertEqual(density, 1.0)

    def test_semantic_heterogeneity(self):
        """
        Verifies candidate-conditioned cosine similarity and semantic heterogeneity calculation (Option B).
        We have 3 documents:
          doc 0 and doc 1 are duplicates (embedding cosine similarity > 0.9)
          doc 2 is distinct (embedding cosine similarity < 0.9)
        Expected Semantic Heterogeneity: (3 - 2) / 3 = 0.333333
        """
        # Embeddings: size 3 x 2
        # doc 0: [1, 0]
        # doc 1: [0.95, 0.312] -> Cosine sim ~ 0.95
        # doc 2: [0.0, 1.0]    -> Cosine sim = 0.0
        embeddings = np.array([
            [1.0, 0.0],
            [0.95, 0.3122498999199199], # sqrt(1 - 0.95^2) = 0.31225...
            [0.0, 1.0]
        ])
        
        heterogeneity, duplicate_list, num_pairs = calculate_semantic_heterogeneity(
            embeddings=embeddings,
            similarity_threshold=0.9
        )
        
        self.assertEqual(num_pairs, 1)
        self.assertEqual(len(duplicate_list), 1)
        self.assertEqual(duplicate_list[0][0], 0)
        self.assertEqual(duplicate_list[0][1], 1)
        self.assertAlmostEqual(duplicate_list[0][2], 0.95, places=4)
        self.assertAlmostEqual(heterogeneity, 1.0 / 3.0, places=6)

if __name__ == "__main__":
    unittest.main()
