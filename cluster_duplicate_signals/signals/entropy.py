import math
from collections import Counter
import pandas as pd
import numpy as np

def calculate_frequencies_and_entropy(
    df: pd.DataFrame,
    column: str
) -> tuple[dict[str, int], float]:
    """
    Calculates the frequency count of each tag in a column and computes the Shannon entropy.
    Handles both single-valued fields (strings) and multi-select fields (lists or arrays of strings).
    
    Shannon Entropy formula:
        H(X) = -sum( P(x_i) * log_2(P(x_i)) )
    where P(x_i) = frequency of tag i / total count of all tags in this column.
    
    Args:
        df (pd.DataFrame): The pooled sample DataFrame.
        column (str): The column name to compute (e.g., 'content_length', 'content_type', 'business_sector').
        
    Returns:
        tuple: (frequencies_dict, shannon_entropy)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
    # Flatten the values in the column to collect all individual tags
    all_tags = []
    
    for val in df[column]:
        if val is None:
            continue
        if isinstance(val, (list, np.ndarray, set)):
            pass
        elif pd.isna(val):
            continue
            
        # Check if the value is list-like (multi-select property)
        if isinstance(val, (list, np.ndarray, set)):
            for tag in val:
                if tag is not None and str(tag).strip() != "":
                    all_tags.append(str(tag).strip())
        else:
            # Single string or other categorical value
            tag_str = str(val).strip()
            if tag_str != "":
                all_tags.append(tag_str)
                
    total_tags = len(all_tags)
    if total_tags == 0:
        return {}, 0.0
        
    # Calculate frequencies using Counter
    frequencies = dict(Counter(all_tags))
    
    # Calculate Shannon Entropy
    entropy = 0.0
    for tag_count in frequencies.values():
        p_x = tag_count / total_tags
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
            
    return frequencies, entropy
