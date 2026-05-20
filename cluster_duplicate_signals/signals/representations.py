import pandas as pd
import numpy as np

def generate_document_tag_sets(df: pd.DataFrame) -> list[set[str]]:
    """
    For each document (row), generates a string set of its tags from:
    - content_length (prefixed 'len:')
    - content_type (prefixed 'type:', multi-select)
    - business_sector (prefixed 'sector:', multi-select)
    
    Example output set: {"len:medium", "type:news_report", "type:blog_post", "sector:tech"}
    
    Args:
        df (pd.DataFrame): The pooled sample DataFrame.
        
    Returns:
        list[set[str]]: A list of string sets, one for each document.
    """
    tag_sets = []
    
    # Iterate through the DataFrame rows to build sets
    for idx, row in df.iterrows():
        doc_tags = set()
        
        # 1. Length Tag
        len_val = row.get("content_length")
        if pd.notna(len_val) and len_val is not None:
            len_str = str(len_val).strip()
            if len_str != "":
                doc_tags.add(f"len:{len_str}")
                
        # 2. Content Type Tags (Multi-select)
        type_val = row.get("content_type")
        if type_val is not None:
            if isinstance(type_val, (list, np.ndarray, set)):
                for t in type_val:
                    if t is not None and str(t).strip() != "":
                        doc_tags.add(f"type:{str(t).strip()}")
            elif pd.notna(type_val):
                type_str = str(type_val).strip()
                if type_str != "":
                    doc_tags.add(f"type:{type_str}")
                    
        # 3. Business Sector Tags (Multi-select)
        sector_val = row.get("business_sector")
        if sector_val is not None:
            if isinstance(sector_val, (list, np.ndarray, set)):
                for s in sector_val:
                    if s is not None and str(s).strip() != "":
                        doc_tags.add(f"sector:{str(s).strip()}")
            elif pd.notna(sector_val):
                sector_str = str(sector_val).strip()
                if sector_str != "":
                    doc_tags.add(f"sector:{sector_str}")
                    
        tag_sets.append(doc_tags)
        
    return tag_sets

def calculate_bucket_density(tag_sets: list[set[str]]) -> float:
    """
    Calculates bucket_density as the number of unique string sets divided by the total number of documents (n).
    Uses frozenset to make the sets hashable so they can be easily deduplicated.
    
    Formula:
        bucket_density = num_unique_string_sets / n
        
    Args:
        tag_sets (list[set[str]]): The list of tag sets generated for each document.
        
    Returns:
        float: The bucket density value.
    """
    n = len(tag_sets)
    if n == 0:
        return 0.0
        
    # Convert sets to frozensets so they are hashable, then find unique ones
    unique_sets = set(frozenset(s) for s in tag_sets)
    
    bucket_density = len(unique_sets) / n
    print(f"Calculated Bucket Density: {len(unique_sets):,} unique tag sets / {n:,} documents = {bucket_density:.6f}")
    return bucket_density
