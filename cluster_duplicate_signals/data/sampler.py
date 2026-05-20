import pandas as pd
import numpy as np

def systematic_block_sample(
    df: pd.DataFrame,
    block_size: int = 10000,
    num_blocks: int = 10
) -> pd.DataFrame:
    """
    Performs systematic block sampling (Option B):
    Splits the dataset of length N into `num_blocks` equal segments (deciles),
    and takes `block_size` consecutive rows starting from the beginning of each segment.
    
    Start indices: start_indices = [int(i * N / num_blocks) for i in range(num_blocks)]
    
    Then pools the sampled blocks into a single combined DataFrame of size (block_size * num_blocks).
    
    Args:
        df (pd.DataFrame): The source DataFrame of length N.
        block_size (int): The number of consecutive rows to sample in each block. Default 10,000.
        num_blocks (int): The number of blocks to sample. Default 10.
        
    Returns:
        pd.DataFrame: A pooled DataFrame containing the systematic blocks.
    """
    N = len(df)
    if N == 0:
        raise ValueError("Cannot perform systematic sampling on an empty dataset.")
        
    print(f"Dataset length (N) = {N:,}. Performing systematic block sampling:")
    print(f"Block size = {block_size:,}, Number of blocks = {num_blocks}")
    
    sampled_blocks = []
    for i in range(num_blocks):
        start_idx = int(i * N / num_blocks)
        end_idx = start_idx + block_size
        
        # Ensure we do not go out of bounds if dataset is small
        end_idx_capped = min(end_idx, N)
        block = df.iloc[start_idx:end_idx_capped]
        
        print(f"  Block {i+1}/{num_blocks}: rows {start_idx:,} to {end_idx_capped:,} (actual size: {len(block):,})")
        sampled_blocks.append(block)
        
    # Pool the blocks into a single combined DataFrame (Option A)
    pooled_df = pd.concat(sampled_blocks, ignore_index=True)
    print(f"Successfully pooled blocks into a single sample of {len(pooled_df):,} rows.")
    return pooled_df
