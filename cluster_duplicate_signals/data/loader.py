import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

def load_annotations(
    repo_id: str = "openeurollm/propella-annotations",
    filename: str = "data/propella-1-4b/finepdfs/swe_Latn/shard000000.parquet",
    columns: list = None
) -> pd.DataFrame:
    """
    Downloads the specified Parquet shard from the Hugging Face Hub using hf_hub_download,
    which provides automatic local caching. Once cached, reads only the specified columns
    into a Pandas DataFrame to minimize memory usage.
    
    Args:
        repo_id (str): Hugging Face repository ID.
        filename (str): Path to the parquet file inside the repository.
        columns (list): List of column names to load. If None, loads target prototype columns.
        
    Returns:
        pd.DataFrame: Pandas DataFrame containing the loaded columns.
    """
    if columns is None:
        columns = [
            "id",
            "content_length",
            "content_type",
            "business_sector",
            "one_sentence_description"
        ]
        
    print(f"Retrieving '{filename}' from Hugging Face dataset '{repo_id}'...")
    
    # hf_hub_download manages continuous stream download and local caching automatically
    local_filepath = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename
    )
    
    print(f"Loading target columns {columns} from local cached file...")
    # Read only the selected columns using PyArrow to keep a minimal memory footprint
    table = pq.read_table(local_filepath, columns=columns)
    df = table.to_pandas()
    
    print(f"Loaded {len(df):,} rows from local file: {local_filepath}")
    return df
