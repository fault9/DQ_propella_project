from __future__ import annotations

from pathlib import Path
from time import sleep

import duckdb
import polars as pl
from huggingface_hub import hf_hub_url, list_repo_files

from learned_reranker.data import canonical_doc_id


INPUT = Path("train/Human_train_propella.csv")
OUTPUT = Path("train/Human_train_finepdf.csv")


def main() -> None:
    propella = pl.read_csv(INPUT)
    ids = [str(value) for value in propella["id"].to_list()]
    canonical_ids = [canonical_doc_id(value) for value in ids]
    lookup_ids = sorted(set(ids) | set(canonical_ids) | {f"<urn:uuid:{value}>" for value in canonical_ids})
    shard_paths = [
        hf_hub_url("HuggingFaceFW/finepdfs", filename, repo_type="dataset")
        for filename in list_repo_files("HuggingFaceFW/finepdfs", repo_type="dataset")
        if filename.startswith("data/swe_Latn/train/") and filename.endswith(".parquet")
    ]

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.register("lookup_ids", pl.DataFrame({"id": lookup_ids}).to_arrow())
    con.register("order_ids", pl.DataFrame({"key": canonical_ids, "row_order": range(len(canonical_ids))}).to_arrow())
    con.execute(
        """
        CREATE TEMP TABLE matches (
            id VARCHAR,
            full_doc_lid VARCHAR,
            full_doc_lid_score DOUBLE,
            minhash_cluster_size BIGINT,
            duplicate_count BIGINT
        )
        """
    )
    for shard_path in shard_paths:
        for attempt in range(5):
            try:
                con.execute(
                    """
                    INSERT INTO matches
                    SELECT
                        id,
                        full_doc_lid,
                        full_doc_lid_score,
                        minhash_cluster_size,
                        duplicate_count
                    FROM read_parquet(?)
                    WHERE id IN (SELECT id FROM lookup_ids)
                    """,
                    [shard_path],
                )
                break
            except duckdb.HTTPException:
                if attempt == 4:
                    raise
                sleep(10 * (attempt + 1))
    con.create_function("canonical_doc_id", canonical_doc_id, [str], str)
    result = con.execute(
        """
        SELECT
            m.id,
            m.full_doc_lid,
            m.full_doc_lid_score,
            m.minhash_cluster_size,
            m.duplicate_count
        FROM order_ids o
        LEFT JOIN matches m ON canonical_doc_id(m.id) = o.key
        ORDER BY o.row_order
        """
    ).pl()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(OUTPUT)
    matched = result.filter(pl.col("id").is_not_null()).height
    print(f"Matched {matched}/{len(ids)} rows. Wrote {OUTPUT}.")


if __name__ == "__main__":
    main()
