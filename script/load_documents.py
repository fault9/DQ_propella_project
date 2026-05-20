from datasets import load_dataset

finepdfs = load_dataset(
    "HuggingFaceFW/finepdfs",
    "swe_Latn",
    split="train",
    streaming=True,
)

for row in finepdfs:
    print(row.keys())
    print(row)
    break


propella = load_dataset(
    "openeurollm/propella-annotations",
    "finepdfs",
    split="swe_Latn",
    streaming=True,
)

for row in propella:
    print(row.keys())
    print(row)
    break
