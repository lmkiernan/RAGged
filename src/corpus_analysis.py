#corpus stats, entropy, length histograms, etc.

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chunks'))

chunk_names = [
    fn for fn in os.listdir(chunks_dir) if fn.endswith(".json")
]

all_chunks = []

print ("Found file names: ")
for fn in chunk_names:
    file_path = os.path.join(chunks_dir, fn)
    with open(file_path, "r", encoding="utf-8") as f:
        chunk_list = json.load(f)
    all_chunks.append(chunk_list)

rows = []

for chunk_list in all_chunks:
    for chunk in chunk_list:
        rows.append(chunk)

df = pd.DataFrame(rows)

df['token_count'] = df['tokens'].apply(lambda d: list(d.values())[0] if isinstance(d, dict) else d)

chunk_based = df[["chunk_id", "strategy", "source", "token_count"]]

grouped_df = df[["strategy", "source", "token_count"]]

grouped_df['num_chunks'] = grouped_df.groupby(['strategy', 'source'])['token_count'].transform('count')

doc_strat_based = grouped_df.groupby(['strategy', 'source']).agg(
    token_count=('token_count', 'sum'),
    num_chunks=('token_count', 'count')
).reset_index()

doc_strat_based['avg_tokens_per_chunk'] = doc_strat_based['token_count'] / doc_strat_based['num_chunks']

strat_based = doc_strat_based.groupby('strategy').agg(
    avg_chunks_per_doc=('num_chunks', 'mean'),
    avg_tokens_per_chunk=('avg_tokens_per_chunk', 'mean'),
    max_chunks_in_doc=('num_chunks', 'max')
).reset_index()

print(doc_strat_based)
print(chunk_based)
print(strat_based)