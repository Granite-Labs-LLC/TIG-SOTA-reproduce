# Download Fashion-MNIST dataset for ANN evaluation

import pandas as pd
import io
import os
import numpy as np
import requests

os.makedirs("Fashion-MNIST", exist_ok=True)
BASE_URL = "https://huggingface.co/datasets/open-vdb/fashion-mnist-784-euclidean/resolve/main"

# download query vectors
url = f"{BASE_URL}/test/test-00001-of-00001.parquet"
print("Downloading query vectors: " + url)
resp = requests.get(url, stream=True)
buffer = io.BytesIO(resp.content)
df = pd.read_parquet(buffer, engine="pyarrow")
query_vectors = np.stack(df["emb"].values).astype(np.float32)
num_queries, vector_dims = query_vectors.shape

# download database vectors
url = f"{BASE_URL}/train/train-00001-of-00001.parquet"
print("Downloading database vectors: " + url)
resp = requests.get(url, stream=True)
buffer = io.BytesIO(resp.content)
df = pd.read_parquet(buffer, engine="pyarrow")
database_vectors = np.stack(df["emb"].values).astype(np.float32)
database_size = len(database_vectors)

# download nearest neighbors
url = f"{BASE_URL}/neighbors/neighbors-vector-emb-pk-idx-expr-None-metric-l2.parquet"
print("Downloading nearest neighbors: " + url)
resp = requests.get(url, stream=True)
buffer = io.BytesIO(resp.content)
df = pd.read_parquet(buffer, engine="pyarrow")
# keep just the nearest neighbour
nearest_neighbours = np.stack(df["neighbors_id"].values)[:, 0].astype(np.int32)

with open("Fashion-MNIST/784-euclidean.bin", "wb") as f:
    f.write(vector_dims.to_bytes(length=4, byteorder="little", signed=False))
    f.write(database_size.to_bytes(length=4, byteorder="little", signed=False))
    f.write(num_queries.to_bytes(length=4, byteorder="little", signed=False))
    f.write(database_vectors.flatten().tobytes())
    f.write(query_vectors.flatten().tobytes())
    f.write(nearest_neighbours.flatten().tobytes())

print("Done")