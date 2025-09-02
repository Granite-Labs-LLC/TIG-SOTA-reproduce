# Download Scale-Invariant Feature Transform (SIFT) dataset

import io
import os
import struct
import tarfile
import urllib.request

url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
os.makedirs("SIFT", exist_ok=True)

def extract_vectors(f):
    """Read float vectors from .fvecs format file"""
    vectors = []
    while True:
        dim_data = f.read(4)  # Read the dimension (int32)
        if not dim_data:
            break  # End of file
        dim = struct.unpack('i', dim_data)[0]
        vectors.append(f.read(dim * 4))
    return vectors

print("Downloading SIFT dataset: " + url)
with urllib.request.urlopen(url) as response:
    obj = io.BytesIO(response.read())
    obj.seek(0)

with tarfile.open(fileobj=obj, mode="r:gz") as tar:
    for member in tar.getmembers():
        if "sift_base.fvecs" in member.name:
            with tar.extractfile(member) as f:
                database_vectors = extract_vectors(f)
        elif "sift_query.fvecs" in member.name:
            with tar.extractfile(member) as f:
                query_vectors = extract_vectors(f)
        elif "sift_groundtruth.ivecs" in member.name:
            with tar.extractfile(member) as f:
                groundtruth_vectors = extract_vectors(f)

vector_dims = len(database_vectors[0]) // 4
database_size = len(database_vectors)
num_queries = len(query_vectors)
nearest_neighbours = [x[:4] for x in groundtruth_vectors]

with open("SIFT/sift.bin", "wb") as f:
    f.write(vector_dims.to_bytes(length=4, byteorder="little", signed=False))
    f.write(database_size.to_bytes(length=4, byteorder="little", signed=False))
    f.write(num_queries.to_bytes(length=4, byteorder="little", signed=False))
    f.write(b"".join(database_vectors))
    f.write(b"".join(query_vectors))
    f.write(b"".join(nearest_neighbours))

print("Database vectors shape:", len(database_vectors), "x", len(database_vectors[0]) // 4)
print("Query vectors shape:", len(query_vectors), "x", len(query_vectors[0]) // 4)
print("Groundtruth vectors shape:", len(groundtruth_vectors), "x", len(groundtruth_vectors[0]) // 4)
print("Done")
