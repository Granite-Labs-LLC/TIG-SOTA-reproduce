# TIG SOTA Metrics for Vector Search

This directory provides tools to evaluate TIG's vector search algorithms against academic test instances, specifically the SIFT dataset, and the Fashion-MNIST dataset. 

## Getting Started

### Downloading Test Instances

#### SIFT

SIFT is a dataset of Scale-Invariant Feature Transform descriptors. Original source: [INRIA TEXMEX Corpus](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz):
- Dimensions: 128
- Base vectors: 1,000,000
- Query vectors: 10,000

To download:
```bash
cd data
python3 download_sift.py
```

#### Fashion-MNIST

The Fashion-MNIST is dataset of clothing item images encoded as vectors. Original source: [HuggingFace Dataset](https://huggingface.co/datasets/open-vdb/fashion-mnist-784-euclidean):
- Dimensions: 784
- Base vectors: 60,000
- Query vectors: 10,000

To download:
```
cd data
pip3 install -r requirements.txt
python3 download_fashion_mnist.py
```

### Evaluating Algorithms

Use the provided script to evaluate your SAT algorithm against a benchmark set:

```bash
# Usage
bash run.sh TEST_INSTANCES_FOLDER ALGORITHM

# Example
bash run.sh data/SIFT simple_search
```

## Output

The evaluation outputs the average query to database euclidean distance for each test instance. Example output:

```
Loading PTX...
CUDA code compiled successfully.
vector_dims: 128, database_size: 1000000, num_queries: 10000
instance: "sift.bin", avg_dist: 187.7513, optimal_dist: 187.75108, time: 184523 ms
```