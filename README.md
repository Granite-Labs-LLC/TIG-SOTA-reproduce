# TIG Stat Filter Algorithm - Build and Test Tutorial

This tutorial walks you through building and running the `stat_filter` algorithm within the TIG vector search evaluation harness.

## Overview

The `stat_filter` algorithm is a vector search implementation that uses statistical filtering techniques with configurable bit modes and top-K selection. This repository contains both "new" and "old" versions of the algorithm for comparison and SOTA (State of the Art) testing.

## ⚠️ CRITICAL: Parameter Configuration Requirements

**MANDATORY READING**: The `stat_filter` algorithm's performance is **extremely sensitive** to parameter configuration. Using incorrect parameters will result in poor recall rates and misleading performance results.

### Required Parameters by Dataset

**For SIFT Dataset (1M vectors, 128 dimensions):**
```bash
# REQUIRED Configuration 1: High recall, 2-bit
STATFILT_BIT_MODE=2 STATFILT_TOP_K=20 STATFILT_MAD_SCALE=5

# REQUIRED Configuration 2: High precision, 4-bit  
STATFILT_BIT_MODE=4 STATFILT_TOP_K=5 STATFILT_MAD_SCALE=5
```

**For Fashion-MNIST Dataset (60K vectors, 784 dimensions):**
```bash
# REQUIRED Configuration 1: Balanced performance, 2-bit
STATFILT_BIT_MODE=2 STATFILT_TOP_K=10 STATFILT_MAD_SCALE=0.4

# REQUIRED Configuration 2: High precision, 4-bit
STATFILT_BIT_MODE=4 STATFILT_TOP_K=4 STATFILT_MAD_SCALE=0.4
```

### ❌ Common Parameter Mistakes

**DO NOT use these combinations - they will produce poor results:**
- MAD_SCALE=0.5 on SIFT dataset → **Will severely degrade recall performance**
- Low TOP_K values (<5) on SIFT → **Will miss optimal candidates**
- High MAD_SCALE (>1.0) on Fashion-MNIST → **Will defeat the statistical filtering**

### Why Parameters Matter

- **MAD_SCALE**: Controls statistical filtering aggressiveness
  - SIFT needs `MAD_SCALE=5` (effectively disables filtering) due to data distribution
  - Fashion-MNIST works well with `MAD_SCALE=0.4` (aggressive filtering)
  
- **TOP_K**: Number of candidates to consider
  - SIFT requires higher values (5-20) due to dataset complexity
  - Fashion-MNIST can use lower values (4-10) effectively
  
- **BIT_MODE**: Quantization precision vs speed tradeoff
  - 2-bit: Faster execution, lower memory usage
  - 4-bit: Better precision, higher computational cost

**The parameters used in this tutorial are the result of extensive optimization and testing. Deviating from these values without understanding the algorithmic implications will likely produce suboptimal results.**


## Repository Structure

```
REPRODUCE_SOTA/
├── enter_docker.sh             # Docker container entry script
├── README.md                   # Main documentation with example outputs
├── data/                       # Test datasets (SIFT, Fashion-MNIST) download scripts
├── stat_filter_new/            # New version of stat_filter algorithm
│   ├── BUILD_RUN.sh            # Build and test runner script
│   ├── src/main.rs             # Algorithm implementation
│   └── target/                 # Compiled binaries (when running `BUILD_RUN.sh rebuild`)
└── tig-monorepo-cb-new/        # TIG benchmarker structure (modified to print more data)
```

## Prerequisites

- Docker with GPU support (`--gpus all`)
- NVIDIA GPU with CUDA support
- The TIG vector search Docker image

## Hardware Testing Environment

The benchmark results in this tutorial were obtained using the following hardware configuration:

### Host System
- **CPU**: AMD EPYC 7513 32-Core Processor
- **CPU Cores**: 30 cores available (single-CPU build constraint applied for fairness)
- **Architecture**: x86_64
- **GPU**: NVIDIA GeForce RTX 4090
- **GPU Memory**: 24,564 MiB (24GB VRAM)
- **NVIDIA Driver**: 575.51.03
- **Host CUDA Version**: 12.9

### Docker Container Environment  
- **Container CUDA Version**: 12.6.3
- **CUDA Compiler**: nvcc V12.6.85
- **Container Base**: `ghcr.io/tig-foundation/tig-monorepo/vector_search/dev:0.0.1`

### Performance Note
**Important**: If your hardware differs significantly from this configuration, especially GPU model, VRAM capacity, or CUDA version, you may observe different performance results. The algorithm's relative performance advantages should remain consistent, but absolute timing values may vary.

**Build Constraint**: All benchmarks follow a "single-CPU build" policy for fair comparison, meaning only one CPU core is used for compilation even though more cores are available.

## Important: TIG Runtime Modifications

**Enhanced Output Reporting**: The TIG runtime in this repository has been modified to provide additional performance metrics beyond the standard implementation. Specifically:

- **Recall Rate Reporting**: The code now reports recall rate percentages (e.g., "Recall rate: 86.85% (8685 / 10000 queries matched)") in addition to the standard average/optimal distance metrics.
- **Enhanced Benchmarking**: This modification was necessary to properly evaluate and compare algorithm performance, as recall rate is a critical metric for assessing the quality of approximate nearest neighbor search results.

**Why This Matters**: Standard TIG evaluations focus primarily on distance metrics, but for research and SOTA comparisons, understanding recall rates is essential. The recall rate tells you how often the algorithm finds the true nearest neighbor versus a close approximation.

**Cargo.toml Integration**: The project structure has been modified to work with the local TIG benchmarker structure (`tig-monorepo-cb-new`) rather than downloading from GitHub, since the SOTA testing framework only works with algorithms that have been assigned branches in the official repository.

## Step-by-Step Guide

### 1. Enter the Docker Environment

Start by entering the TIG vector search development container:

```bash
bash enter_docker.sh
```

This script will:
- Mount your current directory as `/app` in the container
- Enable GPU access for CUDA operations
- Start an interactive bash session in the container

Expected output:
```
Entering Docker container for vector_search development...
```

### 2. Navigate to the Algorithm Directory

Choose which version to work with:

**For the NEW version (recommended):**
```bash
cd stat_filter_new
```

### 3. Build and Run Tests

The `BUILD_RUN.sh` script handles both building and running the algorithm with predefined test configurations.

#### Quick Run (if already built)
```bash
sh BUILD_RUN.sh
```

#### Full Rebuild and Run
If you've made changes to the source code or want to ensure a clean build:
```bash
sh BUILD_RUN.sh rebuild
```

#### Build Only
To just compile without running tests:
```bash
sh BUILD_RUN.sh build
```

## Test Configurations

The `stat_filter` algorithm runs with different configurations optimized for each dataset:

### SIFT Dataset Tests

**Configuration 1: 2-bit mode, K=20**
```bash
STATFILT_BIT_MODE=2 STATFILT_TOP_K=20 STATFILT_MAD_SCALE=5
```
- **Purpose**: High recall with 2-bit quantization
- **Expected Results**: 86.85% recall rate, 261.352ms runtime

**Configuration 2: 4-bit mode, K=5**
```bash
STATFILT_BIT_MODE=4 STATFILT_TOP_K=5 STATFILT_MAD_SCALE=5
```
- **Purpose**: Better precision with 4-bit quantization
- **Expected Results**: 97.56% recall rate, 161.494ms runtime

### Fashion-MNIST Dataset Tests

**Configuration 1: 2-bit mode, K=10 with MAD scaling**
```bash
STATFILT_BIT_MODE=2 STATFILT_TOP_K=10 STATFILT_MAD_SCALE=0.4
```
- **Expected Results**: 90.25% recall rate, 46.595ms runtime

**Configuration 2: 4-bit mode, K=4**
```bash
STATFILT_BIT_MODE=4 STATFILT_TOP_K=4 STATFILT_MAD_SCALE=0.4
```
- **Expected Results**: 95.21% recall rate, 66.401ms runtime

## Configuration Parameters

The algorithm behavior is controlled by environment variables:

- **`STATFILT_BIT_MODE`**: Quantization bit depth (2 or 4)
  - 2-bit: Faster, lower memory, reduced precision
  - 4-bit: Slower, higher memory, better precision

- **`STATFILT_TOP_K`**: Number of top candidates to consider
  - Lower values: Faster but may miss optimal results
  - Higher values: More comprehensive but slower

- **`STATFILT_MAD_SCALE`**: Median Absolute Deviation scaling factor
  - Controls the statistical filtering threshold
  - SIFT uses 5 (no effective MAD filtering)
  - Fashion-MNIST uses 0.4 (aggressive filtering)

## Understanding the Output

### Timing Breakdown
```
Time for nonce: 261.352 ms (sum+stats: 7.303 ms + mad_sort: 0.021 ms + slice: 0.176 ms + search: 252.848 ms + rerank 1.004 ms)
```
*Actual timing from SIFT 2-bit, K=20 configuration*

- **sum+stats**: Time to compute statistics
- **mad_sort**: Time for MAD calculation and sorting  
- **slice**: Time for bit-slicing operations
- **search**: Core search algorithm time
- **rerank**: Time to rerank final results

### Performance Metrics
```
Recall rate: 86.85% (8685 / 10000 queries matched)
instance: "sift.bin", avg_dist: 188.51115, optimal_dist: 187.7512, time: 261 ms
```
*Actual results from SIFT 2-bit, K=20 configuration*

**Note**: The recall rate output shown above is available due to the enhanced TIG runtime modifications described earlier. Standard TIG implementations typically only report distance metrics.

- **Recall rate**: Percentage of queries that found the true nearest neighbor *(enhanced output)*
- **avg_dist**: Average distance to found neighbors *(standard TIG output)*
- **optimal_dist**: Average distance to true nearest neighbors *(standard TIG output)*
- **time**: Total execution time *(standard TIG output)*

## Customizing Test Runs

### Running Individual Configurations

You can run specific configurations manually:

```bash
# Example: Test SIFT with custom parameters
STATFILT_BIT_MODE=3 STATFILT_TOP_K=15 STATFILT_MAD_SCALE=2.0 \
    target/release/vector_search_evaluator ../data/SIFT
```

### Modifying the Algorithm

1. Edit the source code in `src/main.rs`
2. Always rebuild after changes:
   ```bash
   sh BUILD_RUN.sh rebuild
   ```
3. The rebuild is necessary due to CUDA compilation dependencies

## Datasets

### Manual Dataset Download

If the datasets are not already present in the repository, you can download them manually:

```bash
# Navigate to the data directory
cd ../data

# Install Python requirements (if needed)
pip3 install -r requirements.txt

# Download SIFT dataset
python3 download_sift.py

# Download Fashion-MNIST dataset
python3 download_fashion_mnist.py

# Verify downloads
ls -la SIFT/
ls -la Fashion-MNIST/
```

### SIFT Dataset
- **Size**: 1,000,000 base vectors, 10,000 query vectors
- **Dimensions**: 128
- **Type**: Scale-Invariant Feature Transform descriptors
- **Location**: `../data/SIFT/sift.bin`
- **Source**: [INRIA TEXMEX Corpus](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)

### Fashion-MNIST Dataset  
- **Size**: 60,000 base vectors, 10,000 query vectors
- **Dimensions**: 784
- **Type**: Clothing item image vectors
- **Location**: `../data/Fashion-MNIST/784-euclidean.bin`
- **Source**: [HuggingFace Dataset](https://huggingface.co/datasets/open-vdb/fashion-mnist-784-euclidean)

## Troubleshooting

### Common Issues

1. **Build Errors**: Always use `sh BUILD_RUN.sh rebuild` after source changes
2. **CUDA Errors**: Ensure you're running inside the Docker container with GPU access
3. **Data Missing**: Make sure datasets are downloaded in the `../data/` directory
4. **Permission Issues**: Scripts should be executable; use `chmod +x BUILD_RUN.sh` if needed

### Verifying Setup

Check that you're in the correct environment:
```bash
# Should show CUDA-capable GPU
nvidia-smi

# Should show the compiled binary
ls -la target/release/vector_search_evaluator

# Should show available datasets
ls -la ../data/
```

## Performance Tuning

### For SIFT Dataset
- Use higher `STATFILT_TOP_K` values (10-20) for better recall
- `STATFILT_MAD_SCALE=5` effectively disables MAD filtering
- 4-bit mode generally provides better accuracy vs speed tradeoff

### For Fashion-MNIST Dataset  
- Lower `STATFILT_MAD_SCALE` values (0.4) work well due to data characteristics
- Smaller `STATFILT_TOP_K` values (4-10) are sufficient
- 2-bit mode can be effective for this dataset

## Expected Performance Benchmarks

Based on the example runs, you should expect:

| Dataset | Configuration | Recall Rate | Runtime | 
|---------|---------------|-------------|---------|
| SIFT | 2-bit, K=20 | 86.85% | 261.352ms |
| SIFT | 4-bit, K=5 | 97.56% | 161.494ms |
| Fashion-MNIST | 2-bit, K=10 | 90.25% | 46.595ms |
| Fashion-MNIST | 4-bit, K=4 | 95.21% | 66.401ms |

## Next Steps

- Experiment with different parameter combinations
- Compare performance between `stat_filter_new` and `stat_filter_old`
- Analyze the detailed timing breakdowns to identify optimization opportunities
- Test with different datasets if available

## Notes

- The algorithm integrates with the TIG benchmarker structure via modified `Cargo.toml` files
- Results include both recall rates and distance metrics for comprehensive evaluation
- The setup is designed to work with the broader TIG ecosystem for algorithm development and testing
