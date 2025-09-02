# Stat_Filter Algorithm: Optimal Parameters for TIG Testing

**For TIG Team - Critical Parameter Configuration Reference**

---

## ⚠️ INTELLIGENT PARAMETER TUNING

The `stat_filter` algorithm includes **automatic adaptation capabilities** but provides tuning parameters to optimize performance for specific datasets and use cases. The algorithm automatically:
- **Detects data characteristics** (positive/negative/mixed values) and adapts bit-slicing conversion
- **Computes base MAD values** dynamically; scale parameter multiplies this base value  
- **Adapts threshold scaling** based on query batch size for optimal filtering

The parameters below represent **optimal tuning configurations** derived from extensive testing.

---

## OPTIMAL HARDCODED CONFIGURATIONS BY DATASET

### TIG Test Dataset (achieved 31× improvement, 23ms median latency)

**MANDATORY HARDCODED VALUES for TIG Testing Framework:**

```rust
// These are the exact parameters that delivered 31x improvement over improved_search
// with 23ms median latency and 100% success rate in TIG harness testing
const STATFILT_BIT_MODE: u32 = 4;        // Default 4-bit precision
const STATFILT_TOP_K: usize = 20;        // Default internal shortlist size  
const STATFILT_MAD_SCALE: f32 = 1.0;     // Default MAD scaling factor
```

**Results achieved:**
- **31× faster** than improved_search baseline (23ms vs 712ms)
- **32× higher QPS** (58,514 vs 1,806 average QPS)  
- **100% success rate** (vs 90.1% for baseline)
- **0% recall failures** across all 576 TIG test configurations

**Critical**: These are the DEFAULT values that work optimally for the TIG test dataset. Unlike SIFT-1M or Fashion-MNIST which require specific tuning, the TIG dataset performs best with the algorithm's built-in defaults.

---

### SIFT-1M Dataset (1,000,000 vectors, 128 dimensions)

**OPTIMAL Tuning - Choose ONE configuration:**

```bash
# RECOMMENDED: Best Balance (97.56% recall, 161ms, 62,112 QPS)
STATFILT_BIT_MODE=4
STATFILT_TOP_K=5  
STATFILT_MAD_SCALE=0

# Alternative: Highest Recall (98.78% recall, 167ms, 59,880 QPS)  
STATFILT_BIT_MODE=4
STATFILT_TOP_K=8
STATFILT_MAD_SCALE=0

# Alternative: Fastest (96.41% recall, 159ms, 62,893 QPS)
STATFILT_BIT_MODE=4  
STATFILT_TOP_K=4
STATFILT_MAD_SCALE=0
```

**🚨 CRITICAL FOR SIFT**: `STATFILT_MAD_SCALE` **MUST** be `0`. Any non-zero value will cause severe recall degradation due to SIFT's heavy-tailed distribution characteristics.

---

### Fashion-MNIST-60K Dataset (60,000 vectors, 784 dimensions)

**OPTIMAL Tuning - Choose ONE configuration:**

```bash
# RECOMMENDED: Speed/Recall Balance (90.25% recall, 48ms, 208,333 QPS)
STATFILT_BIT_MODE=2
STATFILT_TOP_K=10
STATFILT_MAD_SCALE=0.4

# Alternative: High Precision (95.21% recall, 68ms, 147,059 QPS)
STATFILT_BIT_MODE=4
STATFILT_TOP_K=4  
STATFILT_MAD_SCALE=0.4

# Alternative: Small Batch Perfect Recall (100% recall, 4ms, 2,500 QPS)
STATFILT_BIT_MODE=4
STATFILT_TOP_K=5
STATFILT_MAD_SCALE=0
```

---

## INTELLIGENT PARAMETER SYSTEM

| Parameter | Auto-Behavior | Tuning Purpose | TIG Dataset | SIFT-1M Optimal | Fashion-MNIST Optimal |
|-----------|---------------|----------------|-------------|-----------------|----------------------|
| `STATFILT_BIT_MODE` | Auto-detects positive/negative/mixed data and adapts bit-slicing conversion | Precision vs speed tradeoff | **4** (PROVEN: 31× improvement) | **4** (preserves precision for 96-99% recall) | **2** (large batches) or **4** (small batches) |
| `STATFILT_TOP_K` | Performs exact distance check on candidate shortlist | Controls recall vs speed balance | **20** (PROVEN: 23ms median) | **4-8** (k=5 best balance) | **4-10** depending on target recall |
| `STATFILT_MAD_SCALE` | Computes base MAD automatically; this multiplies the base value | Statistical filtering aggressiveness | **1.0** (PROVEN: 100% success rate) | **0** (disables MAD for heavy-tailed data) | **0.4** (optimal filtering for large batches) |

**Ranges & Defaults:**
- `STATFILT_MAD_SCALE`: float, 0 ≤ x ≤ 5 (default 1.0). **0 = off**, **≥5 = wide open/off**
- `STATFILT_TOP_K`: integer, 1 ≤ x ≤ 20 (default 20)  
- `STATFILT_BIT_MODE`: integer {2, 4} (default 4)

---

## ❌ CRITICAL MISTAKES TO AVOID

| Mistake | Dataset | Consequence |
|---------|---------|-------------|
| `STATFILT_MAD_SCALE=0.5` | SIFT-1M | **Recall degradation** (can drop below 50%) |
| `STATFILT_TOP_K<4` | SIFT-1M | **Missing optimal candidates** (recall collapse) |
| `STATFILT_MAD_SCALE>1.0` | Fashion-MNIST | **Defeats statistical filtering** (performance loss) |

---

## PERFORMANCE VALIDATION

These parameters deliver breakthrough performance improvements:

- **TIG Dataset**: **31× faster** than improved_search baseline (23ms vs 712ms) with **100% success rate**
- **SIFT-1M**: 97.56% recall in 161ms (vs. seconds for index-based methods) - **20-800× speedups** over NVIDIA cuVS GPU
- **Fashion-MNIST**: 90.25% recall in 48ms with 208,333 QPS
- **Zero build time** vs. index methods requiring minutes of preprocessing

---

## TESTING HARDWARE REFERENCE

Results obtained on:
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CPU**: AMD EPYC 7513 (single-CPU build constraint)  
- **CUDA**: 12.6.3 (container) / 12.9 (host)

**Note**: Different hardware may produce different absolute timing but relative performance advantages should remain consistent.

---

## TUNING PHILOSOPHY

**Intelligent Defaults**: The algorithm provides conservative defaults (MAD_SCALE=1.0, TOP_K=20, BIT_MODE=4) that work across most datasets while maintaining high recall.

**Dataset-Specific Optimization**: The configurations above represent optimal tuning discovered through extensive benchmarking. Each parameter leverages the algorithm's automatic adaptation capabilities:

- **MAD Strategy Toggle**: For heavy-tailed distributions (SIFT), disable MAD filtering (MAD_SCALE=0) while keeping the pipeline intact. For well-behaved distributions (Fashion-MNIST), use moderate MAD scaling (0.4) for effective pruning.

- **Adaptive Scaling**: The algorithm automatically adjusts filtering aggressiveness based on batch size - larger batches benefit from more aggressive filtering where statistical benefits are greater.

**Auto-Detection Features**: The algorithm handles complex data transformations automatically, including per-dimension range selection and signed/unsigned data mapping, so tuning focuses on high-level speed/recall tradeoffs rather than low-level implementation details.

## IMPLEMENTATION NOTE

**Current Parameter System:**
1. **Environment variables**: Algorithm reads `STATFILT_BIT_MODE`, `STATFILT_TOP_K`, `STATFILT_MAD_SCALE`
2. **Intelligent defaults**: Conservative fallback values (MAD_SCALE=1.0, TOP_K=20, BIT_MODE=4)
3. **Reference files**: `.parm` files document optimal values for each dataset

**Parameter Reference Files:**
The repository includes parameter files with optimal configurations already in place:
- `data/SIFT/sift.parm` - Documents optimal SIFT parameters
- `data/Fashion-MNIST/fashion_mnist.parm` - Documents optimal Fashion-MNIST parameters

**Current Implementation**: The algorithm reads environment variables directly. Use the values from these reference files when configuring your testing environment or hardcoding parameters.

## ⚠️ IMPORTANCE OF PARAMETER CONFIGURABILITY

**Why Parameter Flexibility is Critical:**

The `stat_filter` algorithm's breakthrough performance depends on **dataset-specific optimization**. While the algorithm includes intelligent auto-adaptation for data characteristics, the tuning parameters are essential for achieving optimal performance across different datasets and use cases:

### Dataset-Specific Requirements
- **SIFT-1M**: Requires `MAD_SCALE=0` due to heavy-tailed distribution - any other value causes significant recall degradation
- **Fashion-MNIST**: Benefits from `MAD_SCALE=0.4` for optimal filtering balance on large batches
- **Batch Size Sensitivity**: Small batches (10 queries) vs large batches (10,000 queries) need different configurations

### Algorithm Design Philosophy
The algorithm was **specifically designed as an adaptive, configurable system**:
- **Auto-detection** handles complex data transformations (positive/negative values, bit-slicing conversion)
- **Tuning parameters** allow optimization for specific datasets and performance requirements
- **Adaptive scaling** automatically adjusts based on batch size, but optimal ranges vary by dataset

### Performance Impact
**Without proper parameter tuning:**
- SIFT-1M with wrong MAD_SCALE: **Recall can drop below 50%** instead of 97.56%
- Fashion-MNIST with suboptimal settings: **Performance degrades from 208,333 QPS to <50,000 QPS**
- Generic "one-size-fits-all" parameters: **Cannot achieve the documented 20-800× speedups**

### Recommendation for Testing Framework
**Flexible Configuration Preferred**: If possible, maintain parameter configurability to:
- Enable proper dataset-specific optimization  
- Allow validation of the full algorithm capability
- Support future dataset additions without code changes
- Preserve the algorithm's adaptive design benefits

**If Hardcoding is Necessary**: Use the exact values from the reference files below, understanding that this limits the algorithm to the specific datasets tested and may not generalize to new datasets or use cases.

## KEY INSIGHT: INTELLIGENT AUTOMATION + OPTIMAL TUNING

The stat_filter algorithm represents a **breakthrough in adaptive vector search**:

- **Automatic data adaptation**: Handles complex data characteristics (positive/negative/mixed, heavy-tailed distributions) without manual configuration
- **Intelligent threshold computation**: Computes optimal MAD values automatically based on data distribution  
- **Batch-aware scaling**: Dynamically adapts filtering aggressiveness based on query batch size
- **Tunable optimization**: Provides high-level tuning parameters for dataset-specific optimization

**The tuning parameters above leverage these automatic capabilities** to achieve optimal performance for specific datasets and use cases, rather than requiring manual low-level parameter tweaking.

---

**Contact**: For questions about parameter selection or performance issues, refer to the complete evidence submission and README.md documentation.

**Source**: Parameters extracted from comprehensive NVIDIA cuVS baseline testing documented in advance evidence submission.
