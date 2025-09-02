/*!
Copyright 2025 Granite Labs LLC

Identity of Submitter [name of person or entity that submits the Work to TIG]

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

//
// stat_filter
//
// Filtering based on Median Absolute Deviation (MAD):
// We compute the median of all L2 norms, then calculate the MAD (median of
// absolute deviations from the median). The threshold is set to:
//      norm_threshold = scale_factor × MAD × 1.4826
// The factor 1.4826 scales MAD to match the standard deviation for normally
// distributed data. This makes the filter more robust to outliers compared to
// filtering methods based on mean and standard deviation, which are more
// sensitive to extreme values.
//
// Reference:
// - NIST Engineering Statistics Handbook:
//   https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
// - See also: https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm
//

/*!
Copyright 2025 Granite Labs LLC
...
*/
#include <float.h>
#include <math_constants.h>   // defines CUDART_INF_F, CUDART_NAN_F, etc.

#define USE_4_BITS
//#define USE_2_BITS

/* Kernels used by 4-bit bit-slicing:

    compute_dim_stats_kernel
    build_divisors_from_max_kernel
    f32_to_u4_packed_perdim_kernel
    compute_vector_stats_u4_packed_kernel
    u4_packed_to_bitplanes
    compute_norms_u4_bitsliced_kernel
    find_topk_neighbors_u4_bitsliced_kernel
    refine_fn

*/


//-------------------- Dimension Stats --------------------------

__device__ inline void atomicMaxFloat(float* addr, float val) {
    // Safe for non-negative floats (your data is 0..255)
    int* addr_i = reinterpret_cast<int*>(addr);
    int  old    = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

extern "C" __global__ void compute_dim_stats_kernel(
    const float* __restrict__ db,  // [num_vecs * dims], original floats
    float* __restrict__ out_max,   // [dims], init to 0 on host
    int num_vecs,
    int dims)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const float* row = db + (size_t)v * dims;
    for (int d = 0; d < dims; ++d) {
        float x = row[d];
        atomicMaxFloat(&out_max[d], x);
    }
}



//-------------------- Calculate Dimension Divisors -------------

// Build per-dimension divisors from max.  Scale the max down so
// we throw away outliers.  For example: 
//     s[d] = max(0.90 * max[d] / 16, 1.0)
#ifndef FRAC_OF_MAX
//#define FRAC_OF_MAX 1.00f
#define FRAC_OF_MAX 0.90f
//#define FRAC_OF_MAX 0.80f
#endif
#ifndef LEVELS
#  ifdef USE_4_BITS
#    define LEVELS 16.0f
#  elif defined(USE_2_BITS)
#    define LEVELS 4.0f
#  else
#    error 'unknown bit count'
#  endif
#endif
#ifndef MIN_STEP
// This allows us to divide by the result ... no zeros
#define MIN_STEP 1.0f
#endif

extern "C" __global__ void build_divisors_from_max_kernel(
    const float* __restrict__ dim_max, // [dims]
    float* __restrict__ s,             // [dims] (output... pre-allocated)
    int dims)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dims) return;

    float mx = fmaxf(0.0f, dim_max[d]);                  // guard negatives/NaN-ish
    mx = fminf(255.0f, mx);                              // guard against overage
    float sd = FRAC_OF_MAX * mx / LEVELS;                // example: 0.90 * max / 16
    s[d] = fmaxf(sd, MIN_STEP);                          // floor at 1.0

    //printf("d:%d  s[d]:%f  max[d]:%f\n",d,s[d],dim_max[d]);
}


//-------------------- Dimension Aware Conversion ---------------


// Packs two 4-bit codes per byte: even dim -> low nibble, odd dim -> high nibble.
// out size per row = (dims + 1) >> 1 bytes.
extern "C" __global__ void f32_to_u4_packed_perdim_kernel(
    const float*  __restrict__ in,   // [num_vecs * dims], original floats
    const float*  __restrict__ s,    // [dims], per-dim divisors (>= 1)
    uint8_t*      __restrict__ out,  // [num_vecs * ((dims+1)>>1)], packed u4
    int num_vecs,
    int dims)
{
    int row_bytes = (dims + 1) >> 1;            // 2 dims per byte
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = num_vecs * row_bytes;
    if (bi >= total_bytes) return;

    int v = bi / row_bytes;                     // vector id
    int b = bi % row_bytes;                     // byte index within row
    int j0 = (b << 1);                          // even dim
    int j1 = j0 + 1;                            // odd dim

    const float* vin = in + (size_t)v * dims;
    const float* ss  = s;

    // Dim j0 -> low nibble
    float x0 = (j0 < dims) ? vin[j0] : 0.0f;
    float y0 = fminf(fmaxf(x0, 0.0f), CLIP_MAX);
    float sj0 = ss[j0 < dims ? j0 : 0];         // safe even if j0>=dims
    int   q0  = (y0 <= 0.0f) ? 0 : __float2int_rn(y0 / sj0);
    q0 = max(0, min(15, q0));

    // Dim j1 -> high nibble (or 0 if odd dim does not exist)
    int q1 = 0;
    if (j1 < dims) {
        float x1 = vin[j1];
        float y1 = fminf(fmaxf(x1, 0.0f), CLIP_MAX);
        float sj1 = ss[j1];
        q1 = (y1 <= 0.0f) ? 0 : __float2int_rn(y1 / sj1);
        q1 = max(0, min(15, q1));
    }

    out[(size_t)v * row_bytes + b] = (uint8_t)((q1 << 4) | (q0 & 0x0F));
}




//----------------- Vector Stats After Conversion ---------------

extern "C" __global__ void compute_vector_stats_u4_packed_kernel(
    const uint8_t* __restrict__ vectors_packed,  // [num_vecs * ((dims+1)>>1)]
    float* __restrict__ norm_l2,                 // [num_vecs]
    float* __restrict__ norm_l2_squared,         // [num_vecs]
    int num_vecs,
    int dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vecs) return;

    const int row_bytes = (dims + 1) >> 1; // 2 dims per byte
    const uint8_t* row = vectors_packed + (size_t)i * row_bytes;

    double acc = 0.0;
    int j = 0;

    // Process full bytes
    for (int by = 0; by < row_bytes; ++by) {
        uint8_t b = row[by];

        // low nibble -> dim j
        if (j < dims) {
            double v = (double)(b & 0x0Fu);
            acc = fma(v, v, acc);
            ++j;
        }

        // high nibble -> dim j
        if (j < dims) {
            double v = (double)(b >> 4);
            acc = fma(v, v, acc);
            ++j;
        }
    }

    float accf = (float)acc;
    norm_l2_squared[i] = accf;
    norm_l2[i]         = sqrtf(accf);
}


//----------------- Nearest Neighbor Search ---------------------

#ifndef KMAX
#define KMAX 64
#endif

__device__ __forceinline__ void topk_try_insert(float d, int i, float* best_d, int* best_i, int K) {
    if (d >= best_d[K-1]) return;
    int pos = K-1;
    while (pos > 0 && d < best_d[pos-1]) {
        best_d[pos] = best_d[pos-1];
        best_i[pos] = best_i[pos-1];
        --pos;
    }
    best_d[pos] = d; best_i[pos] = i;
}



// refine_topk_rerank_kernel.cu

extern "C" __global__ void refine_topk_rerank_kernel(
    const float* __restrict__ query_vectors,    // [num_queries * dim]
    const float* __restrict__ db_vectors,       // [db_len * dim]
    const int*   __restrict__ candidates,       // [num_queries * K]
    int*         __restrict__ out_index,        // [num_queries]
    float*       __restrict__ out_distance,     // [num_queries] (squared L2)
    const int num_queries,
    const int dim,
    const int K
)
{
    int q = blockIdx.x;
    if (q >= num_queries) return;

    extern __shared__ unsigned char shared[];
    float* sm_q = reinterpret_cast<float*>(shared);
    float* red  = sm_q + dim;       // reduction buffer, length = blockDim.x

    // Cache query vector into shared memory
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        sm_q[j] = query_vectors[q * dim + j];
    }
    __syncthreads();

    float best_d = FLT_MAX;
    int   best_i = -1;

    // For each candidate, compute exact squared L2 distance in parallel
    for (int t = 0; t < K; ++t) {
        int db_idx = candidates[q * K + t];
        if (db_idx < 0) continue;

        const float* db = &db_vectors[db_idx * dim];

        // Partial sum over dimensions (strided by thread)
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            float diff = sm_q[j] - db[j];
            sum = fmaf(diff, diff, sum);
        }

        // Block-wide reduction into red[0]
        red[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float d = red[0];
            if (d < best_d) { best_d = d; best_i = db_idx; }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_index[q]    = best_i;
        out_distance[q] = best_d;
    }
}



// ===============================
// 4-bit bit-sliced helper kernels
// ===============================

extern "C" __global__ void u4_packed_to_bitplanes(
    const uint8_t* __restrict__ packed,   // [num_vecs][(D+1)>>1] ; 2 dims/byte (lo nibble, hi nibble)
    unsigned long long* __restrict__ out_b0, // [num_vecs][W] ; bit 0 plane
    unsigned long long* __restrict__ out_b1, // [num_vecs][W] ; bit 1 plane
    unsigned long long* __restrict__ out_b2, // [num_vecs][W] ; bit 2 plane
    unsigned long long* __restrict__ out_b3, // [num_vecs][W] ; bit 3 plane (MSB)
    int num_vecs, int D, int W)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const uint8_t* row = packed + (size_t)v * ((D + 1) >> 1);

    for (int w = 0; w < W; ++w) {
        unsigned long long b0 = 0ULL, b1 = 0ULL, b2 = 0ULL, b3 = 0ULL;
        int j_base = w << 6; // 64 dims per 64b word
        #pragma unroll
        for (int t = 0; t < 64; ++t) {
            int j = j_base + t;
            if (j >= D) break;

            int by = j >> 1;                 // 2 dims per byte
            uint8_t code = (j & 1)
                ? (row[by] >> 4) & 0xF       // high nibble
                : (row[by]      ) & 0xF;     // low nibble

            if (code & 0x1) b0 |= (1ULL << t);
            if (code & 0x2) b1 |= (1ULL << t);
            if (code & 0x4) b2 |= (1ULL << t);
            if (code & 0x8) b3 |= (1ULL << t);
        }
        out_b0[(size_t)v * W + w] = b0;
        out_b1[(size_t)v * W + w] = b1;
        out_b2[(size_t)v * W + w] = b2;
        out_b3[(size_t)v * W + w] = b3;
    }
}

// Optional: compute bin-space L2 norms from 4-bit bitplanes
extern "C" __global__ void compute_norms_u4_bitsliced_kernel(
    const unsigned long long* __restrict__ b0, // [num_vecs][W]
    const unsigned long long* __restrict__ b1, // [num_vecs][W]
    const unsigned long long* __restrict__ b2, // [num_vecs][W]
    const unsigned long long* __restrict__ b3, // [num_vecs][W]
    float* __restrict__ norm_l2,               // [num_vecs]
    float* __restrict__ norm_l2_squared,       // [num_vecs]
    int num_vecs, int D, int W)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const unsigned long long* x0 = b0 + (size_t)v * W;
    const unsigned long long* x1 = b1 + (size_t)v * W;
    const unsigned long long* x2 = b2 + (size_t)v * W;
    const unsigned long long* x3 = b3 + (size_t)v * W;

    int s0=0,s1=0,s2=0,s3=0, p01=0,p02=0,p03=0,p12=0,p13=0,p23=0;

    unsigned long long tail_mask;
    int tail_bits = D & 63;
    tail_mask = (tail_bits==0) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL<<tail_bits)-1ULL);

    for (int w = 0; w < W; ++w) {
        unsigned long long mask = (w==W-1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;
        unsigned long long a0 = x0[w] & mask;
        unsigned long long a1 = x1[w] & mask;
        unsigned long long a2 = x2[w] & mask;
        unsigned long long a3 = x3[w] & mask;

        s0  += __popcll(a0);
        s1  += __popcll(a1);
        s2  += __popcll(a2);
        s3  += __popcll(a3);

        p01 += __popcll(a0 & a1);
        p02 += __popcll(a0 & a2);
        p03 += __popcll(a0 & a3);
        p12 += __popcll(a1 & a2);
        p13 += __popcll(a1 & a3);
        p23 += __popcll(a2 & a3);
    }

    // norm^2 = 1*s0 + 4*s1 + 16*s2 + 64*s3
    //        + 4*p01 + 8*p02 + 16*p03 + 16*p12 + 32*p13 + 64*p23
    int n2 =
          (1  * s0) + (4  * s1) + (16 * s2) + (64 * s3)
        + (4  * p01) + (8  * p02) + (16 * p03)
        + (16 * p12) + (32 * p13) + (64 * p23);

    float f2 = (float)n2;
    norm_l2_squared[v] = f2;
    norm_l2[v]         = sqrtf(f2);
}

// ===============================
// 4-bit bit-sliced Top-K kernel
// ===============================
extern "C" __global__ void find_topk_neighbors_u4_bitsliced_kernel(
    const unsigned long long* __restrict__ q0,   // [M][W]
    const unsigned long long* __restrict__ q1,   // [M][W]
    const unsigned long long* __restrict__ q2,   // [M][W]
    const unsigned long long* __restrict__ q3,   // [M][W]
    const unsigned long long* __restrict__ x0,   // [N][W]
    const unsigned long long* __restrict__ x1,   // [N][W]
    const unsigned long long* __restrict__ x2,   // [N][W]
    const unsigned long long* __restrict__ x3,   // [N][W]
    const float*   __restrict__ norm_l2,               // [N] (bin-space)
    const float*   __restrict__ norm_l2_squared,       // [N] (bin-space)
    int*           __restrict__ topk_indices,          // [M*K]
    float*         __restrict__ topk_distances,        // [M*K]
    const int K,
    const float max_distance,
    const int   vector_database_len,   // N
    const int   query_vectors_len,     // M
    const int   vector_size,           // D
    const float precomputed_threshold,
    const float* __restrict__ query_norm_l2,           // [M] (bin-space)
    const float* __restrict__ query_norm_l2_squared,   // [M] (bin-space)
    const int   W                                         // words per plane
)
{
    int q = blockIdx.x;
    if (q >= query_vectors_len) return;
    if (K > KMAX) return;

    // shared: per-thread heaps + query planes
    extern __shared__ unsigned char smem[];
    int*   sm_idx  = (int*)smem;
    float* sm_dist = (float*)(sm_idx + blockDim.x * K);
    unsigned long long* sm_q0 = (unsigned long long*)(sm_dist + blockDim.x * K);
    unsigned long long* sm_q1 = sm_q0 + W;
    unsigned long long* sm_q2 = sm_q1 + W;
    unsigned long long* sm_q3 = sm_q2 + W;

    __shared__ float norm_threshold;
    __shared__ float query_norm, query_norm_sq;
    __shared__ unsigned long long tail_mask;

    if (threadIdx.x == 0) {
        norm_threshold = precomputed_threshold;
        query_norm_sq  = query_norm_l2_squared[q];
        query_norm     = query_norm_l2[q];
        int tail_bits  = vector_size & 63;
        tail_mask = (tail_bits == 0) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << tail_bits) - 1ULL);
    }
    __syncthreads();

    // load query bitplanes into shared
    const unsigned long long* Q0 = q0 + (size_t)q * W;
    const unsigned long long* Q1 = q1 + (size_t)q * W;
    const unsigned long long* Q2 = q2 + (size_t)q * W;
    const unsigned long long* Q3 = q3 + (size_t)q * W;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        unsigned long long m = (w == W-1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;
        sm_q0[w] = Q0[w] & m;
        sm_q1[w] = Q1[w] & m;
        sm_q2[w] = Q2[w] & m;
        sm_q3[w] = Q3[w] & m;
    }
    __syncthreads();

    // thread-local top-K
    float tk_dist[KMAX];
    int   tk_idx[KMAX];
    #pragma unroll
    for (int t = 0; t < K; ++t) { tk_dist[t] = CUDART_INF_F; tk_idx[t] = -1; }

    // scan DB rows owned by this thread
    for (int i = threadIdx.x; i < vector_database_len; i += blockDim.x) {
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const unsigned long long* X0 = x0 + (size_t)i * W;
        const unsigned long long* X1 = x1 + (size_t)i * W;
        const unsigned long long* X2 = x2 + (size_t)i * W;
        const unsigned long long* X3 = x3 + (size_t)i * W;

        int c00=0,c01=0,c02=0,c03=0,
            c10=0,c11=0,c12=0,c13=0,
            c20=0,c21=0,c22=0,c23=0,
            c30=0,c31=0,c32=0,c33=0;

        #pragma unroll
        for (int w = 0; w < W; ++w) {
            unsigned long long m = (w == W-1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;

            unsigned long long q0w = sm_q0[w];
            unsigned long long q1w = sm_q1[w];
            unsigned long long q2w = sm_q2[w];
            unsigned long long q3w = sm_q3[w];

            unsigned long long x0w = X0[w] & m;
            unsigned long long x1w = X1[w] & m;
            unsigned long long x2w = X2[w] & m;
            unsigned long long x3w = X3[w] & m;

            c00 += __popcll(q0w & x0w);
            c01 += __popcll(q0w & x1w);
            c02 += __popcll(q0w & x2w);
            c03 += __popcll(q0w & x3w);

            c10 += __popcll(q1w & x0w);
            c11 += __popcll(q1w & x1w);
            c12 += __popcll(q1w & x2w);
            c13 += __popcll(q1w & x3w);

            c20 += __popcll(q2w & x0w);
            c21 += __popcll(q2w & x1w);
            c22 += __popcll(q2w & x2w);
            c23 += __popcll(q2w & x3w);

            c30 += __popcll(q3w & x0w);
            c31 += __popcll(q3w & x1w);
            c32 += __popcll(q3w & x2w);
            c33 += __popcll(q3w & x3w);
        }

        // dot = Σ_{i=0..3} Σ_{j=0..3} 2^(i+j) * cij
        int dot_i =
              (1  * c00)
            + (2  * (c01 + c10))
            + (4  * (c02 + c20 + c11))
            + (8  * (c03 + c30 + c12 + c21))
            + (16 * (c13 + c31 + c22))
            + (32 * (c23 + c32))
            + (64 *  c33);

        float dot = (float)dot_i;

        float d2 = query_norm_sq + norm_l2_squared[i] - 2.0f * dot;
        d2 = fmaxf(d2, 0.0f);
        if (max_distance <= 0.0f || d2 <= max_distance) {
            topk_try_insert(d2, i, tk_dist, tk_idx, K);
        }
    }

    // spill & merge per-thread candidates
    int base = threadIdx.x * K;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        sm_idx [base + t] = tk_idx[t];
        sm_dist[base + t] = tk_dist[t];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_d[KMAX];
        int   best_i[KMAX];
        #pragma unroll
        for (int t = 0; t < K; ++t) { best_d[t] = CUDART_INF_F; best_i[t] = -1; }

        int Nspill = blockDim.x * K;
        for (int n = 0; n < Nspill; ++n) {
            float d = sm_dist[n];
            int   i = sm_idx[n];
            if (i >= 0 && isfinite(d)) topk_try_insert(d, i, best_d, best_i, K);
        }
        for (int a = 0; a < K-1; ++a)
            for (int b = a+1; b < K; ++b)
                if (best_d[b] < best_d[a]) {
                    float td=best_d[a]; best_d[a]=best_d[b]; best_d[b]=td;
                    int   ti=best_i[a]; best_i[a]=best_i[b]; best_i[b]=ti;
                }

        int out = q * K;
        for (int t = 0; t < K; ++t) {
            topk_indices  [out + t] = best_i[t];
            topk_distances[out + t] = best_d[t];
        }
    }
}




