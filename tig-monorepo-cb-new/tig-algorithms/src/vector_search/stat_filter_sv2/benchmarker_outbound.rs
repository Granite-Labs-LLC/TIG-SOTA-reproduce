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

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

// when launching kernels, you should not exceed this const or else it may not be deterministic
//const MAX_THREADS_PER_BLOCK: u32 = 1024;

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

//use crate::{seeded_hasher, HashMap, HashSet};

/*!
Copyright 2025 Granite Labs LLC
...
*/

use std::sync::Arc;
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream, LaunchConfig},
    runtime::sys::cudaDeviceProp,
};
use cudarc::driver::PushKernelArg;

use tig_challenges::vector_search::*;
use std::env;

const MAD_SCALE_NORMAL: f32 = 1.4826;
const MAX_THREADS_PER_BLOCK: u32 = 1024;


/// Compile-time K for Top-K retrieval (must be <= kernel KMAX)
//pub const TOPK: usize = 1;
//pub const TOPK: usize = 10;
pub const TOPK: usize = 20;
//pub const TOPK: usize = 32;
//pub const TOPK: usize = 100;

// Each block works on a different query

pub fn solve_challenge(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>> {

    println!("Searching {} DB vectors of length {} for {} queries",challenge.database_size,challenge.vector_dims,challenge.difficulty.num_queries);

    let start_time_total = std::time::Instant::now();

    // Get top-k value to use
    let mut topk = read_topk();
    topk = topk.min(challenge.vector_dims as usize); // keep it relative to vector length
    topk = topk.min(TOPK); // no larger than constant max

    // Allocations for dimension statistics
    let d_db_dim_min = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;
    let d_db_dim_max = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;
    let d_s          = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;

    // Allocations for norms
    let d_db_norm_l2           = stream.alloc_zeros::<f32>(challenge.database_size as usize)?;
    let d_db_norm_l2_squared   = stream.alloc_zeros::<f32>(challenge.database_size as usize)?;
    let d_query_norm_l2        = stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;
    let d_query_norm_l2_squared= stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;

    // Allocation for conversion
    let num_db_el = challenge.database_size * challenge.vector_dims;
    let num_qv_el = challenge.difficulty.num_queries * challenge.vector_dims;

    // Allocate conversion buffers
    let mut d_db_f32 = stream.alloc_zeros::<f32>(num_db_el as usize)?;
    let mut d_qv_f32 = stream.alloc_zeros::<f32>(num_qv_el as usize)?;


    // ---------- 4-bit pack buffers ---------- 

    // Bytes per row when packing 2 dims per byte
    let row_bytes_u4    = ((challenge.vector_dims as usize) + 1) >> 1;
    let num_db_bytes_u4 = (challenge.database_size as usize) * row_bytes_u4;
    let num_qv_bytes_u4 = (challenge.difficulty.num_queries as usize) * row_bytes_u4;

    // Allocate packed outputs
    let mut d_db_u4 = stream.alloc_zeros::<u8>(num_db_bytes_u4)?;
    let mut d_qv_u4 = stream.alloc_zeros::<u8>(num_qv_bytes_u4)?;



    // Take the min of the mins of the dimensions and compare to zero
    // If >= 0, then proceed as normal.  If <0 then shift all the data by that min.


    let dataset_min = -1.0;   // DEBUG TEST

    let mut use_converted = false;
    // Check to see if the minimum is less than zero
    if dataset_min < 0.0 {
        //
        // ---------- Convert input data ---------- 
        //
        use_converted = true;

        let scale_fp32_m1p1_to_positive = module.load_function("scale_fp32_m1p1_to_positive")?;

        let threads_db: u32 = 256;
        let blocks_db:  u32 = ((num_db_el as u32) + threads_db - 1) / threads_db;

        let threads_qv: u32 = 256;
        let blocks_qv:  u32 = ((num_qv_el as u32) + threads_qv - 1) / threads_qv;

        // DB
        let cfg_db = LaunchConfig { grid_dim: (blocks_db, 1, 1), block_dim: (threads_db, 1, 1), shared_mem_bytes: 0 };
    
        unsafe {
            stream
                .launch_builder(&scale_fp32_m1p1_to_positive)
                .arg(&challenge.d_database_vectors)
                .arg(&d_db_f32)
                .arg(&num_db_el)
                .launch(cfg_db)?;
        }
    
        // Queries
        let cfg_qv = LaunchConfig { grid_dim: (blocks_qv, 1, 1), block_dim: (threads_qv, 1, 1), shared_mem_bytes: 0 };
    
        unsafe {
            stream
                .launch_builder(&scale_fp32_m1p1_to_positive)
                .arg(&challenge.d_query_vectors)
                .arg(&d_qv_f32)
                .arg(&num_qv_el)
                .launch(cfg_qv)?;
        }

        stream.synchronize()?;


    }

    // Set the pointer to the DB and query buffers based on whether we converted the data or not

    let d_db_f32_ptr = if use_converted { &d_db_f32 } else { &challenge.d_database_vectors };
    let d_qv_f32_ptr = if use_converted { &d_qv_f32 } else { &challenge.d_query_vectors };


    //
    // ---------- Compute Dimensional Stats ---------- 
    //

    let compute_dim_stats_kernel = module.load_function("compute_dim_stats_kernel")?;

    let threads_db: u32 = 256;
    let blocks_db:  u32 = ((num_db_el as u32) + threads_db - 1) / threads_db;

    let threads_qv: u32 = 256;
    let blocks_qv:  u32 = ((num_qv_el as u32) + threads_qv - 1) / threads_qv;

    // Launch compute dim stats kernel
    let cfg_db_ds = LaunchConfig {
        grid_dim: (blocks_db, 1, 1),
        block_dim: (threads_db, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&compute_dim_stats_kernel)
            .arg(d_db_f32_ptr)
            .arg(&d_db_dim_max)
            .arg(&challenge.database_size)
            .arg(&challenge.vector_dims)
            .launch(cfg_db_ds)?;
    }

    stream.synchronize()?;


    // Calculate the per-dim divisors based on max

    let build_u4_divisors_from_max_kernel = module.load_function("build_u4_divisors_from_max_kernel")?;

    let cfg_db_dm = LaunchConfig {
        grid_dim: (blocks_db, 1, 1),
        block_dim: (threads_db, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&build_u4_divisors_from_max_kernel)
            .arg(&d_db_dim_max)
            .arg(&d_s)
            .arg(&challenge.vector_dims)
            .launch(cfg_db_dm)?;
    }

    stream.synchronize()?;


    //
    // ---------- Convert input data by packing into bits ---------- 
    //

    // ---------- 4-bit pack ---------- 
    let f32_to_u4_packed_perdim_kernel = module.load_function("f32_to_u4_packed_perdim_kernel")?;

    // DB
    let threads_db: u32 = 256;
    let blocks_db:  u32 = ((num_db_bytes_u4 as u32) + threads_db - 1) / threads_db;
    let cfg_db = LaunchConfig { grid_dim: (blocks_db, 1, 1), block_dim: (threads_db, 1, 1), shared_mem_bytes: 0 };
    
    unsafe {
        stream
            .launch_builder(&f32_to_u4_packed_perdim_kernel)
            .arg(d_db_f32_ptr)                   // const float* in   [num_db * D]
            .arg(&d_s)                           // const float* s    [D]
            .arg(&d_db_u4)                       // uint8_t* out      [num_db * ((D+1)>>1)]
            .arg(&challenge.database_size)       // num_vecs
            .arg(&challenge.vector_dims)         // dims
            .launch(cfg_db)?;
    }

    // Queries
    let threads_qv: u32 = 256;
    let blocks_qv:  u32 = ((num_qv_bytes_u4 as u32) + threads_qv - 1) / threads_qv;
    let cfg_qv = LaunchConfig { grid_dim: (blocks_qv, 1, 1), block_dim: (threads_qv, 1, 1), shared_mem_bytes: 0 };

    unsafe {
        stream
            .launch_builder(&f32_to_u4_packed_perdim_kernel)
            .arg(d_qv_f32_ptr)                   // const float* in   [num_query * D]
            .arg(&d_s)
            .arg(&d_qv_u4)
            .arg(&challenge.difficulty.num_queries)
            .arg(&challenge.vector_dims)
            .launch(cfg_qv)?;
    }

    stream.synchronize()?;


    //
    // ---------- Compute Vector Stats ---------- 
    //

    // ---------- 4-bit pack ---------- 

    let compute_vector_stats_u4_packed_kernel = module.load_function("compute_vector_stats_u4_packed_kernel")?;

    let threads_per_block_stats = prop.maxThreadsPerBlock as u32;
    let num_blocks_db = (challenge.database_size + threads_per_block_stats - 1) / threads_per_block_stats;

    let cfg_stats = LaunchConfig {
        grid_dim: (num_blocks_db, 1, 1),
        block_dim: (threads_per_block_stats, 1, 1),
        shared_mem_bytes: 0
    };

    // DB norms
    unsafe {
        stream
            .launch_builder(&compute_vector_stats_u4_packed_kernel)
            .arg(&d_db_u4)                 // const uint8_t* packed [num_db * ((D+1)>>1)]
            .arg(&d_db_norm_l2)            // float* norm_l2        [num_db]
            .arg(&d_db_norm_l2_squared)    // float* norm_l2_sq     [num_db]
            .arg(&challenge.database_size) // num_vecs
            .arg(&challenge.vector_dims)   // dims
            .launch(cfg_stats)?;
    }

    // Query norms
    let num_blocks_qv = (challenge.difficulty.num_queries + threads_per_block_stats - 1) / threads_per_block_stats;

    let cfg_stats_qv = LaunchConfig {
        grid_dim: (num_blocks_qv, 1, 1),
        block_dim: (threads_per_block_stats, 1, 1),
        shared_mem_bytes: 0
    };

    unsafe {
        stream
            .launch_builder(&compute_vector_stats_u4_packed_kernel)
            .arg(&d_qv_u4)
            .arg(&d_query_norm_l2)
            .arg(&d_query_norm_l2_squared)
            .arg(&challenge.difficulty.num_queries)
            .arg(&challenge.vector_dims)
            .launch(cfg_stats_qv)?;
    }

    stream.synchronize()?;


    let elapsed_time_ms_1 = start_time_total.elapsed().as_micros() as f32 / 1000.0;


    //
    // ---------- Compute MAD Stats ---------- 
    //

    let mut norm_threshold: f32 = f32::MAX;   
    let scale = env::var("STATFILT_MAD_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or_else(|| scale_factor(challenge.difficulty.num_queries as usize));
    println!("stat_filter scale: {}", scale);

    // Only compute and apply MAD if within range
    if scale > 0.0 && scale < 5.0 {

        // MAD threshold on DB norms (unchanged logic)
        let mut h_norms = stream.memcpy_dtov(&d_db_norm_l2)?;
        h_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = h_norms.len() / 2;
        let median = if h_norms.len() % 2 == 0 {
            (h_norms[mid - 1] + h_norms[mid]) / 2.0
        } else {
            h_norms[mid]
        };

        let mut deviations: Vec<f32> = h_norms.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if deviations.len() % 2 == 0 {
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[mid]
        };

        norm_threshold = scale * mad * MAD_SCALE_NORMAL;
    }


    let elapsed_time_ms_2 = start_time_total.elapsed().as_micros() as f32 / 1000.0;


    //
    // ---------- Search ---------- 
    //

    // --- TopK outputs ---
    let mut d_topk_indices = stream.alloc_zeros::<i32>((challenge.difficulty.num_queries as usize) * topk)?;
    let mut d_topk_dist    = stream.alloc_zeros::<f32>((challenge.difficulty.num_queries as usize) * topk)?;

    // --- Geometry ---
    let dims = challenge.vector_dims as usize;             // D
    let n_db = challenge.database_size as usize;           // N
    let n_q  = challenge.difficulty.num_queries as usize;  // M
    let words_per_plane = ((dims + 63) >> 6) as usize;     // W
    let words_per_plane_i32: i32 = words_per_plane as i32;

    // --- Allocate 4 bitplanes for DB and Q (each plane: [count * W] u64) ---
    let mut d_db_b0 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
    let mut d_db_b1 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
    let mut d_db_b2 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
    let mut d_db_b3 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
    let mut d_q_b0  = stream.alloc_zeros::<u64>(n_q  * words_per_plane)?;
    let mut d_q_b1  = stream.alloc_zeros::<u64>(n_q  * words_per_plane)?;
    let mut d_q_b2  = stream.alloc_zeros::<u64>(n_q  * words_per_plane)?;
    let mut d_q_b3  = stream.alloc_zeros::<u64>(n_q  * words_per_plane)?;

    // --- Convert packed-u4 (2 dims/byte) -> 4 bitplanes ---
    let u4_packed_to_bitplanes = module.load_function("u4_packed_to_bitplanes")?;
    let blk_conv = (256u32, 1u32, 1u32);
    let grd_db   = (((n_db as u32) + 255) / 256, 1, 1);
    let grd_q    = (((n_q  as u32) + 255) / 256, 1, 1);

    unsafe {
        // DB
        stream
            .launch_builder(&u4_packed_to_bitplanes)
            .arg(&d_db_u4)                     // packed [N][(D+1)>>1]
            .arg(&mut d_db_b0) .arg(&mut d_db_b1)
            .arg(&mut d_db_b2) .arg(&mut d_db_b3)
            .arg(&(challenge.database_size))   // N
            .arg(&(challenge.vector_dims))     // D
            .arg(&words_per_plane_i32)         // W
            .launch(LaunchConfig { grid_dim: grd_db, block_dim: blk_conv, shared_mem_bytes: 0 })?;

        // Queries
        stream
            .launch_builder(&u4_packed_to_bitplanes)
            .arg(&d_qv_u4)                     // packed [M][(D+1)>>1]
            .arg(&mut d_q_b0) .arg(&mut d_q_b1)
            .arg(&mut d_q_b2) .arg(&mut d_q_b3)
            .arg(&(challenge.difficulty.num_queries)) // M
            .arg(&(challenge.vector_dims))            // D
            .arg(&words_per_plane_i32)                // W
            .launch(LaunchConfig { grid_dim: grd_q, block_dim: blk_conv, shared_mem_bytes: 0 })?;
    }

    stream.synchronize()?;


    let elapsed_time_ms_3 = start_time_total.elapsed().as_micros() as f32 / 1000.0;


    // --- Shared memory sizing for Top-K ---

    // Per-thread spill for heap:
    let per_thread_bytes = topk * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
    // 4 planes * W words * 8B per word
    let base_query_bytes = 4 * words_per_plane * std::mem::size_of::<u64>();

    let smem_limit = prop.sharedMemPerBlock as usize;
    let mut threads_per_block: usize = 256;
    while base_query_bytes + threads_per_block * per_thread_bytes > smem_limit && threads_per_block > 32 {
        threads_per_block >>= 1;
    }
    if base_query_bytes + threads_per_block * per_thread_bytes > smem_limit {
        return Err(anyhow!(
            "Insufficient shared memory for topk={} with dims={} (need ~{}B, have {}B)",
            topk, challenge.vector_dims,
            base_query_bytes + threads_per_block * per_thread_bytes, smem_limit
        ));
    }
    let threads_per_block = threads_per_block as u32;

    let shared_mem_bytes = (base_query_bytes + (threads_per_block as usize) * per_thread_bytes) as u32;

    let cfg_topk = LaunchConfig {
        grid_dim: (challenge.difficulty.num_queries, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: shared_mem_bytes,
    };

    // --- Launch 4-bit bit-sliced Top-K ---

    let k_i32: i32 = topk as i32;
    let find_topk_neighbors_u4_bitsliced_kernel = module.load_function("find_topk_neighbors_u4_bitsliced_kernel")?;

    unsafe {
        stream
            .launch_builder(&find_topk_neighbors_u4_bitsliced_kernel)
            .arg(&d_q_b0).arg(&d_q_b1).arg(&d_q_b2).arg(&d_q_b3)   // query planes
            .arg(&d_db_b0).arg(&d_db_b1).arg(&d_db_b2).arg(&d_db_b3) // db planes
            .arg(&d_db_norm_l2)           // bin-space norms (sqrt) [N]
            .arg(&d_db_norm_l2_squared)   // bin-space norms^2      [N]
            .arg(&mut d_topk_indices)     // [M*K]
            .arg(&mut d_topk_dist)        // [M*K]
            .arg(&k_i32)
            .arg(&challenge.max_distance)
            .arg(&challenge.database_size)            // N
            .arg(&challenge.difficulty.num_queries)   // M
            .arg(&challenge.vector_dims)              // D
            .arg(&norm_threshold)
            .arg(&d_query_norm_l2)         // bin-space query norms (sqrt) [M]
            .arg(&d_query_norm_l2_squared) // bin-space query norms^2      [M]
            .arg(&words_per_plane_i32)     // W
            .launch(cfg_topk)?;
    }

    stream.synchronize()?;

    // Pull back top-K indices, build Top-1 for the Solution, and compute Recall@K if provided
    let h_topk: Vec<i32> = stream.memcpy_dtov(&d_topk_indices)?;
    let mut top1 = Vec::<usize>::with_capacity(challenge.difficulty.num_queries as usize);
    for q in 0..(challenge.difficulty.num_queries as usize) {
        let base = q * topk;
        top1.push(h_topk[base] as usize); // assuming kernel writes sorted asc by distance
    }


    let elapsed_time_ms_4 = start_time_total.elapsed().as_micros() as f32 / 1000.0;


    //
    // === Re-rank Top-K on FP32 ===
    //
    // NOTE: We only return the best match, not an array.  This is an "internal" top-k.
    //

    let refine_fn = module.load_function("refine_topk_rerank_kernel")?;

    let threads_refine: u32 = 128;
    let grid_refine = challenge.difficulty.num_queries;
    let shared_refine = (challenge.vector_dims as usize * std::mem::size_of::<f32>()
                        + threads_refine as usize * std::mem::size_of::<f32>()) as u32;

    let mut d_refined_index    = stream.alloc_zeros::<i32>(challenge.difficulty.num_queries as usize)?;
    let mut d_refined_distance = stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;
    let k_i32: i32 = topk as i32;

    let cfg_refine = LaunchConfig {
        grid_dim: (grid_refine, 1, 1),
        block_dim: (threads_refine, 1, 1),
        shared_mem_bytes: shared_refine,
    };

    unsafe {
        stream
            .launch_builder(&refine_fn)
            .arg(&challenge.d_query_vectors)        // Original FP32 queries
            .arg(&challenge.d_database_vectors)     // Original FP32 DB
            .arg(&d_topk_indices)                   // [num_queries * K] (i32)
            .arg(&mut d_refined_index)              // OUT best index per query
            .arg(&mut d_refined_distance)           // OUT best distance per query
            .arg(&challenge.difficulty.num_queries) // num_queries
            .arg(&challenge.vector_dims)            // original vector dim
            .arg(&k_i32)                            // K
            .launch(cfg_refine)?;
    }
    stream.synchronize()?;

    // Use refined Top-1 as the final Solution
    let top1_refined: Vec<i32> = stream.memcpy_dtov(&d_refined_index)?;
    let mut final_idxs = Vec::<usize>::with_capacity(top1_refined.len());
    for &idx in &top1_refined {
        final_idxs.push(idx as usize);
    }


    let elapsed_time_ms = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    // Internal timing statistics

    println!("===== stat_filter bitslice 4-bit ( Top-{} ) =====", topk);
    println!(
        "Time for nonce: {:.3} ms (sum+stats: {:.3} ms + mad_sort: {:.3} ms + slice: {:.3} ms + search: {:.3} ms + rerank {:.3} ms)",
        elapsed_time_ms,
        elapsed_time_ms_1,
        elapsed_time_ms_2 - elapsed_time_ms_1,
        elapsed_time_ms_3 - elapsed_time_ms_2,
        elapsed_time_ms_4 - elapsed_time_ms_3,
        elapsed_time_ms - elapsed_time_ms_4
    );


    Ok(Some(Solution { indexes: final_idxs }))
}

//------------ MAD Scale Factor Adjustment -------------

fn scale_factor(num_queries: usize) -> f32 {
    match num_queries {
        q if q <= 700 => 0.20,
        q if q <= 1000 => 0.20 + (q as f32 - 700.0) * (0.10 / 300.0),       // 0.30 at 1000
        q if q <= 1500 => 0.30 + (q as f32 - 1000.0) * (0.20 / 500.0),      // 0.50 at 1500
        q if q <= 2000 => 0.50 + (q as f32 - 1500.0) * (0.44 / 500.0),      // 0.94 at 2000
        q if q <= 2500 => 0.94 + (q as f32 - 2000.0) * (1.08 / 500.0),      // 2.02 at 2500
        _ => 1.00,
    }
}

//----------------- 4-bit conversion -------------------

fn read_topk() -> usize {
    env::var("STATFILT_TOP_K")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(TOPK)
}

