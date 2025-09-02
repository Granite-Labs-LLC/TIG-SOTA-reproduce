use std::panic::catch_unwind;
use tig_algorithms::vector_search::stat_filter;
use tig_challenges::vector_search::*;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;


#[cfg(not(feature = "cuda"))]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: &Challenge) -> Result<Option<Solution>, String>
{
    return catch_unwind(|| {
        stat_filter::solve_challenge(challenge).map_err(|e| e.to_string())
    }).unwrap_or_else(|_| {
        Err("Panic occurred calling solve_challenge".to_string())
    });
}


#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>, String>
{
    return catch_unwind(|| {
        stat_filter::solve_challenge(challenge, module, stream, prop).map_err(|e| e.to_string())
    }).unwrap_or_else(|_| {
        Err("Panic occurred calling solve_challenge".to_string())
    });
}