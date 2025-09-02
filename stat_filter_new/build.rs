use std::{env, fs, process::Command};

fn main() {
    let algorithm = env::var("ALGORITHM").expect("ALGORITHM env variable not set");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR env variable not set");

    let cuda_code = format!(
        "{}\n\n{}",
        fs::read_to_string(&format!(
            "{}/src/vector_search.cu",
            tig_challenges::BUILD_TIME_PATH
        ))
        .expect("Failed to read cuda code for vector_search"),
        fs::read_to_string(&format!(
            "{}/src/vector_search/{}/benchmarker_outbound.cu",
            tig_algorithms::BUILD_TIME_PATH,
            algorithm
        ))
        .expect(&format!("Failed to read cuda code for {}", algorithm))
    );

    let cu_path = format!("{}/algo.cu", out_dir);
    let ptx_path = format!("{}/algo.ptx", out_dir);

    fs::write(&cu_path, cuda_code).expect("Failed to write combined_code.rs");

    // Compile the `.cu` file using nvcc
    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg(&cu_path)
        .arg("-o")
        .arg(&ptx_path)
        .arg("--use_fast_math")
        .arg("-dopt=on")
        .arg("--gpu-architecture=compute_89")
        .arg("--gpu-code=sm_89")
        .arg("-O3")
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed with status: {}", status);
    }

    println!("cargo:rerun-if-changed={}", algorithm);
}
