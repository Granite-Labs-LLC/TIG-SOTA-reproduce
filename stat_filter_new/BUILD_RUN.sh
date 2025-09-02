#!/bin/sh

DATADIR=../data

if [ "$1" = "rebuild" ]; then
    rm -rf target
    ALGORITHM=stat_filter cargo build --release --features cudarc
fi

# build
if [ "$1" = "build" ]; then
    ALGORITHM=stat_filter cargo build --release --features cudarc
fi

# run
set -x
#target/release/vector_search_evaluator ${DATADIR}/SIFT
#target/release/vector_search_evaluator ${DATADIR}/Fashion-MNIST/

# Wide open

echo '============ NEW stat_filter: SIFT: No MAD, 2-bit, K=20 ============'
STATFILT_BIT_MODE=2 STATFILT_TOP_K=20 STATFILT_MAD_SCALE=5 target/release/vector_search_evaluator ${DATADIR}/SIFT

echo '============ NEW stat_filter: SIFT: No MAD, 4-bit, K=5 ============'
STATFILT_BIT_MODE=4 STATFILT_TOP_K=5 STATFILT_MAD_SCALE=5 target/release/vector_search_evaluator ${DATADIR}/SIFT

#STATFILT_TOP_K=4 STATFILT_MAD_SCALE=1.5 target/release/vector_search_evaluator ${DATADIR}/SIFT
#STATFILT_MAD_SCALE=1.0 target/release/vector_search_evaluator ${DATADIR}/SIFT
#STATFILT_MAD_SCALE=0.50 target/release/vector_search_evaluator ${DATADIR}/SIFT
#STATFILT_MAD_SCALE=0.25 target/release/vector_search_evaluator ${DATADIR}/SIFT
#STATFILT_MAD_SCALE=0.10 target/release/vector_search_evaluator ${DATADIR}/SIFT

echo '============ NEW stat_filter: Fasion-MNIST: MAD Scale 0.4, 2-bit, K=10 ============'
STATFILT_BIT_MODE=2 STATFILT_TOP_K=10 STATFILT_MAD_SCALE=0.4 target/release/vector_search_evaluator ${DATADIR}/Fashion-MNIST

echo '============ NEW stat_filter: Fasion-MNIST: No MAD, 4-bit, K=4 ============'
STATFILT_BIT_MODE=4 STATFILT_TOP_K=4 STATFILT_MAD_SCALE=0.4 target/release/vector_search_evaluator ${DATADIR}/Fashion-MNIST




