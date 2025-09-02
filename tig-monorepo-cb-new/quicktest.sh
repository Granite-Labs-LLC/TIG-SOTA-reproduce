#!/bin/sh

set -x

export STATFILT_MAD_SCALE=0
#export STATFILT_TOP_K=20
export STATFILT_TOP_K=10
export STATFILT_BIT_MODE=4

#scripts/test_algorithm_timer --tig_runtime_path target/release/tig-runtime --nonces 1 --verbose --fuel 100000000000000  stat_filter [2000,500]

#scripts/test_algorithm_timer --nonces 10 --verbose stat_filter [2000,500]
#
scripts/test_algorithm_timer --nonces 1 --verbose stat_filter [2000,500]

#scripts/test_algorithm_timer --nonces 10 --verbose stat_filter [2000,350]

#scripts/test_algorithm_timer --nonces 10 --verbose stat_filter [200,50]

#scripts/test_algorithm_timer --nonces 1 --verbose stat_filter [200,50]

export STATFILT_MAD_SCALE=0
#export STATFILT_TOP_K=20
export STATFILT_TOP_K=10
export STATFILT_BIT_MODE=2
scripts/test_algorithm_timer --nonces 1 --verbose stat_filter [2000,500]
