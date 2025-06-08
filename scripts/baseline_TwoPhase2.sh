#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCS22_NJOBS=16  # 16 Cores at max

# Hyperparameters
repeat=1
SNRs=(-20 -10 0 10 20)
missing_link_ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)
spurious_link_ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)

# Parallelize
MAX_JOBS=1
function wait_for_jobs { while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 1; done }

# Initialize
script="python ./scripts/baseline_TwoPhase.py --data ./data/synthetic/KUR.json --vars x omega --target_var dx" 
basic_library="--library Polynomial Trigonometric Exponential Fractional CoupledPolynomial CoupledTrigonometric CoupledExponential CoupledFractional"

for repeat_index in $(seq 1 $repeat); do
    for SNR in "${SNRs[@]}"; do
        wait_for_jobs
        echo "Running SNR=${SNR} (basic)"
        $script $basic_library --name "Basic_Kuramoto_SNR${SNR}" --obs_noise_SNR $SNR &
    done

    for spurious_link_ratio in "${spurious_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running spurious_link_ratio=${spurious_link_ratio} (basic)"
        $script $basic_library --name "Basic_Kuramoto_spurious_link_ratio${spurious_link_ratio}" --spurious_link_ratio $spurious_link_ratio &
    done

    for missing_link_ratio in "${missing_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running missing_link_ratio=${missing_link_ratio} (basic)"
        $script $basic_library --name "Basic_Kuramoto_missing_link_ratio${missing_link_ratio}" --missing_link_ratio $missing_link_ratio &
    done

    for SNR in "${SNRs[@]}"; do
        wait_for_jobs
        echo "Running SNR=${SNR}"
        $script --name "Kuramoto_SNR${SNR}" --obs_noise_SNR $SNR &
    done

    for spurious_link_ratio in "${spurious_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running spurious_link_ratio=${spurious_link_ratio}"
        $script --name "Kuramoto_spurious_link_ratio${spurious_link_ratio}" --spurious_link_ratio $spurious_link_ratio &
    done

    for missing_link_ratio in "${missing_link_ratios[@]}"; do
        wait_for_jobs
        echo "Running missing_link_ratio=${missing_link_ratio}"
        $script --name "Kuramoto_missing_link_ratio${missing_link_ratio}" --missing_link_ratio $missing_link_ratio &
    done
done
