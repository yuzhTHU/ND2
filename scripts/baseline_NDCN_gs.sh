#!/bin/bash

# Parallize
MAX_JOBS=10
function wait_for_jobs { while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 1; done }
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Hyperparameters
datas=('ecological' 'gene')
lambdas=(1e-4 1e-3 1e-2)
weight_decays=(1e-8 1e-4 1e-2)
dropouts=(0.0 0.3 0.5 0.7)
num_layers=(2 3 4)
hidden_dims=("32" "64" "128")

# Initialize
mkdir -p log/grid_search_NDCN

# Grid Search
for data in "${datas[@]}"; do
for lambda in "${lambdas[@]}"; do
for w in "${weight_decays[@]}"; do
for p in "${dropouts[@]}"; do
for N in "${num_layers[@]}"; do
for D in "${hidden_dims[@]}"; do
    wait_for_jobs
    echo "Running data=$data, λ=$lambda, w=$w, p=$p, N=$N, D=$D"
    python ./scripts/baseline_NDCN.py \
        --data $data \
        --lr "$lambda" \
        --weight_decay "$w" \
        --dropout "$p" \
        --hidden_dim $(printf "%0.s$D " $(seq 1 $N)) \
        --data "$data" \
        > log/grid_search_NDCN/lr${lambda}_bs${B}_drop${p}_N${N}_D${D}_${data}.log 2>&1 &
done
done
done
done
done
done

# 等待所有后台任务完成
wait
echo "Grid search Down"
