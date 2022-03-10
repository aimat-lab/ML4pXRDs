#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

rm -f ./worker_1_ready
rm -f ./worker_2_ready
rm -f ./head_node_ip

out_dir="classifier_spgs/$(date +%d-%m-%Y_%H-%M-%S)"

mkdir -p "../$out_dir"
mkdir -p "../$out_dir/ray_log"

sbatch -o "../$out_dir/slurm_head.out" submit_head.sh "$out_dir"
sbatch -o "../$out_dir/slurm_worker_1.out" submit_worker_1.sh "$out_dir"
sbatch -o "../$out_dir/slurm_worker_2.out" submit_worker_2.sh "$out_dir"