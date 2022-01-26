#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

rm -f ./worker_ready
rm -f ./head_node_ip

out_dir="classifier_spgs/$(date +%d-%m-%Y_%H-%M-%S)"

mkdir -p "../$out_dir"

sbatch -o "../$out_dir/slurm_head.out" submit_head.sh "$out_dir"
sbatch -o "../$out_dir/slurm_worker.out" submit_worker.sh
