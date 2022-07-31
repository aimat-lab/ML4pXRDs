#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal

rm -f ./worker_1_ready
rm -f ./head_node_ip

date_time="$(date +%d-%m-%Y_%H-%M-%S)"
out_dir="unet/${date_time}"

mkdir -p "../$out_dir"

sbatch -o "../$out_dir/slurm_head.out" submit_head.sh "$date_time"
sbatch -o "../$out_dir/slurm_worker_1.out" submit_worker_1.sh