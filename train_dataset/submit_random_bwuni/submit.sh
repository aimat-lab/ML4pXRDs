#!/bin/bash

export TMPDIR=/home/kit/iti/la2559/ray_tmp/

source ~/mambaforge/etc/profile.d/conda.sh
conda activate tf

rm -f ./worker_*_ready
rm -f ./head_node_ip

date_time="$(date +%d-%m-%Y_%H-%M-%S)"
out_dir="classifier_spgs/${date_time}"

mkdir -p "../$out_dir"
#mkdir -p "../$out_dir/ray_log"

sbatch -o "../$out_dir/slurm_head.out" submit_head.sh "$date_time" {0..3}

for i in {0..3}
do
    sbatch -o "../$out_dir/slurm_worker_$i.out" submit_worker.sh "$i"
done