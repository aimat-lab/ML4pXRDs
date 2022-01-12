#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

rm -f ./worker_ready
rm -f ./head_node_ip

sbatch submit_head.sh
sbatch submit_worker.sh
