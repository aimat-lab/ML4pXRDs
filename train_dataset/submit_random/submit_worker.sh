#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem=20000mb
#SBATCH --tasks-per-node=1
#SBATCH --job-name=random_worker
#SBATCH --time=10-48:00:00

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

port=6379
read head_node_ip < ./random_simulation_com
ip_head=$head_node_ip:$port

echo "Starting worker"

srun --nodes=1 --ntasks=1 ray start --address "$ip_head" --redis-password='5241590000000000' --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "0" --block
