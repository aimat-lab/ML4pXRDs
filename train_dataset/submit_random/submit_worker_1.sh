#!/bin/bash
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1
#SBATCH --mem=256000mb
#SBATCH --tasks-per-node=1
#SBATCH --job-name=random_worker
#SBATCH --time=10-48:00:00

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

# wait until head is ready
until [ -f ./head_node_ip ]
do
         sleep 5
done

port=6379
read head_node_ip < ./head_node_ip
ip_head=$head_node_ip:$port

echo "Starting worker"

{ sleep 20; echo "ready" > ./worker_1_ready; } & # signal that worker is connected

#srun --nodes=1 --ntasks=1 ray start --address "$ip_head" --redis-password='5241590000000000' --temp-dir "/home/ws/uvgnh/ray_tmp" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "0" --block
srun --nodes=1 --ntasks=1 ray start --address "$ip_head" --redis-password='5241590000000000' --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "0" --block
