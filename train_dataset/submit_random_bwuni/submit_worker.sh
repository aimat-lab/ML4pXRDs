#!/bin/bash
#SBATCH --cpus-per-task=80
#SBATCH --ntasks=1
#SBATCH --mem=180000mb
#SBATCH --tasks-per-node=1
#SBATCH --job-name=random_worker
#SBATCH --time=48:00:00
#SBATCH --partition=single

source ~/mambaforge/etc/profile.d/conda.sh
conda activate tf

# wait until head is ready
until [ -f ./head_node_ip ]
do
         sleep 5
done

port=6379
read head_node_ip < ./head_node_ip
ip_head=$head_node_ip:$port

echo "Starting worker"

{ sleep 20; echo "ready" > "./worker_$1_ready"; } & # signal that worker is connected

#srun --nodes=1 --ntasks=1 ray start --address "$ip_head" --redis-password='5241590000000000' --temp-dir "../$1/ray_log" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "0" --block
srun --nodes=1 --ntasks=1 ray start --address "$ip_head" --redis-password='5241590000000000' --num-cpus "40" --num-gpus "0" --block
