#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1 
#SBATCH --mem=80000mb
#SBATCH --tasks-per-node=1 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=random_head
#SBATCH --time=10-48:00:00

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

my_mode=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$my_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head # pass along
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"

pipe=/tmp/random_simulation_com
if [[ ! -p $pipe ]]; then
    mkfifo $pipe
fi
echo $ip_head > $pipe 

srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &