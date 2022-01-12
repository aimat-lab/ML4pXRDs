#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=80000mb
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_head
#SBATCH --time=10-48:00:00

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

port=6379
head_node_ip=$(hostname --ip-address)

ip_head=$head_node_ip:$port
export ip_head # pass along
echo "Starting HEAD at $ip_head"

echo $head_node_ip > ./random_simulation_com

srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password='5241590000000000' --num-cpus "14" --num-gpus "1" --block &

cd ..

sleep 60

python train_random_classifier.py
