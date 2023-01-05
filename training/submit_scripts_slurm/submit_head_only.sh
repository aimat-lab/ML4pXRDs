#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128000mb
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=random_head
#SBATCH --time=10-48:00:00

source ~/.bashrc
conda activate pyxtal

port=6379
head_node_ip=$(hostname --ip-address)

ip_head=$head_node_ip:$port
export ip_head # pass along
echo "Starting HEAD at $ip_head"

date_time="$(date +%d-%m-%Y_%H-%M-%S)"
out_dir="classifier_spgs/${date_time}"
mkdir -p "../$out_dir"

srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password='5241590000000000' --num-cpus "30" --num-gpus "2" --block &

cd ..

python train_classifier.py $date_time head_only