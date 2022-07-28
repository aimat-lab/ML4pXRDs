#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=160000mb
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_head
#SBATCH --time=30

source ~/mambaforge/etc/profile.d/conda.sh
conda activate tf

port=6379
head_node_ip=$(hostname --ip-address)

ip_head=$head_node_ip:$port
export ip_head # pass along
echo "Starting HEAD at $ip_head"

date_time="$(date +%d-%m-%Y_%H-%M-%S)"
out_dir="classifier_spgs/${date_time}"
mkdir -p "../$out_dir"

srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password='5241590000000000' --num-cpus "14" --num-gpus "1" --block &

cd ..

python train_random_classifier.py $date_time
