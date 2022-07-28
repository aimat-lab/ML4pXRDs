#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=128000mb
#SBATCH --tasks-per-node=1
#SBATCH --partition=gpu_8
#SBATCH --job-name=random_head
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

export TMPDIR=/home/kit/iti/la2559/ray_tmp/

source ~/mambaforge/etc/profile.d/conda.sh
conda activate tf

port=6379
head_node_ip=$(hostname --ip-address)

ip_head=$head_node_ip:$port
export ip_head # pass along
echo "Starting HEAD at $ip_head"

#srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password='5241590000000000' --temp-dir "../$1/ray_log" --num-cpus "14" --num-gpus "1" --block &
srun --nodes=1 --ntasks=1 ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password='5241590000000000' --num-cpus "4" --num-gpus "1" --temp-dir "/home/kit/iti/la2559/ray_tmp/" --block &

sleep 20
echo $head_node_ip > ./head_node_ip # signal that worker can connect

for ((i=2; i<=$#; i++))
do
    # wait until worker has connected
    until [ -f "./worker_${!i}_ready" ]
    do
            sleep 5
    done
done

cd ..

python train_random_classifier.py $1 "both" 40