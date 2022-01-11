#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh
conda activate pyxtal_debug

rm -f ./random_simulation_com

sbatch submit_head.sh

# wait until file is found
until [ -f ./random_simulation_com ]
do
         sleep 5
done

read head_node_ip < ./random_simulation_com
echo "IP of head is $head_node_ip"

sleep 10

sbatch submit_worker.sh
