#!/bin/sh

source /home/henrik/anaconda3/etc/profile.d/conda.sh
conda activate pyxtal_debug

for ((i=3; i<=$#; i++))
do
  python ./compare_random_distribution.py $1 $2 "${!i}"
done

python ./compare_random_distribution.py $1 $2 "${@:3}"