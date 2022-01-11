salloc --cpus-per-task=4 --mem-per-cpu=1  --ntasks=1 : --cpus-per-task=1 --mem-per-cpu=16 --ntasks=1 bash
srun --het-group=0 printenv SLURM_JOB_ID
srun --het-group=1 printenv SLURM_JOB_ID
