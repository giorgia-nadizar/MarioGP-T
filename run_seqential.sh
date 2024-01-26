#!/bin/bash

#SBATCH -o run_sequential_long.sh.log-%j-%a
#SBATCH -a 0-9
#SBATCH --exclusive
#SBATCH --time=28:00:00

module load anaconda/2023a-pytorch
source /home/gridsan/sjorgensen/.profile

which conda

source activate /home/gridsan/sjorgensen/.conda/envs/mariogp

java -cp "server/mario/*:server/libs/*" PlayPython &
pid=$!
python main.py configs/cv_sequential_config.yaml $SLURM_ARRAY_TASK_ID
kill -9 $pid
