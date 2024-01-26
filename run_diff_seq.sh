#!/bin/bash

#SBATCH -o run_diff_seq_long.sh.log-%j-%a
#SBATCH -a 0-5
#SBATCH --exclusive
#SBATCH --time=28:00:00

module load anaconda/2023a-pytorch
source /home/gridsan/sjorgensen/.profile

which conda

source activate /home/gridsan/sjorgensen/.conda/envs/mariogp

java -cp "server/mario/*:server/libs/*" PlayPython &
pid=$!
python main.py configs/difficult_sequential_config.yaml $SLURM_ARRAY_TASK_ID
kill -9 $pid
