#!/bin/bash
# Reduced experiment (real students only), all 100 seeds in one job. From the repository root:
#   mkdir -p logs
#   sbatch cluster/reduced.sh
# Resources match cluster/full.sh and the original AAMAS runs, so runtimes are comparable.
#SBATCH --job-name=reduced
#SBATCH --output=logs/reduced_%j.out
#SBATCH --error=logs/reduced_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=amd7763

module load conda
conda activate course-allocation-aceei
export PYTHONHASHSEED=0  # fixed set/dict order, so reruns are reproducible

cd /work/pi_yzick_umass_edu/pnavarrete/course-allocation-data-aceei
echo "all seeds on $(hostname)"
python -u scripts/exp_reduced.py
