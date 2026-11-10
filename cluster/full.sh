#!/bin/bash
# Full experiment, one job per range of seeds. From the repository root:
#   mkdir -p logs
#   for s in 0 10 20 30 40 50 60 70 80 90; do sbatch cluster/full.sh $s $((s + 10)); done
# Resources match cluster/reduced.sh and the original AAMAS runs, so runtimes are comparable.
#SBATCH --job-name=full
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=amd7763

module load conda
conda activate course-allocation-aceei
export PYTHONHASHSEED=0  # fixed set/dict order, so reruns are reproducible

cd /work/pi_yzick_umass_edu/pnavarrete/course-allocation-data-aceei
echo "seeds $1 to $(($2 - 1)) on $(hostname)"
python -u scripts/exp_full.py "$1" "$2"
