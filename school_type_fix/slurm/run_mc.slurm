#!/bin/bash
# ==============================================================================
# run_mc.slurm — Monte Carlo simulation: grade-inflation hyperparameter recovery
# ==============================================================================
# Submit with:  sbatch slurm/run_mc.slurm
# Monitor with: squeue -u keitaro2
# Output:       slurm/logs/mc_<jobid>.out
# Results:      output/results/mc_hyperparams.rds
# ==============================================================================

#SBATCH --job-name=mc_inflation
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/u/keitaro2/A-Levels/school_type_fix/slurm/logs/mc_%j.out
#SBATCH --error=/u/keitaro2/A-Levels/school_type_fix/slurm/logs/mc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=keitaro2@illinois.edu

# --- Environment ---
module load R/4.5.1

# Force single-threaded BLAS/OMP in each forked mclapply worker.
# Without this, forked children inherit a dead OpenBLAS/MKL thread pool
# and hang on the first crossprod/solve call (0% CPU, never complete).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "========================================"
echo " Monte Carlo: grade inflation hyperparams"
echo " Job ID : $SLURM_JOB_ID"
echo " Node   : $SLURMD_NODENAME"
echo " CPUs   : $SLURM_CPUS_PER_TASK"
echo " Started: $(date)"
echo "========================================"

cd /u/keitaro2/A-Levels/school_type_fix

Rscript Post_LASSO/R/03_simulation_runner.R \
  --reps  50 \
  --cores "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo " Finished: $(date)"
echo "========================================"
