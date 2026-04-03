#!/bin/bash
#SBATCH --time=0-04:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/output.%A.out
#SBATCH -e slurm_jobs/error.%A.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <molecule_name>" >&2
	exit 1
fi

MOLNAME="$1"
export SM_MOLNAME="${MOLNAME}"

# Echo all job information to stderr so it appears in the error log file
echo "Job started at: $(date)" >&2
echo "Working directory: $(pwd)" >&2
echo "Molecule: ${MOLNAME}" >&2
echo "Config file: ${SM_CONFIG_YML:-<auto-resolved by config.py>}" >&2
echo "Environment variables:" >&2
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-}" >&2
echo "  SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}" >&2
echo "  HOSTNAME: ${HOSTNAME:-}" >&2
echo "  SM_MOLNAME: ${SM_MOLNAME}" >&2
echo "  SM_CONFIG_YML: ${SM_CONFIG_YML:-}" >&2

nsteps=200000

python 1_gen_cg_topo.py
python 2_aa_md.py
python 3_boltz_inv.py
python 4_cg_md.py nsteps $nsteps # 10000 steps for 100 ps 100 samples
python 5_cgmd_upd.py
python 4_cg_md.py md nsteps $nsteps
python 5_cgmd_upd.py plot

echo "----------------------------------------" >&2
echo "Job completed at: $(date)" >&2
