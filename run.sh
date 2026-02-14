#!/bin/bash
#SBATCH --time=0-02:00:00                                                       # upper bound time limit for job to finish d-hh:mm:ss
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/output.%A.out
#SBATCH -e slurm_jobs/error.%A.err

# Get the Python script (first argument)
PYSCRIPT="$1"
shift  # Remove the script name from arguments

has_gpus() {
	# Return success (0) iff this job appears to have GPUs allocated.
	# Prefer SLURM variables; fall back to CUDA_VISIBLE_DEVICES.
	# Notes:
	# - SLURM_* may be unset for non-SLURM runs.
	# - CUDA_VISIBLE_DEVICES can be "" or "NoDevFiles" when no GPUs are exposed.
	local v
	for v in "${SLURM_GPUS:-}" "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}" "${SLURM_GPUS_ON_NODE:-}"; do
		if [[ -n "$v" ]]; then
			return 0
		fi
	done
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
		return 0
	fi
	return 1
}

# Echo all job information to stderr so it appears in the error log file
echo "Job started at: $(date)" >&2
echo "Running script: $PYSCRIPT" >&2
echo "Working directory: $(pwd)" >&2
echo "Arguments: $@" >&2
echo "Environment variables:" >&2
echo "  SLURM_JOB_ID: $SLURM_JOB_ID" >&2
echo "  SLURM_JOB_NAME: $SLURM_JOB_NAME" >&2
echo "  HOSTNAME: $HOSTNAME" >&2

MPS_ENABLED=0
if has_gpus; then
	# Set up CUDA MPS
	TMPDIR="tmp/mps_${SLURM_JOB_ID:-$$}"
	export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR}/nvidia-mps"
	export CUDA_MPS_LOG_DIRECTORY="${TMPDIR}/nvidia-log"
	mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

	# Start MPS daemon (if available)
	if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
		nvidia-cuda-mps-control -d
		MPS_ENABLED=1
	else
		echo "Note: GPUs detected but nvidia-cuda-mps-control not found; skipping MPS." >&2
	fi
else
	echo "No GPUs detected for this job; skipping CUDA MPS setup." >&2
fi
echo "----------------------------------------" >&2

# Run the Python script
python 4_cg_md.py 

# Clean up MPS
if [[ "$MPS_ENABLED" -eq 1 ]]; then
	echo quit | nvidia-cuda-mps-control
fi

echo "----------------------------------------" >&2
echo "Job completed at: $(date)" >&2
