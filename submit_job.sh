#!/usr/bin/env bash
#SBATCH --job-name=xlsr_mamba_a3_asv5
#SBATCH --nodelist=dgx02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/home/user14/thuhb/baselines/XLSR-Mamba
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /home/user14/miniconda3/etc/profile.d/conda.sh
conda activate xlsr_mamba

DATA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof5
XLSR_PATH=${XLSR_MAMBA_XLSR_PATH:-/home/user14/thuhb/data_spoof/xlsr2_300m.pt}

test -f "${DATA_ROOT}/protocols/ASVspoof5.train.tsv"
test -d "${DATA_ROOT}/flac_T"
test -f "${XLSR_PATH}"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

srun python -u main.py \
  --dataset asvspoof5 \
  --database_path "${DATA_ROOT}" \
  --xlsr_path "${XLSR_PATH}" \
  --algo 3 \
  --batch_size 20 \
  --num_workers 8 \
  --num_epochs 7 \
  --max_epochs 75 \
  --eval_after_train false \
  --output_dir outputs/asvspoof5_algo3 \
  --comment asv5_algo3 \
  "${RESUME_ARGS[@]}"
