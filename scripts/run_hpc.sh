#!/bin/bash
# Execute every solution notebook on gold-srv2 (GPU1), in order, with a fresh HF cache so the
# download times are real. Produces the participants' outputs (my_cnn.pt, my_emb_mistral17m.npz, my_part*_results.json).
# Usage (on the HPC): bash run_hpc.sh [notebook stems...]
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aidays-2026
export CUDA_VISIBLE_DEVICES=1
RUN=~/ai-days-2026/run
mkdir -p $RUN/data
cd $RUN
for f in ~/ai-days-2026/data/CLIPSEQ_AGO2.*.fa; do ln -sf "$f" data/; done
export AIDAYS_DATA_DIR=$RUN/data
export HF_HOME=$RUN/hf_fresh
[ "${KEEP_HF:-0}" = "1" ] || rm -rf "$HF_HOME"
STEMS=${@:-01_cnn 02_foundation_models 03_shortcuts_and_controls}
for s in $STEMS; do
  echo "=== $s $(date +%T)"
  [ "$s" = "02_foundation_models" ] && export AIDAYS_RUN_NT500M=1 || unset AIDAYS_RUN_NT500M
  t=$(date +%s)
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
      --output "$RUN/${s}_solution.ipynb" "$RUN/src_nb/${s}_solution.ipynb" > "$RUN/$s.log" 2>&1
  echo "exit $? after $(( $(date +%s) - t ))s"
  tail -3 "$RUN/$s.log"
  # the reference files (cnn.pt, emb_*.npz) are fixed and hosted on Hugging Face; never overwrite them here
done
echo "=== done $(date +%T)"
