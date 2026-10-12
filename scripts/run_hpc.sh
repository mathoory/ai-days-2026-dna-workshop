#!/bin/bash
# Execute every solution notebook on gold-srv2 (GPU1), in order, with a fresh HF cache so the
# download times are real. Produces the rejoin files (cnn.pt, emb_*.npz, slot6_results.json).
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
STEMS=${@:-02_build_cnn 03_look_inside 05_pretrained_model 06_break_it 07_fix_it}
for s in $STEMS; do
  echo "=== $s $(date +%T)"
  [ "$s" = "05_pretrained_model" ] && export AIDAYS_RUN_NT500M=1 || unset AIDAYS_RUN_NT500M
  t=$(date +%s)
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
      --output "$RUN/${s}_solution.ipynb" "$RUN/src_nb/${s}_solution.ipynb" > "$RUN/$s.log" 2>&1
  echo "exit $? after $(( $(date +%s) - t ))s"
  tail -3 "$RUN/$s.log"
  case $s in
    02_build_cnn) cp data/my_cnn.pt data/cnn.pt ;;
    05_pretrained_model) cp data/my_emb_mistral17m.npz data/emb_mistral17m.npz ;;
    06_break_it) cp data/my_slot6_results.json data/slot6_results.json ;;
  esac
done
echo "=== done $(date +%T)"
