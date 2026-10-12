#!/bin/bash
# Execute the STUDENT notebooks (with `...` in every TODO) in a fresh folder that only has the
# shipped files (FASTAs + rejoin files, no my_* outputs). Checks that every given cell runs and that
# the rejoin fallbacks work. Errors are allowed; check_student.py reports where they happen.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aidays-2026
export CUDA_VISIBLE_DEVICES=1
RUN=~/ai-days-2026/run_student
rm -rf $RUN && mkdir -p $RUN/data && cd $RUN
for f in ~/ai-days-2026/data/CLIPSEQ_AGO2.*.fa ~/ai-days-2026/run/data/{cnn.pt,emb_mistral17m.npz,emb_nt500m.npz,slot6_results.json}; do ln -sf "$f" data/; done
export AIDAYS_DATA_DIR=$RUN/data HF_HOME=~/ai-days-2026/run/hf_fresh
for s in ${@:-02_build_cnn 03_look_inside 05_pretrained_model 06_break_it 07_fix_it}; do
  jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=1800 \
      --output "$RUN/$s.ipynb" ~/ai-days-2026/run/src_nb/$s.ipynb > $RUN/$s.log 2>&1
  echo "$s exit $?"
done
