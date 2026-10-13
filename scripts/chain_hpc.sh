#!/bin/bash
cd ~/ai-days-2026/run
KEEP_HF=1 bash run_hpc.sh 01_cnn 03_shortcuts_and_controls > run_7.log 2>&1
bash src_nb/run_student_hpc.sh > run_student.log 2>&1
echo ALLDONE >> run_student.log
