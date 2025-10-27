#!/bin/sh
set -x
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export KMP_INIT_AT_FORK=FALSE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=python3

exp_name=$1
config=$2
dataset=$3
stage=$4

if [ "${stage}" = "s1" ]; then
    echo "TRAINING STAGE 1" 
    TRAIN_CODE=train_joint_data_vq_bs.py
    echo "Training for Discrete Motion Prior"
elif [ "${stage}" = "s2interactive" ]; then
    echo "TRAINING STAGE 2 INTERACTIVE MODE" 
    TRAIN_CODE=train_artalk_interactive.py
    echo "Training for Speech-Driven Motion Synthesis (Interactive)"
else
    echo "TRAINING STAGE 2" 
    TRAIN_CODE=train_moods_pred_bs.py
    #TEST_CODE=test_artalk_pred.py
    echo "Training for Speech-Driven Motion Synthesis"
fi


exp_dir=logs/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${exp_dir}/result

export PYTHONPATH=./
echo $OMP_NUM_THREADS | tee -a ${exp_dir}/train-$now.log
nvidia-smi | tee -a ${exp_dir}/train-$now.log
which pip | tee -a ${exp_dir}/train-$now.log


## TRAIN
$PYTHON -u main/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log

## TEST
#$PYTHON -u main/${TEST_CODE} \
#  --config=${config} \
#  save_folder ${exp_dir}/result \
#  model_path ${model_dir}/model.pth.tar \
#  2>&1 | tee -a ${exp_dir}/test-$now.log