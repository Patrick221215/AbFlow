#!/bin/bash
###
 # @Author: Patrick221215 1427584833@qq.com
 # @Date: 2026-06-13 19:33:31
 # @LastEditors: Patrick221215 1427584833@qq.com
 # @LastEditTime: 2026-06-20 19:50:24
 # @FilePath: /cjm/project/AbFlow/scripts/test/test.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
set -euo pipefail

########## adjust configs according to your needs ##########
CODE_DIR=`realpath $(dirname "$0")/../..`
NUM_WORKERS=8
BATCH_SIZE="${BATCH_SIZE:-20}"
N_STEPS="${N_STEPS:-10}"
SHOW_SAMPLE_PROGRESS="${SHOW_SAMPLE_PROGRESS:-1}"
GPU="${GPU:-0}"

CKPT="${1:-}"
TEST_SET="${2:-}"
SAVE_DIR="${3:-}"
TASK="${4:-}" 
SURF_FILE="${5:-}"

# validity check
if [ -z "$CKPT" ] || [ -z "$TEST_SET" ]; then
    echo "Usage: bash $0 <checkpoint> <test set> [save_dir] [task] [surf_file]"
    echo "  task: rabd (generate.py), igfold (struct_generate.py), or custom path"
    exit 1
fi

CKPT=`realpath "$CKPT"`
TEST_SET=`realpath "$TEST_SET"`

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="$(dirname "$CKPT")/results"
fi
SAVE_DIR=`realpath -m "$SAVE_DIR"`

TEST_DIR=`dirname "$TEST_SET"`

if [ "$TASK" = "rabd" ]; then
    PEP_ARG="--pep_file ${TEST_DIR}/test.pkl"
    SURF_ARG="--surf_file ${TEST_DIR}/test_surf.pkl"
    SCRIPT="generate.py"
elif [ "$TASK" = "igfold" ]; then
    PEP_ARG="--pep_file ${TEST_DIR}/test.pkl"
    SURF_ARG="--surf_file ${TEST_DIR}/test_surf.pkl"
    SCRIPT="struct_generate.py"
else
    PEP_ARG=""
    SURF_ARG="${SURF_FILE:+--surf_file ${SURF_FILE}}" 
    SCRIPT="generate.py"
fi
######### end of adjust ##########


# validity check
if [ -z "$CKPT" ]; then
	echo "Usage: bash $0 <checkpoint> <test set> [save_dir] [task]"
	echo "  task: rabd (generate.py), igfold (struct_generate.py), or custom path"
	exit 1;
else
	CKPT=`realpath $CKPT`
	SAVE_DIR=`realpath $SAVE_DIR`
fi

# echo Configurations
echo "Locate the project folder at ${CODE_DIR}"
echo "Using GPU: ${GPU}"
echo "Evaluating ${CKPT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sampling steps: ${N_STEPS}"
echo "Show sample progress: ${SHOW_SAMPLE_PROGRESS}"
echo "Test set: ${TEST_SET}"
echo "Test dir: ${TEST_DIR}"
echo "Pep arg: ${PEP_ARG}"
echo "Surf arg: ${SURF_ARG}"
echo "Results will be written to ${SAVE_DIR}"
echo "Task: ${TASK:-none}"
echo "Script: ${SCRIPT}"

# set gpu
export CUDA_VISIBLE_DEVICES=$GPU

# generate
cd ${CODE_DIR}

mkdir -p "${SAVE_DIR}"

GEN_EXTRA_ARGS="--n_steps ${N_STEPS}"

if [ "${SHOW_SAMPLE_PROGRESS}" = "1" ]; then
    GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS} --show_sample_progress"
fi

python ${SCRIPT} \
    --ckpt ${CKPT} \
    --test_set ${TEST_SET} \
    --save_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gpu 0 \
    ${PEP_ARG} \
    ${SURF_ARG} \
    ${GEN_EXTRA_ARGS}

echo "Done generation"

SUMMARY_FILE="${SAVE_DIR}/summary.json"
if [ ! -f "${SUMMARY_FILE}" ]; then
    echo "[ERROR] Generation failed: ${SUMMARY_FILE} was not created."
    exit 1
fi

# calculate metrics
OPENMM_CPU_THREADS=1 python cal_metrics.py \
    --test_set ${SUMMARY_FILE} \
    --num_workers ${NUM_WORKERS}
    

echo "Done evaluation"
