BS_EVAL=32

STEP_NUM=-1

K=0
LATENT_SIZE=64
MAX_LENGTH=$((1024-${K}*${LATENT_SIZE}))
AUG_MAX_LENGTH=256
TARGET_MAX_LENGTH=256

DATASET_DIR_NAME=$1
DATASETS_ROOT=$2
DATASETS_DIR="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"
MODEL_PATH=$3
OUTPUT_DIR=$4

# accelerate launch --main_process_port 20655 eval_original_task_only_xattn.py \
CUDA_VISIBLE_DEVICES=$5 accelerate launch --main_process_port $6 --num_processes $7 eval_original_task_only_xattn.py \
    --processed_dataset_paths ${DATASETS_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --saved_model_step ${STEP_NUM} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size ${BS_EVAL} \
    --max_length ${MAX_LENGTH} \
    --aug_max_length ${AUG_MAX_LENGTH} \
    --target_max_length ${TARGET_MAX_LENGTH}
