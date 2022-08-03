

DATASET_DIR_NAME=$1
DATASETS_ROOT=$2

LR=$3
BS=$4

BS_EVAL=16

K=0
LATENT_SIZE=64
MAX_LENGTH=$((1024-${K}*${LATENT_SIZE}))
AUG_MAX_LENGTH=512
TARGET_MAX_LENGTH=256


STEP_NUM=-1 # run all steps


DATASETS_DIR="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"

### eval T5ForConditionalGenerationWithPerceiverResamplerXattnOnEncoder ###
# MODEL_ARCHITECTURE="fusionOnEncoder"

# ### eval T5ForConditionalGenerationWithPerceiverResamplerXattnOnDecoder ###
# MODEL_ARCHITECTURE="fusionOnDecoder"

### eval T5ForConditionalGenerationSharedEncoderXattnOnDecoder ###
# MODEL_ARCHITECTURE="SharedEncoder_fusionOnDecoder"

### train T5ForConditionalGenerationMultiAug ### v1
MODEL_ARCHITECTURE="SharedEncoderDecoder_MultiAug"

### train T5ForConditionalGenerationMultiAug ###
# MODEL_ARCHITECTURE="SharedEncoderDecoder_MultiAug_Base"


### no calibrate setting & input aug seperated

MODEL_PATH=$5
OUTPUT_DIR=$6

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_8_with_KiC_aug_k-1_single_task_${MODEL_ARCHITECTURE}/t5-base/${DATASET_DIR_NAME}_${MODEL_ARCHITECTURE}_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_8"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_8_with_KiC_aug_k-1_single_task_${MODEL_ARCHITECTURE}/${DATASET_DIR_NAME}_${MODEL_ARCHITECTURE}_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_8_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_7_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v7/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v7_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_7"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_7_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v7/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v7_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_7_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_7_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v6/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v6_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_7"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_7_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v6/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v6_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_7_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_6_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v1/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v1_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_6"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_6_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v1/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v1_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_6_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_5_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v5/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v5_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_5"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_5_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v5/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v5_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_5_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_5_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v4/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v4_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_5"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_5_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v4/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v4_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_5_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_4_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v3/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v3_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_4"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_4_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v3/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v3_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_4_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_4_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v2/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v2_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_4"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_4_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v2/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_v2_lr-${LR}_bs-${BS}_KiC_aug_unfreeze_lm_7_4_step_${STEP_NUM}"

# MODEL_PATH="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_2_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v1/t5-base/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_lr-${LR}_bs-2_KiC_aug_unfreeze_lm_7_2"
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_2_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug_v1/${DATASET_DIR_NAME}_SharedEncoderDecoder_MultiAug_lr-${LR}_bs-2_KiC_aug_unfreeze_lm_7_2_step_${STEP_NUM}"


CUDA_VISIBLE_DEVICES=$7 accelerate launch --main_process_port $8 --num_processes $9 eval_original_task_only_xattn_KiC_augmentation.py \
    --processed_dataset_paths ${DATASETS_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --saved_model_step ${STEP_NUM} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size ${BS_EVAL} \
    --max_length ${MAX_LENGTH} \
    --target_max_length ${TARGET_MAX_LENGTH} \
    --model_architecture ${MODEL_ARCHITECTURE}