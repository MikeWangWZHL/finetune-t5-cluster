### kic single tasks
## input format
    # DATASET_DIR_NAME=$1
    # DATASETS_ROOT=$2
    # LR=$3
    # BS=$4
    # MODEL_PATH=$5
    # OUTPUT_DIR=$6
    # CUDA_VISIBLE_DEVICES=$7
    # --main_process_port $8
    # --num_processes $9

# CAUSAL_AUG_PATH="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_knowledge_augmented/processed_causal"
# OHTER_AUG_PATH="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_knowledge_augmented/processed_nokic_lexicon_commonsense_event_script"
KNOWLEDGE_AUG_PATH="/cephfs/user/mikeeewang/summer_22/workspace/data/opendomain_qa_datasets/preprocessed/from_FiD"
# GOLD_AUG_PATH="/cephfs/user/mikeeewang/summer_22/workspace/data/opendomain_qa_datasets/preprocessed/from_FiD/Gold_k-20"
GOLD_AUG_PATH="/cephfs/user/mikeeewang/summer_22/workspace/data/opendomain_qa_datasets/preprocessed/from_FiD/Gold_k-20_with_answers"
OUTPUT_ROOT="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval"



#################
#################
## generation evaluation

DATASET_NAME="NQ"
bash run_eval_finetuned_multitask_generation_concat_input_at_aug.sh \
${DATASET_NAME} \
${GOLD_AUG_PATH} \
0.0001 \
2 \
"/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_25_opendomain_qa_gold/t5-base/7_25_NQ_gold_concat_input_at_aug_5aug_latent_64" \
"${OUTPUT_ROOT}/7_25_opendomain_qa_gold/7_25_NQ_gold_concat_input_at_aug_5aug_latent_64" \
4,5,6,7 \
20655 \
4

# DATASET_NAME="TQA"
# DATASET_NAME="NQ"
# bash run_eval_finetuned_multitask_generation.sh \
# ${DATASET_NAME} \
# ${GOLD_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_23_opendomain_qa_gold/t5-base/7_23_NQ_gold_5aug_latent_64" \
# "${OUTPUT_ROOT}/7_23_opendomain_qa_gold/7_23_NQ_gold_5aug_latent_64_new_eval_code" \
# 0,1,2,3 \
# 20656 \
# 4

# ## generation evaluation
# DATASET_NAME="TQA"

# bash run_eval_finetuned_single_xattn_KiC_augmentation_generation.sh \
# ${DATASET_NAME} \
# ${KNOWLEDGE_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_23_mulcqa_with_kic_generation_template/7_23_mulcqa_with_kic_generation_template_single_aug_commonsense_latent_64" \
# "${OUTPUT_ROOT}/7_23_mulcqa_with_kic_generation_template/${DATASET_NAME}__7_23_mulcqa_with_kic_generation_template_single_aug_commonsense_latent_64" \
# 4,5,6,7 \
# 20655 \
# 4

# DATASET_NAME="piqa"
# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_21_mulcqa_with_kic/7_21_mulcqa_with_kic_single_aug_commonsense_latent_64" \
# "${OUTPUT_ROOT}/7_21_mulcqa_with_kic_single_aug_commonsense_latent_64/${DATASET_NAME}__7_21_mulcqa_with_kic_single_aug_commonsense_latent_64" \
# 4,5,6,7 \
# 20655 \
# 4

### sep aug

# DATASET_NAME="openbookqa_main"
# bash run_eval_finetuned_single_xattn_KiC_augmentation_sep_aug.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_21_mulcqa_with_kic/7_21_mulcqa_with_kic_sep_aug_commonsense_5aug_latent_64" \
# "${OUTPUT_ROOT}/7_21_mulcqa_with_kic_sep_aug_commonsense_5aug_latent_64/${DATASET_NAME}__7_21_mulcqa_with_kic_single_aug_commonsense_latent_64" \
# 4,5,6,7 \
# 20655 \
# 4

# DATASET_NAME="piqa"
# bash run_eval_finetuned_single_xattn_KiC_augmentation_sep_aug.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_21_mulcqa_with_kic/7_21_mulcqa_with_kic_sep_aug_commonsense_5aug_latent_64" \
# "${OUTPUT_ROOT}/7_21_mulcqa_with_kic_sep_aug_commonsense_5aug_latent_64/${DATASET_NAME}__7_21_mulcqa_with_kic_single_aug_commonsense_latent_64" \
# 4,5,6,7 \
# 20655 \
# 4






#################
#################

# DATASET_NAME="cos_e_v1.11"

# # bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# # ${DATASET_NAME} \
# # ${CAUSAL_AUG_PATH} \
# # 0.0001 \
# # 2 \
# # "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# # "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# # 4,5,6,7 \
# # 20655 \
# # 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4


#################
#################
# DATASET_NAME="piqa"

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${CAUSAL_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4


#################
#################
# DATASET_NAME="super_glue_wic"

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${CAUSAL_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_causal_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_event_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_lexicon_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_script_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_single_xattn_KiC_augmentation.sh \
# ${DATASET_NAME} \
# ${OHTER_AUG_PATH} \
# 0.0001 \
# 2 \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# "${OUTPUT_ROOT}/7_19_with_KiC_aug_k-1_single_task_SharedEncoderDecoder_MultiAug/t5-base/${DATASET_NAME}_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_commonsense_unfreeze_lm_7_19" \
# 4,5,6,7 \
# 20655 \
# 4