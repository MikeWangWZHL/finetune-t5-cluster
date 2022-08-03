### Training mixture
navigate to `/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/training`
run the following training scripts:
- no demo:
`bash run_mixture_k-0.sh`
- k = 1
`bash run_mixture_k-1.sh`
- k = 3
`bash run_mixture_k-3.sh`

#### conda env path example (on CA31) : `/data2/mikeeewang/miniconda3/envs/t-zero` 



### Training

1. set accelerate

    `accelerate config`

2. example of running command

    - `accelerate launch single_task_fine_tune.py --model_name_or_path google/t5-small-lm-adapt --output_dir /workspace/my_output --use_processed_dataset --path_to_preprocessed_training_data /workspace/processed/wic/original/merged_training.json --path_to_preprocessed_eval_data /workspace/processed/wic/original/merged_eval.json --save_model -tb 32 -ep 1`

    - `accelerate launch multi_task_fine_tune.py --model_name_or_path google/t5-small-lm-adapt --processed_dataset_paths /workspace/data/bigscience_P3/wiki_qa_Decide_good_answer /workspace/data/bigscience_P3/qasc_is_correct_1 /workspace/data/bigscience_P3/quarel_choose_between /workspace/data/bigscience_P3/quartz_answer_question_based_on /workspace/data/bigscience_P3/sciq_Direct_Question_Closed_Book_ /workspace/data/bigscience_P3/anli_GPT_3_style_r1 /workspace/data/bigscience_P3/super_glue_cb_GPT_3_style /workspace/data/bigscience_P3/super_glue_rte_GPT_3_style /workspace/data/bigscience_P3/glue_mrpc_equivalent /workspace/data/bigscience_P3/paws_labeled_final_Concatenation --warmup_ratio 0.25 --output_dir /workspace/my_output --save_model`