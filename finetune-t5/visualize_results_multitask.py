import os
import json
from glob import glob
import statistics
from collections import defaultdict
import pprint

from numpy import average

pp = pprint.PrettyPrinter(depth=6)


def get_step_mean_median(step_dir_path, metric_key="accuracy"):
    scores = []
    for json_file in glob(os.path.join(step_dir_path,"*")):
        result = json.load(open(json_file))
        scores.append(result['evaluation'][metric_key])
    mean_ = statistics.mean(scores)
    median_ = statistics.median(scores)
    return mean_, median_

def rank_steps(results):
    
    average_mean_plus_median_steps = []
    for step_name, values in results.items():
        mean_sum = 0
        median_sum = 0
        for task, scores in values.items():
            mean_sum += scores['mean']
            median_sum += scores['median']
        avg_mean = mean_sum/len(values)
        avg_median = median_sum/len(values)
        # print(f'------ {step_name} --------')
        # print("avg mean:", avg_mean)
        # print("avg median:", avg_median)
        # print(f'---------------------------')
        avg_mean_plus_median = avg_mean + avg_median
        average_mean_plus_median_steps.append((step_name, avg_mean_plus_median))
    average_mean_plus_median_steps = sorted(average_mean_plus_median_steps, key=lambda x: x[1], reverse=True)
    pp.pprint(average_mean_plus_median_steps)
    return average_mean_plus_median_steps[0][0]

def write_out(best_step, results, input_root, output_root):
    output_dir = os.path.join(output_root, os.path.basename(input_root))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{best_step}.txt")
    result = results[best_step]
    with open(output_path, 'w') as out:
        out.write(f"{os.path.basename(input_root)}\t\t\n")
        for task_name in [
            "openbookqa_main",
            "piqa",
            "rotten_tomatoes",
            "super_glue_cb",
            "super_glue_copa",
            "super_glue_wic",
            "wiki_qa",
            "hellaswag"  
        ]:
            if task_name in result:
                values = result[task_name]
                out.write(f"{task_name}\t{values['mean']}\t{values['median']}\n")
            else:
                out.write(f"{task_name}\t\t\n")

def write_out_v1(best_step, results, input_root, output_root):
    output_dir = os.path.join(output_root, os.path.basename(input_root))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{best_step}.txt")
    result = results[best_step]
    with open(output_path, 'w') as out:
        out.write(f"{os.path.basename(input_root)}\t\t\n")
        for task_name in [
            "openbookqa_main",
            "piqa",
            "super_glue_cb",
            "super_glue_wic",
            "hellaswag" 
        ]:
            if task_name in result:
                values = result[task_name]
                out.write(f"{task_name}\t{values['mean']}\t{values['median']}\n")
            else:
                out.write(f"{task_name}\t\t\n")


TASKS_V1 = [
    "openbookqa_main",
    "piqa",
    "rotten_tomatoes",
    "super_glue_cb",
    "super_glue_copa",
    "super_glue_wic",
    "hellaswag"    
]
TASKS_V1_PLUS = [
    "openbookqa_main",
    "piqa",
    "super_glue_cb",
    "super_glue_wic",
    "hellaswag"
]

if __name__ == "__main__":


    #TODO: set up output root
    output_root = "/data2/mikeeewang/finetune-t5/output/eval/visualization_official"
    
    metric_key = "accuracy"

    TASKS = TASKS_V1

    #TODO: set up input root: the output directory in eval/
    input_roots = [
        # "/data2/mikeeewang/finetune-t5/output/eval/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_baseline_no_aug",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_concat_baseline_5aug",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_5aug",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_5_5aug_FiD_baseline",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_5aug__dummy_aug",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_mulcqa_n_2_c4_5percent_10aug_latent_64",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_mulcqa_n_2_c4_5percent_20aug_latent_64_FrozenAugEncoder",
        # "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_mulcqa_n_2_c4_5percent_30aug_latent_64_FrozenAugEncoder",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_1_multitask_mixture_mulcqa_n_2_c4_5percent_NoTanhGate_5aug_latent64",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_3_multitask_mixture_mulcqa_n_2_c4_5percent_5aug_shuffled_latent_64",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_5aug_cross_shuffled",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_5aug_latent_32",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_5aug_latent_128",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_5aug_t5_large",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_t5_large_baseline_no_aug",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_fit_10_to_5aug_max_512_latent_64",
        # "/data2/mikeeewang/finetune-t5/output/eval/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_fit_10_to_5aug_max_512_latent_128"
        "/data2/mikeeewang/finetune-t5/output/eval/8_12_multitask_mixture_mulcqa_n_2_c4_5percent_t5_large_concat"
    ]
    write_function = write_out

    # TASKS = TASKS_V1_PLUS
    # input_roots = [
    #     "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_v1plus_n_2_c4_5percent_baseline_no_aug",
    #     "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_v1plus_n_2_c4_5percent_baseline_concat_5_aug",
    #     "/data2/mikeeewang/finetune-t5/output/eval/7_30_multitask_mixture_v1plus_n_2_c4_5percent_5aug_latent_64"
    # ]
    # write_function = write_out_v1

    for input_root in input_roots:
        
        results = defaultdict(dict)
        
        for task_name in TASKS:
            task_dir = os.path.join(input_root,task_name)
            if not os.path.exists(task_dir):
                print(f"NOTE: skip {task_name}!!!")
                continue
            for step_path in sorted(glob(os.path.join(task_dir,"*"))):
                step_name = os.path.basename(step_path)
                mean_, median_ = get_step_mean_median(step_path, metric_key=metric_key)
                results[step_name][task_name] = {
                    "mean": mean_,
                    "median": median_
                }
        
        pp.pprint(results)

        best_step = rank_steps(results)
        print("best step:", best_step)
        pp.pprint(results[best_step])

        ### output ###
        write_function(best_step, results, input_root, output_root)

