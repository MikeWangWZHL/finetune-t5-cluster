# Setup docker for cluster

## Data and code directories to be mounted:
As set in `run_container.sh` (all path are on tw2):
- CODE_DIR=`/data2/mikeeewang/cluster_workdir/finetune-t5`
- DATA_DIR=`/data2/mikeeewang/cluster_workdir/data`
- CACHE_DIR=`/data2/mikeeewang/cluster_workdir/cache_for_docker`

## Run the docker container
check if the image with name: `mirrors.tencent.com/ai-lab-seattle/mikeeewang_t0:latest` already exist using `sudo docker images`

- if yes: run the container
    `bash run_container.sh`

- if not: first build and then run container
    `bash docker_build.sh`
    `bash run_container.sh`

## [7/30] Run the experiments

`cd /code/training`

- job 0: single task finetuning (four settings):
    `bash run_jobs_docker_7_30_0.sh`

- job 1: multi-task finetuning 1:
    `bash run_jobs_docker_7_30_1.sh`

- job 2: multi-task finetuning 2:
    `bash run_jobs_docker_7_30_2.sh`

- job 3: multi-task finetuning 3:
    `bash run_jobs_docker_7_30_3.sh`

- job 4: multi-task finetuning 4 v1+:
    `bash run_jobs_docker_7_30_4.sh`


## [8/3 6:00pm] Experiments
At the "/code" dir, run:

- job 1: cont. eval 8_3_1:
`bash ./eval/run_eval_jobs_docker_8_3_1_rerun.sh`

- job 2: cont. eval 8_3_2:
`bash ./eval/run_eval_jobs_docker_8_3_2_rerun.sh`

- job 3: multitask shuffled version:
`bash ./training/run_jobs_docker_train_eval_8_3_5.sh`


## [8/4 1:30pm] Experiments
At the "/code" dir, run:

- job 0: 5aug smaller learning rate + more epoch:

    `bash ./training/run_jobs_docker_train_eval_8_4_0__learning_rate.sh`

- job 1: longer per aug:

    `bash ./training/run_jobs_docker_train_eval_8_4_1__fit_10_to_5aug_max_512_latent_64.sh`

- job 2: longer per aug + larger latent:

    `bash ./training/run_jobs_docker_train_eval_8_4_2__fit_10_to_5aug_max_512_latent_128.sh`

- job 3: ablation: 5aug latent 128:

    `bash ./training/run_jobs_docker_train_eval_8_4_3__5aug_latent_128.sh`

- job 4: ablation: 5aug latent 32:

    `bash ./training/run_jobs_docker_train_eval_8_4_4__5aug_latent_32.sh`


## [8/4 5:30pm] Experiments

NOTE: cache dir `/data2/mikeeewang/cluster_workdir/cache_for_docker` is updated with t5-large-lm-adapt checkpoint. Cluster needs to be synced.


- job 5: t5-large 5aug:
    `bash ./training/run_jobs_docker_train_eval_8_4_5__t5_large_5aug.sh`

- job 6: t5-large no aug baseline:
    `bash ./training/run_jobs_docker_train_eval_8_4_6__t5_large_baseline_no_aug.sh`


## [8/4 6:20pm] Experiments

- job 7: t5-large 5aug v1 plus mixture:
    `bash ./training/run_jobs_docker_train_eval_8_4_7__t5_large_5aug_v1plus.sh`