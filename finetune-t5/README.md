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




