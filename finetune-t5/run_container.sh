# usage="Usage: $0 CODE_DIR WORKSPACE_DIR"

# if [ "$#" -ne 2 ]; then
#     echo $usage
#     exit 1
# fi

# CODE_DIR=$1;
# WORKSPACE_DIR=$2; shift

CODE_DIR="/data2/mikeeewang/cluster_workdir/finetune-t5"
DATA_DIR="/data2/mikeeewang/cluster_workdir/data"
CACHE_DIR="/data2/mikeeewang/cluster_workdir/cache_for_docker"

# IMAGE_NAME="mirrors.tencent.com/ai-lab-seattle/kic-t0-apex:v0"
# IMAGE_NAME="mirrors.tencent.com/ai-lab-seattle/kic-t0:latest"
IMAGE_NAME="mirrors.tencent.com/ai-lab-seattle/mikeeewang_t0"

# disable network
    # --network none \

sudo docker run --ipc=host --network=host --rm -it --gpus=all \
    --privileged=true \
    --name="mikeewang_t0_container" \
    -v $CODE_DIR:/code \
    -v $DATA_DIR:/data \
    -v $CACHE_DIR:/cache \
    -w /code \
    ${IMAGE_NAME} /bin/bash
