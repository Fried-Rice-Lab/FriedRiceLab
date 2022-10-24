#!/bin/bash
######################################################################
#
# A rudimentary Bash script.
# The `cpu mode` and `--auto_resume` are not supported.
#
# sh run.sh func.py    task.yml   expe.yml   debug      force_yml
#           [Required] [Required] [Required] [Optional] [Optional]
#
######################################################################

python_path="python"
devices=0 # 0,1,2,3
num_devices=1

if [ -z "$4" ]; then
  CUDA_VISIBLE_DEVICES="$devices" \
    "$python_path" \
    -u -m torch.distributed.run \
    --nproc_per_node="$num_devices" \
    --master_port=4001 \
    "$1" \
    -expe_opt "$2" \
    -task_opt "$3" \
    --launcher pytorch
else
  if [ "$4" = "debug" ]; then
    if [ -z "$5" ]; then
      CUDA_VISIBLE_DEVICES="$devices" \
        "$python_path" \
        -u -m torch.distributed.run \
        --nproc_per_node="$num_devices" \
        --master_port=4001 \
        "$1" \
        -expe_opt "$2" \
        -task_opt "$3" \
        --debug \
        --launcher pytorch
    else
      CUDA_VISIBLE_DEVICES="$devices" \
        "$python_path" \
        -u -m torch.distributed.run \
        --nproc_per_node="$num_devices" \
        --master_port=4001 \
        "$1" \
        -expe_opt "$2" \
        -task_opt "$3" \
        --debug \
        --force_yml "$5" \
        --launcher pytorch
    fi
  else
    CUDA_VISIBLE_DEVICES="$devices" \
      "$python_path" \
      -u -m torch.distributed.run \
      --nproc_per_node="$num_devices" \
      --master_port=4001 \
      "$1" \
      -expe_opt "$2" \
      -task_opt "$3" \
      --force_yml "$4" \
      --launcher pytorch
  fi
fi
