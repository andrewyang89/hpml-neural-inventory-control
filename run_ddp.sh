#!/bin/bash

if [ $# != 1 ]; then
  echo "Please provide an integer argument for number of processes to run".
  exit 1
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$1" main_run_ddp.py
