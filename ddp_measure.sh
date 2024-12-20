#!/bin/bash

config="vanilla_one_store_med"

for ((nproc=1; nproc<=8; nproc++)); do
  torchrun --standalone --nnodes=1 --nproc_per_node="${nproc}" main_run_ddp.py one_store_lost "${config}"
done
