#!/bin/bash

config="vanilla_one_store_med"
nproc=4

# 0. baseline
python main_run.py train one_store_lost "${config}"

# 1. mixed precision
python main_run.py train one_store_lost "${config}" mprecision

# 2. JIT - default backend
python main_run.py train one_store_lost "${config}" compile

# 3. DDP
torchrun --standalone --nnodes=1 --nproc_per_node="${nproc}" main_run_ddp.py one_store_lost "${config}"
