#!/bin/sh

export TPU_NAME="tpu-vm-1"
export TPU_LOAD_LIBRARY=0

# configure training runs
python SAE_train.py esm2-1280-ef8-v2 1280 8 &> esm2-1280-ef8-v2.log
python SAE_train.py esm2-1280-ef16-v2 1280 16 &> esm2-1280-ef16-v2.log
python SAE_train.py esm2-1280-ef32-v2 1280 32 &> esm2-1280-ef32-v2.log
python SAE_train.py esm2-1280-ef64-v2 1280 64 &> esm2-1280-ef64-v2.log