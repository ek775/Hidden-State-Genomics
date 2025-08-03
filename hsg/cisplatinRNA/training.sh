#!/bin/bash

cd "$(git rev-parse --show-toplevel)"

# run CNNtrain.py training script on each layer of NTv2-500m-human-ref
# and across all expansion sizes
expansion_sizes=("8" "16" "32")

for expansion_size in "${expansion_sizes[@]}"; do
    for i in {0..23}; do
        python -m hsg.cisplatinRNA.CNNtrain \
            --cisplatin_positive data/cisplatin_positive \
            --cisplatin_negative data/cisplatin_negative \
            --batch_size 16 \
            --epochs 100 \
            --layer_idx $i \
            --exp_factor $expansion_size \
    ; done \
; done

sudo shutdown -h now