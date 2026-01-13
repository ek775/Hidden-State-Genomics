#!/bin/bash

cisplatin_positive=("1302" "1376" "3278" "6469" "7137" "9704")

# positive set
for feature in "${cisplatin_positive[@]}"; do
    for min_act in 0.1 10.0; do
        for act_factor in 0.0 10.0; do
            python -m hsg.featureanalysis.intervention \
                --feature $feature \
                --min_act $min_act \
                --act_factor $act_factor \
                --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
        ; done \
    ; done \
; done