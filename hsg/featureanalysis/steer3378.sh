#!/bin/bash

cisplatin_positive=("3378")

# positive set
for feature in "${cisplatin_positive[@]}"; do
    for min_act in 0.0 0.5 1.0 5.0 8.0; do
        for act_factor in 0.0 5.0 8.0; do
            python -m hsg.featureanalysis.intervention \
                --feature $feature \
                --min_act $min_act \
                --act_factor $act_factor \
                --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
                --folder-name "tuning-3378" \
        ; done \
    ; done \
; done