#!/bin/bash


# run multiple intervention analyses on the central features

cisplatin_positive=("407" "3378" "4793")
cisplatin_negative=("3378" "7030" "8161")

# positive set
for feature in "${cisplatin_positive[@]}"; do
    for min_act in 0.01 0.1 1.0; do
        for act_factor in 5.0 10.0 50.0; do
            python -m hsg.featureanalysis.intervention \
                --feature $feature \
                --min_act $min_act \
                --act_factor $act_factor \
                --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
        ; done \
    ; done \
; done

# negative set
for feature in "${cisplatin_negative[@]}"; do
    for min_act in 0.01 0.1 1.0; do
        for act_factor in 5.0 10.0 50.0; do
            python -m hsg.featureanalysis.intervention \
                --feature $feature \
                --min_act $min_act \
                --act_factor $act_factor \
                --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
        ; done \
    ; done \
; done