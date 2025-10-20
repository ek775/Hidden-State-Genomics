#!/bin/bash


# run multiple intervention analyses on the differentially central features

cisplatin_positive=("407" "3378" "4793" "2558" "1545")
cisplatin_negative=("1422" "7030" "8161" "5984" "7949")

# positive set
for feature in "${cisplatin_positive[@]}"; do
    for min_act in 0.1 1.0; do
        for act_factor in 0.0 10.0; do
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
    for min_act in 0.1 1.0; do
        for act_factor in 0.0 10.0; do
            python -m hsg.featureanalysis.intervention \
                --feature $feature \
                --min_act $min_act \
                --act_factor $act_factor \
                --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
        ; done \
    ; done \
; done