#!/bin/bash


# run multiple intervention analyses on the differentially central features

cisplatin_positive=(1371 9853 8596 407 3901)
cisplatin_negative=(9021 8161 2453 2778 6421)

# positive set
for min_act in 0.1 10.0; do
    for act_factor in 0.0 10.0; do
        python -m hsg.featureanalysis.intervention \
            --feature "${cisplatin_positive[@]}" \
            --min_act $min_act \
            --act_factor $act_factor \
            --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
            --folder_name intervention_reports/positive/ \
    ; done \
; done

# negative set
for min_act in 0.1 10.0; do
    for act_factor in 0.0 10.0; do
        python -m hsg.featureanalysis.intervention \
            --feature "${cisplatin_negative[@]}" \
            --min_act $min_act \
            --act_factor $act_factor \
            --cnn gs://hidden-state-genomics/cisplatinCNNheads/ef8/layer_23/embeddings.pt \
            --folder_name intervention_reports/negative/ \
    ; done \
; done


gsutil -m cp -r ./data/intervention_reports/ gs://hidden-state-genomics/featureanalysis/intervention_results3/

sudo shutdown -h now