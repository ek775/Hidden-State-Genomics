#!/bin/bash

cd ~/Hidden-State-Genomics

python -m hsg.featureanalysis.featureKG --input data/cisplatin_neg45k.fa --output data/cisplatin_neg45k_kg.json
python -m hsg.featureanalysis.featureKG --input data/cisplatin_pos.fa --output data/cisplatin_pos_kg.json

sudo shutdown -h now