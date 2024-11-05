import json
from Bio import SeqIO
import re
import os
import pandas as pd

# Load the metadata
print("Loading metadata...")
with open('./data/query_all_heme_vars.json', 'r', encoding='latin-1') as file:
    metadata = json.load(file) # why is this latin-1 encoded???

# drop the stuff we don't need
metadata_df = pd.DataFrame(metadata)
metadata_df = metadata_df.drop(
    columns=[
        'references', # unnecessary
        'hematology', # clinical data
        'syllabusName', # unclear relevance
        'externalLinks', # internet links
        'id', # local database id, not useful
        'occurrenceComment', # population frequency irrelevant
        'variantType', # original SQL condition (truism)
        'occurence', # population frequency irrelevant
        ]
    )
print(f"Found metadata for {len(metadata_df)} variants.")

# Load the sequence data
print("Loading sequences...")
sequences = []
with open('./data/Human_Hb_variants.fasta', 'r') as file:
    for record in SeqIO.parse(file, 'fasta'):
        sequences.append(record)

# extract variant names
pattern = r'.*?\:(.*?),.*'
seq_entries = []
wack_names = 0
for seq in sequences:
    try:
        var_name = re.search(pattern, seq.description).group(1)
    except:
        wack_names +=1
        var_name = seq.description
    seq_entry = {
        'hemoglobin_chain': seq.id,
        'sequence': str(seq.seq),
        'variant_name': var_name
    }
    seq_entries.append(seq_entry)

var_seq_df = pd.DataFrame(seq_entries)
print(f"{wack_names} variant name descriptions could not be parsed.")
print(len(var_seq_df), "sequences loaded.")

# merge on variant names
print("Merging metadata with sequences...")
merged_df = pd.merge(metadata_df, var_seq_df, how='inner', left_on='name', right_on='variant_name')
print(merged_df.head())
print("Saving results...")
merged_df.to_csv('./data/hbvar_metadata_sequence_merge.csv', index=False)