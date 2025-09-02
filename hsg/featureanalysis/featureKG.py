from hsg.stattools.features import get_latent_model
from hsg.featureanalysis.regelementcorr import read_bed_file, get_sequences_from_dataframe

import networkx as nx
import numpy as np
import torch
from biocommons.seqrepo import SeqRepo
from tqdm import tqdm
from networkx.readwrite import json_graph

import os, json

from dotenv import load_dotenv
load_dotenv()

### helper functions ###
def extract_features_and_tokens(sequences: list[str], descriptions: list[str], model) -> tuple[dict, dict]:

    seq_features = {}
    seq_tokens = {}

    for i, seq in enumerate(tqdm(sequences, desc="Extracting features")):
        with torch.no_grad():
            feats, tokens = model.forward(seq, return_tokens=True)
            # trim "pad" tokens
            tokens = tokens[:feats.size(0)]
            # get best feature across each token
            best_feats = torch.argmax(feats, dim=1).cpu().numpy().tolist()
            token_feats = [(token, best_feats[i]) for i, token in enumerate(tokens)]

        # aggregate sequence-level features
        if seq_features.get(descriptions[i]) is None:
            seq_features[descriptions[i]] = [int(feat) for feat in set(best_feats)]
        else:
            seq_features[descriptions[i]].extend([int(feat) for feat in set(best_feats)])
            seq_features[descriptions[i]] = list(set(seq_features[descriptions[i]]))

        # aggregate token-level features
        for token, feat in token_feats:
            if seq_tokens.get(token):
                seq_tokens[token].append(int(feat))
            else:
                seq_tokens[token] = [int(feat)]

    # reduce token feature ids to uniques
    for token, feats in seq_tokens.items():
        seq_tokens[token] = list(set(feats))

    return seq_features, seq_tokens


### main ###
def main(input: str, output: str, exp_factor: int = 8, layer_idx: int = 23, sae_dir: str = None):
    """
    Args:
        input (str): Path to the input file (BED format).
        output (str): Path to the output file (JSON format).
        exp_factor (int): Expansion factor for SAE.
        layer_idx (int): Layer index from parent model to extract features from.
        sae_dir (str, optional): Directory containing the SAE model checkpoint. Defaults to author's local path.
    """
    
    if sae_dir is None:
        sae_path = f"/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef{exp_factor}/sae/layer_{layer_idx}.pt"
    else:
        sae_path = f"{sae_dir}/layer_{layer_idx}.pt"


    print("Loading model...")
    model = get_latent_model(parent_model_path=os.environ["NT_MODEL"], layer_idx=layer_idx, sae_path=sae_path)
    print(model)
    print("----------- <model> -----------")

    # load data
    if input.endswith(".bed"):
        seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
        sequences, dataframe = get_sequences_from_dataframe(
            read_bed_file(input, max_columns=6,), # limit=100), # set limit for debugging 
            pad_size=0, 
            seqrepo=seqrepo, 
            return_df=True,
        )

        descriptions = [f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}({row['strand']})" for index, row in dataframe.iterrows()]
        del dataframe

    elif input.endswith((".fasta", ".fa")):
        from Bio import SeqIO
        from Bio.SeqRecord import SeqRecord
        records: list[SeqRecord] = list(SeqIO.parse(input, "fasta"))
        sequences = [str(record.seq) for record in records]
        descriptions = [record.id for record in records]

    else:
        raise ValueError("Unsupported file format")

    print("----------- <data> -----------")

    # extract features and tokens
    seq_feats, token_feats = extract_features_and_tokens(sequences, descriptions, model)

    # construct a networkx graph with sequences, features, and tokens as KG nodes connected by co-occurrences
    G = nx.Graph()

    for seq, feats in tqdm(seq_feats.items(), desc="Adding sequence edges"):
        G.add_edges_from([(seq, feat) for feat in feats])

    for token, feats in tqdm(token_feats.items(), desc="Adding token edges"):
        G.add_edges_from([(token, feat) for feat in feats])

    print(G)

    # save the knowledge graph as a json file
    with open(output, "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=4)

    # visualize the knowledge graph
    import matplotlib.pyplot as plt
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.show()




if __name__ == "__main__":
    from tap import tapify
    tapify(main)
