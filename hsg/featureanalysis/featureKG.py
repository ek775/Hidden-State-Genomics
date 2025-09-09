from hsg.stattools.features import get_latent_model
from hsg.featureanalysis.regelementcorr import read_bed_file, get_sequences_from_dataframe

import networkx as nx
import numpy as np
import torch
from biocommons.seqrepo import SeqRepo
import gffutils
from tqdm import tqdm
from networkx.readwrite import json_graph

import os, json, re

from dotenv import load_dotenv
load_dotenv()

### helper functions ###
def extract_features_and_tokens(sequences: list[str], descriptions: list[str], model) -> dict[dict]:

    seq_features = {}
#    seq_tokens = {}

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
            seq_features[descriptions[i]] = {k:v for k, v in token_feats}
        else:
            seq_features[descriptions[i]] = {k:v for k, v in token_feats if k not in seq_features[descriptions[i]]}

    return seq_features

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
    
    # initialize database with gffutils for gene annotation data
    annotdb_path = '.'.join(os.environ["REFSEQ_GTF"].split('.')[:-1]) + '.db'

    if not os.path.exists(annotdb_path):
        print("Creating annotation database...")
        annotdb = gffutils.create_db(os.environ["REFSEQ_GTF"], 
                                        dbfn='.'.join(os.environ["REFSEQ_GTF"].split('.')[:-1]) + '.db', 
                                        verbose=True, 
                                        keep_order=True, 
                                        merge_strategy='merge', 
                                        sort_attribute_values=True)
    
    # connect
    print("Connecting to annotation database...")
    annotdb = gffutils.FeatureDB(annotdb_path, keep_order=True)
    print("DONE")

    print("----------- <data> -----------")

    # extract features and tokens
    seq_token_feats = extract_features_and_tokens(sequences, descriptions, model)

    # construct a networkx graph with sequences connected to tokens by feature typed edges
    G = nx.MultiDiGraph()

    for sequence in tqdm(seq_token_feats, desc="Building knowledge graph"):
        for token, feat in seq_token_feats[sequence].items():

            # var sequence is in this case the hg38 coordinate description
            match = re.match(r"(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)\((?P<strand>[+-])\)", sequence)
            if match:
                chrom = match.group("chrom")
                start = int(match.group("start"))
                end = int(match.group("end"))
                strand = match.group("strand")
            else:
                raise ValueError(f"Invalid sequence format: {sequence}")
            
            # search for gene features in ncbi refseq annotation
            feature_types = ['gene', 'transcript', 'exon', 'CDS', 'start_codon', 'stop_codon', '3UTR', '5UTR']
            annotations = annotdb.region(region=(chrom, start, end), strand=strand, featuretype=feature_types)

            seqmetadata = {}
            for ann in annotations:
                attr = {k: v[0] for k, v in ann.attributes.items()}
                attr['featuretype'] = ann.featuretype
                attr['strand'] = ann.strand
                attr['start'] = ann.start
                attr['end'] = ann.end
                
                id = attr.get(f"{ann.featuretype}_id", f"{ann.id}:{ann.start}-{ann.end}")
                seqmetadata[id] = attr

            # add nodes and edges
            # tokens point to features bc tokens activate features
            G.add_edge(token, feat, sequence=sequence, **seqmetadata)

    # show graph info
    print(G)

    # save the knowledge graph as a json file
    with open(output, "w") as f:
        json.dump(json_graph.node_link_data(G, edges="edges"), f, indent=4)

    # visualize the knowledge graph
    import matplotlib.pyplot as plt
    def node_color(x):
        return "lightblue" if type(x) == str else "red"
    colors = [node_color(x) for x in G.nodes]
    nx.draw(G, pos=nx.spring_layout(G, k=1/(len(G.nodes)**1e-100_000)), with_labels=True, node_color=colors)
    plt.show()




if __name__ == "__main__":
    from tap import tapify
    tapify(main)
