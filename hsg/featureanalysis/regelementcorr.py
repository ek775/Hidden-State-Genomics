from tap import tapify
from tqdm import tqdm
from dotenv import load_dotenv

import pandas as pd
from biocommons.seqrepo import SeqRepo

from hsg.stattools.features import get_latent_model
from hsg.stattools.crosscorr import construct_alignments, binarize_features, bed_to_array, features_to_bed, cross_correlation, xcorr_pearson

import os

##################################################################################################################################
# functions
###################################################################################################################################
def read_bed_file(bed_file: str) -> pd.DataFrame:
    """
    Read a BED file and return a DataFrame.
    
    Args:
        bed_file (str): Path to the BED file.
        
    Returns:
        pd.DataFrame: DataFrame containing the BED file data.
    """
    # Read the BED file line by line - cleaner skipping of header lines which may vary
    with open(bed_file, 'r') as f:

        rows = []
        columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
                   "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
        
        # Read each line in the file
        for line in tqdm(f.readlines()):

            # filter header lines
            if not line.startswith("chr"):
                continue
            # Split the line into columns
            items = line.strip().split('\t')

            rows.append(items)

        # if the rows are ragged, truncate the rows to the shortest length
        min_length = min(len(row) for row in rows)
        rows = [row[:min_length] for row in rows]

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=columns[:min_length])

        # Convert the columns to appropriate data types
        try:
            df['chrom'] = df['chrom'].astype(str)
            df['chromStart'] = df['chromStart'].astype(int)
            df['chromEnd'] = df['chromEnd'].astype(int)
            df['name'] = df['name'].astype(str)
            df['score'] = df['score'].astype(int)
            df['strand'] = df['strand'].astype(str)
            df['thickStart'] = df['thickStart'].astype(int)
            df['thickEnd'] = df['thickEnd'].astype(int)

            df['itemRgb'] = df['itemRgb'].astype(str)
            df['itemRgb'] = df['itemRgb'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

            df['blockCount'] = df['blockCount'].astype(int)

            df['blockSizes'] = df['blockSizes'].astype(str)
            df['blockSizes'] = df['blockSizes'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

            df['blockStarts'] = df['blockStarts'].astype(str)
            df['blockStarts'] = df['blockStarts'].apply(lambda x: list(map(int, x.split(','))) if x != '.' else [])

        except KeyError:
            # Most bed files do not use all 12 columns
            pass


        return df

def get_sequences_from_dataframe(df: pd.DataFrame, seqrepo: SeqRepo) -> list[str]:
    """
    Get sequences from a DataFrame using SeqRepo.
    
    Args:
        df (pd.DataFrame): DataFrame containing the BED file data.
        seqrepo (SeqRepo): SeqRepo object for fetching sequences.
        
    Returns:
        list: List of sequences.
    """
    sequences = []
    for _, row in df.iterrows():
        chrom = row['chrom']
        start = row['chromStart']
        end = row['chromEnd']
        seq = seqrepo.fetch(namespace="GRCh38", alias=chrom, start=start, end=end)
        sequences.append(seq)
    
    return sequences

#################################################################################################################################
# main
#################################################################################################################################
def main(data_path: str,):

    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    model = get_latent_model(os.environ["NT_MODEL"], layer_idx=23, 
                             sae_path="/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef16/sae/layer_23.pt")

    # load data
    data = read_bed_file(data_path)
    data = data.groupby("name")

    # calculate correlations for each annotation

    stats = {} # annotation: [pearson R for each feature]
    
    for group, df in tqdm(data):
        seq_list = get_sequences_from_dataframe(df, seqrepo)

    return None


###############################################################################
# run module
###############################################################################
if __name__ == "__main__":
    load_dotenv()
    tapify(main)