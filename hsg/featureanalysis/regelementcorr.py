from tap import tapify
from tqdm import tqdm
from dotenv import load_dotenv

import pandas as pd

from hsg.stattools.features import get_latent_model
from hsg.stattools.crosscorr import construct_alignments, binarize_features, bed_to_array, features_to_bed, cross_correlation, xcorr_pearson

import os


# functions
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


# main
def main(data_path: str,):

    model = get_latent_model(os.environ["NT_MODEL"], layer_idx=23, 
                             sae_path="/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef16/sae/layer_23.pt")

    # load data



# run module
if __name__ == "__main__":
    load_dotenv()
    tapify(main)