import torch
import os
import numpy as np
import pandas as pd
from biocommons.seqrepo import SeqRepo
from dotenv import load_dotenv
from tqdm import tqdm


def binarize_features(features: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Binarize features based on a threshold.

    Args:
        features (torch.Tensor): The input features to binarize.
        threshold (float): The threshold for binarization.

    Returns:
        torch.Tensor: The binarized features.
    """
    return torch.where(features > threshold, torch.tensor(1), torch.tensor(0))



def construct_alignments(features: torch.Tensor, tokens: list[str], chromosome: int, start: int) -> list[str]:
    """
    Construct alignments from features.

    Args:
        features (torch.Tensor): The binarized input features as a 1d array.
        chromosome (int): The chromosome number.
        start (int): The starting position of the features.

    Returns:
        list[str]: A list of strings representing the alignments in BED format.
    """
    alignments = []
    sequence = ''
    
    alignment_start = None
    alignment_end = None

    for i, token in enumerate(tokens):
        
        active = features[i].item()

        if active == 1 and alignment_start is None:
            # Start of a new alignment
            alignment_start = start + len(sequence)
            sequence += token

        if active == 1 and alignment_start is not None:
            # Contiguous activations
            sequence += token
            continue

        if active == 0 and alignment_start is not None:
            sequence += token
            # End of the current alignment
            alignment_end = start + len(sequence)
            # Add alignment to the list
            alignments.append(f"chr{chromosome}\t{alignment_start}\t{alignment_end}")
            # Reset for next alignment
            alignment_start = None
            alignment_end = None

        else:
            # No alignment found
            sequence += token
            continue

    return alignments



def features_to_bed(features: torch.Tensor, tokens: list[str], feature_id: str, chromosome: int, 
                    start: int, end: int, filepath: str, description: str) -> None:
    """
    Convert features to BED format and write to file.
    """

    header = f"""track name="{feature_id}" description="{description}" """

    alignments = construct_alignments(features, tokens, chromosome, start)

    if os.path.exists(filepath):
        with open(filepath, mode='a') as f:
            for a in alignments:
                f.write(a + '\n')

    if not os.path.exists(filepath):
        with open(filepath, mode='w') as f:
            f.write(header + '\n')
            for a in alignments:
                f.write(a + '\n')



def bed_to_array(filepath: str) -> dict:
    """
    Reads an annotation track from a BED file and constructs a 1D array for comparison against features.

    Args:
        filepath (str): Path to the BED file.

    Returns:
        dict: A dictionary where keys are chromosome names and values are 1D arrays representing the annotations.

        Dictionary format: {chromosome name: str, annotations: np.array}
    """
    load_dotenv()

    results = {}

    with open(filepath, "r") as f:
        print(f"Reading BED file: {filepath}")

        prev_end = None

        for line in tqdm(f.readlines()):
            # ignore header
            if not line.startswith("chr"):
                continue
            # ignore empty lines
            if line.strip() == "":
                continue
            # split line into columns
            elements = line.strip().split("\t")

            # unpack only core 3 elements for now
            # TODO: add support for score and other params
            chrome, chromStart, chromEnd = elements[:3]
            chromStart = int(chromStart)
            chromEnd = int(chromEnd)

            # add chromosome to results if not already present
            if chrome not in results.keys():
                results[chrome] = np.empty(0)
                prev_end = None

            # pad zeros from chr index zero to annotation start
            if chromStart != 0 and len(results[chrome]) == 0:
                results[chrome] = np.zeros(chromStart)
                prev_end = None

            # pad zeros from the last annotation end to the next start
            if prev_end is not None and chromStart != prev_end:
                results[chrome] = np.append(results[chrome], np.zeros(chromStart - prev_end))

            # append ones for the annotation
            results[chrome] = np.append(results[chrome], np.ones(chromEnd - chromStart))

            # cache the previous end for next iteration
            prev_end = chromEnd

    # use seqrepo to get the chromosome length and pad zeros to the end
    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    for key in results.keys():
        seq_length = len(str(seqrepo[f"GRCh38:{key}"]))
        if len(results[key]) < seq_length:
            results[key] = np.append(results[key], np.zeros(seq_length - len(results[key])))

    # return the arrays
    return results



def cross_correlation(features: np.array, annotation_array: np.array) -> np.ndarray:
    """
    Computes the normalized cross-correlation between two binary 1D arrays of equal length.
    """
    assert len(features) == len(annotation_array), "Features and annotation array must be of the same length."
    assert features.ndim == 1, "Features must be a 1D array."
    assert annotation_array.ndim == 1, "Annotation array must be a 1D array."

    # Compute the cross-correlation
    cross_corr = np.correlate(features, annotation_array, mode='full')
    # Normalize the cross-correlation
    cross_corr = cross_corr / ( np.sqrt(np.sum(features**2) * np.sum(annotation_array**2)) )

    return cross_corr



def xcorr_pearson(features: np.array, annotation_array: np.array) -> np.float64:
    """
    Computes the Pearson correlation coefficient between two binary 1D arrays of equal length.
    """
    assert len(features) == len(annotation_array), "Features and annotation array must be of the same length."
    assert features.ndim == 1, "Features must be a 1D array."
    assert annotation_array.ndim == 1, "Annotation array must be a 1D array."

    # Compute the Pearson correlation coefficient
    corr = np.corrcoef(features, annotation_array)[0, 1]
    return corr


##################################################################################################################
# Testing to make sure the functions work as expected
##################################################################################################################

if __name__ == "__main__":

    print("Testing cross-correlation functions...")

    # example usage
    import warnings
    warnings.filterwarnings("ignore")

    import re

    import torch
    import pandas as pd
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    from hsg.pipelines.variantmap import DNAVariantProcessor
    from hsg.stattools.features import *

    from dotenv import load_dotenv
    load_dotenv()

    layer = 23
    extractor = get_latent_model(os.environ["NT_MODEL"], layer_idx=layer, sae_path=f"/home/ek224/Hidden-State-Genomics/checkpoints/hidden-state-genomics/ef8/sae/layer_{layer}.pt")

    # load clingen dataset
    df = pd.read_csv(os.environ["CLIN_GEN_CSV"], header="infer", sep="\t")

    # parse the variants and get some unique refseq sequences
    processor = DNAVariantProcessor()

    refseqs = []
    varseqs = []
    var_objs = []
    for row in tqdm(df["HGVS Expressions"][42:45]):

        # try to pick out expressions listed in genomic coordinates
        exp = row.split(",")
        expression = None

        for i, e in enumerate(exp):
            if ":g." in e:
                expression = e
                break
            if i == len(exp) -1:
                expression = e
            else:
                continue

        try:
            # strip gene names & protein mutation tags
            expression = processor.clean_hgvs(expression)
            # parse variant
            var_obj = processor.parse_variant(expression)
            # perform liftover for various assemblies
            var_obj = processor.genomic_sequence_projection(var_obj)
            # get sequences
            refseq, offset = processor.retrieve_refseq(var_obj, return_offset=True)
            varseq = processor.retrieve_variantseq(var_obj)
            # append to lists
            refseqs.append((refseq, offset))
            varseqs.append(varseq)
            var_objs.append(var_obj)
    
        # return nulls if we can't parse the variant
        except:
            refseqs.append(None)
            varseqs.append(None)
            var_objs.append(None)

    # drop nulls
    for i in range(len(refseqs)):
        if refseqs[i] is None or varseqs[i] is None or var_objs[i] is None:
            refseqs.pop(i)
            varseqs.pop(i)
            var_objs.pop(i)
            continue

    features = []
    original_embeds = []
    token_sets = []

    # get the features and embeddings
    errors = []
    for entry in tqdm(varseqs):
    
        with torch.no_grad():
            feats, embeds, tokens, reconstructions = extractor.forward(entry, return_hidden_states=True, return_tokens=True, return_reconstructions=True)
            features.append(feats.cpu())
            original_embeds.append(embeds.cpu())
            token_sets.append(tokens)

            error = torch.nn.functional.mse_loss(reconstructions, embeds)
            errors.append(error.item())
        
    # print the average error
    print(f"Average Reconstruction MSE: {sum(errors)/len(errors)}")
    
    ############################################################################################################################################################
    # Test functions with loaded data
    ############################################################################################################################################################

    # construct alignments
    for i, feat in enumerate(features):

        binarized_feat = binarize_features(feat)
        
        selected_feature = binarized_feat[:, 1096]
        chromosome = re.findall(r'(\d{2})\.', str(var_objs[i].ac))[0]

        alignments = construct_alignments(selected_feature, token_sets[i], chromosome, var_objs[i].posedit.pos.start.base-refseqs[i][1]) 

        features_to_bed(features=selected_feature, tokens=token_sets[i], feature_id=f"feature_{i}", chromosome=chromosome, 
                        start=var_objs[i].posedit.pos.start.base-refseqs[i][1], end=var_objs[i].posedit.pos.end.base+refseqs[i][1], 
                        filepath=f"feature_{i}.bed", description="Feature description")
        
        print(f"Aligned Chromosome Feature Vectors for {i}:")
        features = bed_to_array(filepath=f"feature_{i}.bed")
        print(features)
        print(f"Cross-Correlation for {i} against random array:")
        random_array = np.random.randint(0, 2, size=len(features[f"chr{chromosome}"]))
        print("Pearson Cross-Correlation:", xcorr_pearson(features[f"chr{chromosome}"], random_array))