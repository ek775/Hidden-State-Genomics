from Bio import SeqIO, SeqRecord
from hsg.featureanalysis.regelementcorr import read_bed_file, get_sequences_from_dataframe
from biocommons.seqrepo import SeqRepo
import os
from dotenv import load_dotenv

load_dotenv()

def bed_to_fasta(bedfile: os.PathLike, fastafile:os.PathLike):
    seqrepo = SeqRepo(os.environ["SEQREPO_PATH"])
    sequences, dataframe = get_sequences_from_dataframe(read_bed_file(bedfile, max_columns=6), seqrepo, pad_size=0, return_df=True)
    print(dataframe.head())
    description = [f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}({row['strand']})" for index, row in dataframe.iterrows()]
    prefix = os.path.basename(bedfile).split(".")[0]

    print("Saving FASTA file...")

    with open(fastafile, "w") as f:
        for i, seq in enumerate(sequences):
            seqrecord = SeqRecord.SeqRecord(seq, id=description[i])
            SeqIO.write(seqrecord, f, "fasta")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert BED to FASTA")
    parser.add_argument("--bedfile", type=str, help="Input BED file")
    parser.add_argument("--fastafile", type=str, help="Output FASTA file")
    args = parser.parse_args()

    bed_to_fasta(args.bedfile, args.fastafile)