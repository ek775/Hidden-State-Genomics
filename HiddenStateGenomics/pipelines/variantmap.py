# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant
import re


class DNAVariantProcessor():

    """
    Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles
    """

    def __init__(self, assembly:str = "GRCh37", seqrepo_path:str = "./genome_databases/2024-05-23") -> None:
        # hgvs tools
        self.parser:Parser = Parser()
        self.assembly_mapper:AssemblyMapper = AssemblyMapper(connect(), assembly_name=assembly, alt_aln_method='splign')
        self.seq_repo:SeqRepo = SeqRepo(seqrepo_path) # TODO: dynamically set database 


    def clean_hgvs(self, raw_hgvs_expression:str) -> str:

        """
        Processes hgvs expressions with gene name and predicted protein change annotations and removes them for machine readability.
        """

        pattern = r"\([^)]*\)"
        clean_hgvs = re.sub(pattern, '', raw_hgvs_expression)

        return clean_hgvs.strip()


    def parse_variant(self, hgvs_expression:str) -> SequenceVariant:

        """
        Uses parser to parse input hgvs_expression
        """

        try:
            return self.parser.parse_hgvs_variant(hgvs_expression)  
         
        except Exception as e: 
            print(f"Invalid or Uncertain HGVS expression: {hgvs_expression}")
            print(e)
            return None
        

    def standardize_expression_type(self, hgvs_ref:SequenceVariant, expression_type:str = ["g","m","c","n","r","p"]):

        """
        Standardize hgvs types for proper mapping of refseqs and generation of variant sequences.

        Different hgvs types utilize different coordinate systems and may use non-contiguous indices, 
        making it difficult to properly modify refseqs to obtain variant sequences.
        """

        pass


    def retrieve_refseq(self, hgvs_ref:SequenceVariant) -> str:

        """
        Retrieve reference sequence from seqrepo.
        """

        sequence_proxy = self.seq_repo[f"refseq:{hgvs_ref.ac}"]
        return sequence_proxy.__str__()


    def retrieve_variantseq(self, hgvs_ref: SequenceVariant) -> str:

        """
        Modifies obtained refseq sequence to obtain the variant sequence.
        """

        variant_start: int = hgvs_ref.posedit.pos.start.base
        variant_end: int = hgvs_ref.posedit.pos.end.base
        ref: str = hgvs_ref.posedit.edit.ref
        var: str = hgvs_ref.posedit.edit.alt

        pass

