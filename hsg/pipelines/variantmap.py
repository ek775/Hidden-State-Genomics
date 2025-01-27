# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant
import re
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()


class DNAVariantProcessor():

    """
    Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles
    """

    def __init__(self, assembly:str = "GRCh37", seqrepo_path:str = os.environ["SEQREPO_PATH"]) -> None:
        # hgvs tools
        self.parser:Parser = Parser()
        self.assembly_mapper:AssemblyMapper = AssemblyMapper(connect(), assembly_name=assembly, alt_aln_method='splign')
        self.seq_repo:SeqRepo = SeqRepo(seqrepo_path) 


    def clean_hgvs(self, raw_hgvs_expression:str) -> str:

        """
        Processes hgvs expressions with gene name and predicted protein change annotations and removes them for machine readability.
        """

        # remove gene annotations
        pattern = r"\([^)]*\)"
        clean_hgvs = re.sub(pattern, '', raw_hgvs_expression)

        # resulting string
        return clean_hgvs.strip()


    def parse_variant(self, hgvs_expression:str, return_exceptions:bool = True ) -> SequenceVariant:

        """
        Uses parser to parse input hgvs_expression.

        If hgvs cannot parse the expression, it will return an exception.

        This behavior can be changed by setting return_exceptions to False, which, will cause invalid or uncertain expressions to return None -- useful during ML training.
        """

        try:
            return self.parser.parse(hgvs_expression)  
         
        except Exception as e: 
            # sometimes we want information, sometimes we just want to move on
            if return_exceptions == True:
                return e
            else:
                return None
        

    def standardize_expression_type(self, hgvs_ref:SequenceVariant, expression_type:str = ["g","m","c","n","r","p"]):

        """
        TODO:
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
        Modifies obtained refseq sequence to obtain the variant sequence
        """

        variant_start: int = hgvs_ref.posedit.pos.start.base
        variant_end: int = hgvs_ref.posedit.pos.end.base

        # some variant types may not have a reference or variant allele (i.e. copy number variants)
        try:
            ref: str = hgvs_ref.posedit.edit.ref
            assert ref is not None
        except:
            ref:str = ''

        try:
            var: str = hgvs_ref.posedit.edit.alt
            assert var is not None
        except:
            var:str = ''

        # ref and var should not both be None
        if ref == None and var == None:
            raise ValueError("Reference and Variant alleles cannot both be None")

        # get refseq
        refseq = self.seq_repo[f"refseq:{hgvs_ref.ac}"]
        varseq = ''

        # modify the refseq to obtain the variant sequence
        # TODO: HGVS extracts locations from hgvs expressions without validation. Need to implement some sort of 
        # coordinate system unification and position validation to reduce mapping errors.
        try:
            assert ref == refseq[variant_start:variant_end+1]
            varseq = refseq[:variant_start] + var + refseq[variant_end:]
        except:
            varseq = None
        
        return varseq

