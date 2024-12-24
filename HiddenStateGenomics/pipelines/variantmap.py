# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant

def clean_hgvs_expression(hgvs_expression:str) -> str:

    """
    Given names and hgvs expressions in clinvar dataset include annotations that cause issues for hgvs parser.
    
    This function removes these annotations from the hgvs expressions and generates corrected hgvs expressions.
    """

    pass



class DNAVariant():

    """
    Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles
    """

    def __init__(self, hgvs_expression:str) -> None:
        # params
        self.hgvs_expression:str = hgvs_expression
        self.assembly:str = "GRCh37" # TODO: dynamically set assembly 

        # hgvs tools
        self.parser:Parser = Parser()
        self.assembly_mapper:AssemblyMapper = AssemblyMapper(connect(), assembly_name=self.assembly, alt_aln_method='splign')
        self.seq_repo:SeqRepo = SeqRepo("./genome_databases/2024-05-23") # TODO: dynamically set database

        # the important stuff
        self.hgvs_ref: SequenceVariant = self.parser.parse_hgvs_variant(self.hgvs_expression)        
        self.ref_seq: str = ""
        self.var_seq: str = ""
    

    def standardize_expression_type(self, expression_type:str = ["g","m","c","n","r","p"]):

        """
        Standardize hgvs types for proper mapping of refseqs and generation of variant sequences.

        Different hgvs types utilize different coordinate systems and may use non-contiguous indices, 
        making it difficult to properly modify refseqs to obtain variant sequences.
        """

        pass


    def retrieve_refseq(self, assembly:str) -> None:
        """
        Retrieve reference sequence from seqrepo.
        """
        var_ref = self.assembly_mapper.c_to_g(self.hgvs_ref) # TODO: remove in favor of standardized expression and loci
        sequence_proxy = self.seq_repo[f"refseq:{var_ref.ac}"]
        pass


    def retrieve_variantseq(self) -> None:
        pass

