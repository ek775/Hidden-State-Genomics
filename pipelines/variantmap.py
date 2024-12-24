# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant

class DNAVariant():
    """Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles"""
    def __init__(self, hgvs_expression:str) -> None:
        self.hgvs_expression:str = hgvs_expression
        self.hgvs_ref: SequenceVariant = None
        self.retrieve_refseq:str = ""
        self.retrieve_variantseq:str = ""

    def parse_hgvs(self) -> None:
        parser = Parser()
        self.hgvs_ref = parser.parse_hgvs_variant(self.hgvs_expression)
    
    def retrieve_refseq(self, assembly:str) -> None:

        # Connect to UTA and align variant to reference
        hdp = connect()
        variant_mapper = AssemblyMapper(hdp, assembly_name=assembly, alt_aln_method='splign')
        var_ref = variant_mapper.c_to_g(self.hgvs_ref)

        # Connect to local seqrepo to obtain reference sequence
        seqrepo = SeqRepo("./src/genome_databases/2024-05-23") # TODO: dynamically set database by config
        sequence_proxy = seqrepo[f"refseq:{var_ref.ac}"]

    def retrieve_variantseq(self) -> None:
        pass