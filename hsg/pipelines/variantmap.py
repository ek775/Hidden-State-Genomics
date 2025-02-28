# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant
import re
import os


class DNAVariantProcessor():

    """
    Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles
    """

    def __init__(self, assembly:str = "GRCh37", seqrepo_path:str = os.environ["SEQREPO_PATH"]) -> None:
        self.parser:Parser = Parser()
        self.assembly_mapper:AssemblyMapper = AssemblyMapper(connect(), assembly_name=assembly, alt_aln_method='splign')
        self.seq_repo:SeqRepo = SeqRepo(seqrepo_path) 


    def parse_variant(self, hgvs_expression:str, return_exceptions:bool = True ) -> SequenceVariant:

        """
        Uses parser to parse input hgvs_expression.

        If hgvs cannot parse the expression, it will return an exception.

        This behavior can be changed by setting return_exceptions to False, which, will cause invalid or uncertain expressions to return None -- useful during ML training.
        """

        try:
            parsed_variant = self.parser.parse(hgvs_expression)
            return parsed_variant
         
        except Exception as e: 
            if return_exceptions == True:
                return e
            else:
                parsed_variant = None
                
            
        
    def determine_variant_type(self, hgvs_expression: str) -> str:
        """
        Determines the type of variant based on the HGVS expression.
        """
        if "del" in hgvs_expression:
            return "del"
        elif ">" in hgvs_expression:
            return "SNP"
        elif "dup" in hgvs_expression:
            return "dup"
        else:
            return "unknown"
        

    def process_del(self, hgvs_ref: str, variant_start: int, variant_end: int) -> str:
        """
        Processes a deletion by deleting the specified sequence region.
        """
        if int(variant_start) == int(variant_end):
            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2999 : int(variant_end)+2999]
            var_seq = var_seq = ref_seq[:2998]+ref_seq[2999:]
            return(var_seq)

        elif int(variant_start) != int(variant_end):
            del_len = int(variant_end) - (int(variant_start)-1)
            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2999 : int(variant_end)+2999]
            var_seq = ref_seq[:2998]+ref_seq[2998+del_len:]
            return(var_seq)
        

    def process_snp(self, hgvs_ref: str, variant_start: int, variant_end: int) -> str:
        """
        Processes a SNP by replacing the affected base.
        """
        ref_seq = self.retrieve_refseq(hgvs_ref)
        alt_nuc = str(hgvs_ref.posedit.edit.alt)
        var_seq = ref_seq[:2998]+alt_nuc+ref_seq[2999:]
        return var_seq
    

    def process_dup(self, hgvs_ref: str, variant_start: int, variant_end: int) -> str:
        """
        Processes a duplication by duplicating the specified sequence region.
        """
        if int(variant_start) == int(variant_end):
            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2998 : int(variant_end)+2998]
            var_seq = ref_seq[:2998]+ref_seq[2997]+ref_seq[2998:]
            return ref_seq
        elif int(variant_start) != int(variant_end):
            dup_len = int(variant_end) - (int(variant_start)-1)
            half_dup_len = dup_len/2
            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start-(2999+half_dup_len)) : int(variant_end+(2999-half_dup_len))]
            mid_ref = int(len(ref_seq)/2)
            dup_seq = ref_seq[mid_ref:mid_ref+dup_len]
            var_seq = ref_seq[:mid_ref+dup_len]+dup_seq+ref_seq[mid_ref+dup_len:]
            trimed_var_seq = self.trim_string_odd(var_seq, 5997)
            return trimed_var_seq


    def retrieve_refseq(self, hgvs_ref:SequenceVariant) -> str:

        """
        Retrieve reference sequence from seqrepo.
        """

        variant_start: int = hgvs_ref.posedit.pos.start.base
        variant_end: int = hgvs_ref.posedit.pos.end.base
        seq_ref = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2999 : int(variant_end)+2998]
    
        return seq_ref


    def retrieve_variantseq(self, hgvs_ref: SequenceVariant) -> str:

        """
        Modifies obtained refseq sequence to obtain the variant sequence
        """
        var_type = self.determine_variant_type(str(hgvs_ref))

        variant_start: int = hgvs_ref.posedit.pos.start.base
        variant_end: int = hgvs_ref.posedit.pos.end.base

        if var_type == "del":
            return self.process_del(hgvs_ref, variant_start, variant_end)
        elif var_type == "SNP":
            return self.process_snp(hgvs_ref, variant_start, variant_end)
        elif var_type == "dup":
            return self.process_dup(hgvs_ref, variant_start, variant_end)
        else:
            raise ValueError(f"Unsupported variant type: {var_type}")
        

    def trim_string_odd(self, s: str, target_length: int) -> str:
        current_length = len(s)
    
        if target_length >= current_length:
            return s 
    
        trim_amount = (current_length - target_length) // 2
        extra = (current_length - target_length) % 2  
    
        return s[trim_amount: -(trim_amount + extra)] if trim_amount > 0 else s
