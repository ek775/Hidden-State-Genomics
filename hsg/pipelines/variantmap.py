# imports
from hgvs.parser import Parser
from hgvs.dataproviders.uta import connect
from hgvs.assemblymapper import AssemblyMapper
from biocommons.seqrepo import SeqRepo
from hgvs.sequencevariant import SequenceVariant
from hgvs.normalizer import Normalizer
from hgvs.validator import Validator
import re
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class DNAVariantProcessor():

    """
    Class for processing hgvs variant expressions to obtain raw sequences for reference and variant alleles.
    Aggregates biocommons utilities for parsing hgvs nomenclature and retrieving sequences as strings.

    Methods:
    - clean_hgvs
    - parse_variant
    - genomic_sequence_projection
    - determine_variant_type
    - process_del
    - process_snp
    - process_dup
    - retrieve_refseq
    - retrieve_variantseq
    - trim_string_odd
    """

    def __init__(self, assembly:str = "GRCh38.p14", seqrepo_path:str = os.environ["SEQREPO_PATH"]) -> None:
        """
        Start me up with a parser, assembly mapper, and sequence repository.
        """
        # hgvs tools
        self.parser:Parser = Parser()
        self.assembly_mapper:AssemblyMapper = AssemblyMapper(
            connect(), 
            normalize=False, # found this can cause problems with some clingen variants
            assembly_name=assembly, 
            alt_aln_method='blat',
            replace_reference=True
        )
        self.seq_repo:SeqRepo = SeqRepo(seqrepo_path) 


    def clean_hgvs(self, raw_hgvs_expression:str) -> str:

        """
        Processes hgvs expressions with gene name and predicted protein change annotations in parenthesis and removes 
        them for machine readability.
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
        

        Args:

        - hgvs_expression: str

            HGVS expression to be processed.

        - return_exceptions: bool

            If True, exceptions will be returned. If False, None will be returned for invalid or uncertain expressions.

            
        Returns:

        - SequenceVariant object or None

        """

        try:
            parsed_variant = self.parser.parse(hgvs_expression)
            return parsed_variant
         
        except Exception as e: 
            if return_exceptions == True:
                return e
            else:
                parsed_variant = None     


    def genomic_sequence_projection(self, var_obj:SequenceVariant) -> SequenceVariant:

        """
        Uses the Assembly Mapper to project the variant to genomic coordinates and normalize the variant according to HGVS standards.

        Returns a SequenceVariant object.
        1. Determine the coordinate type of the variant (g, c, n, t).
        2. Use the AssemblyMapper to project the variant to genomic coordinates.
        3. If the variant appears to be referenced with a different assembly, AssemblyMapper attempts to convert the reference.
        4. Return the projected variant.
        5. If the coordinate type is not supported, raise a ValueError.
        """

        # type checking
        assert isinstance(var_obj, SequenceVariant), "Input must be a SequenceVariant object"

        # determine the coordinate type of the variant
        coord_type = var_obj.type
        projection = None

        # project the variant to genomic coordinates (normalization set on class initialization)
        if coord_type == "g":
            projection = var_obj
        elif coord_type == "c":
            projection = self.assembly_mapper.c_to_g(var_obj)
        elif coord_type == "n":
            projection = self.assembly_mapper.n_to_g(var_obj)
        elif coord_type == "t":
            projection = self.assembly_mapper.t_to_g(var_obj)
        else:
            raise ValueError(f"Unsupported coordinate type: {coord_type}")

        return projection      
          
        
    def determine_variant_type(self, hgvs_expression: str) -> str:

        """
        Determines the type of variant based on the HGVS expression.

        
        Args:

        - hgvs_expression: str

            HGVS expression to be processed.

            
        Returns:

        One of: ["del", "SNP", "dup", "unknown"]

        """

        if "del" in hgvs_expression:
            return "del"
        elif ">" in hgvs_expression:
            return "SNP"
        elif "dup" in hgvs_expression:
            return "dup"
        else:
            return "unknown"
        

    def process_del(self, hgvs_ref: SequenceVariant, variant_start: int, variant_end: int) -> str:

        """
        Processes a deletion by deleting the specified sequence region.

        
        Args:

        - hgvs_ref: SequenceVariant

            SequenceVariant object representing the target variant.

        - variant_start: int

            Start position of the deletion ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].

        - variant_end: int

            End position of the deletion ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].


        Returns:

        - var_seq: str

            Sequence with the deletion applied.

        """

        del_len = int(variant_end) - (int(variant_start)-1)
        ref_seq, offset = self.retrieve_refseq(hgvs_ref, return_offset=True)
        var_seq = ref_seq[:offset]+ref_seq[(offset+del_len):] # start idx is inclusive, end idx is exclusive
        return(var_seq)
    
#        if int(variant_start) == int(variant_end):
#            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2999 : int(variant_end)+2999]
#            var_seq = var_seq = ref_seq[:2998]+ref_seq[2999:]
#            return(var_seq)

#        elif int(variant_start) != int(variant_end):
#            del_len = int(variant_end) - (int(variant_start)-1)
#            ref_seq = str(self.seq_repo[f"refseq:{hgvs_ref.ac}"])[int(variant_start)-2999 : int(variant_end)+2999]
#            var_seq = ref_seq[:2998]+ref_seq[2998+del_len:]
#            return(var_seq)    

    def process_snp(self, hgvs_ref: SequenceVariant, variant_start: int, variant_end: int) -> str:

        """
        Processes a SNP by replacing the affected base.

        
        Args:

        - hgvs_ref: SequenceVariant

            SequenceVariant object representing the target variant.

        - variant_start: int

            Start position of the SNP ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].

        - variant_end: int

            End position of the SNP ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].

            
        Returns:

        - var_seq: str

            Sequence with the SNP applied.

        """

        ref_seq, offset = self.retrieve_refseq(hgvs_ref, return_offset=True)
        alt_nuc = str(hgvs_ref.posedit.edit.alt)
        var_seq = ref_seq[:offset]+alt_nuc+ref_seq[offset+1:] # start idx is inclusive, end idx is exclusive
        return var_seq

    def process_dup(self, hgvs_ref: SequenceVariant, variant_start: int, variant_end: int) -> str:

        """
        Processes a duplication by duplicating the specified sequence region.

        
        Args:

        - hgvs_ref: SequenceVariant

            SequenceVariant object representing the target variant.

        - variant_start: int

            Start position of the duplication ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].

        - variant_end: int

            End position of the duplication ***Note that this should be a zero-indexed genomic coordinate shifted by [-1].

            
        Returns:

        - var_seq: str

            Sequence with the duplication applied.

        """

        dup_length = int(variant_end) - (int(variant_start)-1)

        ref_seq, offset = self.retrieve_refseq(hgvs_ref, return_offset=True)
        
        var_seq = ref_seq[:offset]+ref_seq[offset:offset+dup_length]+ref_seq[offset:] # start idx is inclusive, end idx is exclusive

        return var_seq


    def retrieve_refseq(self, hgvs_ref:SequenceVariant, return_offset:bool = False) -> str:

        """
        Retrieve reference sequence from seqrepo for a given variant as a string. 

        
        Args:

        - hgvs_ref: SequenceVariant

            SequenceVariant object representing the target variant.

        - return_offset: bool

            If True, the offset for the context window will be returned along with the sequence. This is also the index of the 
            variant start position relative to the context window. For example, if the variant is at position 3000 in the 
            context window, the offset will be 3000. 
            
            However, if the variant occurs near the end of an accession sequence, it may be shifted to avoid "index 
            out of bounds" errors. For example, a variant at genomic coordinate 1755 has only 1754 bases to its left, thus,
            you cannot have 3000 bases to the left of the variant. In this case, the offset will be 1755, indicating that the 
            variant start position at string index position 1755, as opposed to referencing a genomic coordinate. 

            
        Returns:

        The reference sequence as a string [~6000 bases long (tokenizer context window limit)]. 
        
        If return_offset is True, the offset will be returned as well.

        """

        full = str(self.seq_repo[hgvs_ref.ac])

#        print(full[hgvs_ref.posedit.pos.start.base-1], ":", hgvs_ref.posedit.edit.ref) # verify zero index offset
#        print(full[hgvs_ref.posedit.pos.start.base-3:hgvs_ref.posedit.pos.start.base+3]) # verify zero index offset

        # Find offset for context window (avoid index out of bounds)
        dist = 0
        start = max(int(hgvs_ref.posedit.pos.start.base-1)-3000, 0)
        if start == 0:
            dist = abs(int(hgvs_ref.posedit.pos.start.base-1)-3000)
            
        # gimme as much sequence as possible, center the variant in the context window
        end = min(int(hgvs_ref.posedit.pos.end.base-1)+3000+dist, len(full))

        seq_ref = full[start:end]

        offset = 3000-dist # offset for context window

        if return_offset:
            return seq_ref, offset
        else:
            return seq_ref


    def retrieve_variantseq(self, hgvs_ref: SequenceVariant) -> str:

        """
        Modifies obtained refseq sequence to obtain the variant sequence

        Args:
        - hgvs_ref: SequenceVariant

            SequenceVariant object representing the target variant.

        Returns:
        - var_seq: str

            Sequence mutated per HGVS variant expression.
        """

        var_type = self.determine_variant_type(str(hgvs_ref))

        # Correct for zero index offset
        variant_start: int = hgvs_ref.posedit.pos.start.base-1
        variant_end: int = hgvs_ref.posedit.pos.end.base-1

        if var_type == "del":
            return self.process_del(hgvs_ref, variant_start, variant_end)
        elif var_type == "SNP":
            return self.process_snp(hgvs_ref, variant_start, variant_end)
        elif var_type == "dup":
            return self.process_dup(hgvs_ref, variant_start, variant_end)
        else:
            raise ValueError(f"Unsupported variant type: {var_type}")
        

    def trim_string_odd(self, s: str, target_length: int) -> str:

        """
        Trims a sequence to a target length from both ends. 

        If the sequence length is odd, the extra base will be removed from the end of the sequence.

        Args:
        - s: str

            Sequence to be trimmed.
        
        - target_length: int

            Target length for the sequence.
        
            
        Returns:

        The trimmed sequence as a string.
        """
        
        current_length = len(s)
    
        if target_length >= current_length:
            return s 
    
        trim_amount = (current_length - target_length) // 2
        extra = (current_length - target_length) % 2  
    
        return s[trim_amount: -(trim_amount + extra)] if trim_amount > 0 else s
