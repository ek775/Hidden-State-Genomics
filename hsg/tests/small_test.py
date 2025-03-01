# test classes and functions in pipelines module
import unittest
import os

from hgvs.sequencevariant import SequenceVariant
import pandas as pd
from tqdm import tqdm

from hsg.pipelines.variantmap import DNAVariantProcessor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class TestDNAVariant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.clin_gen = pd.read_csv(os.environ["CLIN_GEN_CSV"], header="infer", delimiter="\t")
        self.worker = DNAVariantProcessor()
        
    def test_parse_hgvs(self):

        print("Testing HGVS Parsing")
        print("====================")

        invalid_expressions = 0

        for var in tqdm(self.clin_gen["HGVS Expressions"]):

            var = str(var)
            HGVS = str(var).split()[1][:-1]

            var_obj = self.worker.parse_variant(HGVS, return_exceptions=False)  

            if var_obj is None:
                invalid_expressions += 1
                continue
            
            self.assertIsInstance(var_obj, SequenceVariant)

        print(f"Unable to process {invalid_expressions} out of {len(self.clin_gen)}")
        print("""This may be due to invalid or uncertain HGVS expressions. See https://hgvs-nomenclature.org/stable/recommendations/summary/ for proper syntax. \n
              Note that the hgvs package does not currently support all HGVS expression types.""")
        print("====================")

    def test_retrieve_refseq(self):

        print("Testing RefSeq Retrieval")
        print("========================")

        bad_mapping = 0

        for var in tqdm(self.clin_gen["HGVS Expressions"]):

            var = str(var)
            HGVS = str(var).split()[1][:-1]

            if ":c." in HGVS:
                bad_mapping += 1
                continue

            var_obj = self.worker.parse_variant(HGVS, return_exceptions=False)

            if var_obj is None:
                continue

            else:
                refseq = self.worker.retrieve_refseq(var_obj)
            
            if refseq is None:
                bad_mapping += 1
                
            else:
                self.assertIsInstance(refseq, str)
        
        print("========================")
        print(f"Unable to map {bad_mapping} out of {len(self.clin_gen)}")
        print("RefSeq Retrieval Passed")

    
    def test_retrieve_variantseq(self):

        print("Testing Variant Sequence Retrieval")
        print("===================================")

        bad_mapping = 0

        for var in tqdm(self.clin_gen["HGVS Expressions"]):

            var = str(var)
            HGVS = str(var).split()[1][:-1]

            if ":c." in HGVS:
                bad_mapping += 1
                continue

            var_obj = self.worker.parse_variant(HGVS, return_exceptions=False)
            varseq = ''

            if var_obj is None:
                continue

            else:
                varseq = self.worker.retrieve_variantseq(var_obj)

            if varseq is None:
                bad_mapping += 1
            
            else:
                self.assertIsInstance(varseq, str)

        print(f"Unable to map {bad_mapping} out of {len(self.clin_gen)}")
        print("===================================")


# making unittest run from command line
if __name__ == '__main__':
    unittest.main()