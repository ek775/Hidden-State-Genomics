# test classes and functions in pipelines module
import unittest

from hgvs.sequencevariant import SequenceVariant
import pandas as pd
from tqdm import tqdm

from HiddenStateGenomics.pipelines.variantmap import DNAVariantProcessor

class TestDNAVariant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.clin_gen = pd.read_csv("~/Downloads/erepo.tabbed.txt", header="infer", sep="\t")
        self.worker = DNAVariantProcessor()


    def test_parse_hgvs(self):
        
        invalid_expressions = 0

        for var in tqdm(self.clin_gen["#Variation"]):

            var = str(var)

            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression, return_exceptions=False)

            if var_obj is None:
                invalid_expressions += 1
                continue
            
            self.assertIsInstance(var_obj, SequenceVariant)

        print(f"Unable to process {invalid_expressions} out of {len(self.clin_gen)}")
        print("""This may be due to invalid or uncertain HGVS expressions. See https://hgvs-nomenclature.org/stable/recommendations/summary/ for proper syntax. \n
              Note that the hgvs package does not currently support all HGVS expression types.""")


    def test_retrieve_refseq(self):

        for var in tqdm(self.clin_gen["#Variation"]):

            var = str(var)
            
            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression, return_exceptions=False)

            if var_obj is None:
                continue

            else:
                refseq = self.worker.retrieve_refseq(var_obj)
                self.assertIsInstance(refseq, str)

    
    def test_retrieve_variantseq(self):

        bad_mapping = 0

        for var in tqdm(self.clin_gen["#Variation"]):

            var = str(var)

            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression, return_exceptions=False)
            varseq = ''

            if var_obj is None:
                continue

            else:
                varseq = self.worker.retrieve_variantseq(var_obj)
            
            ###
            if varseq is None:
                bad_mapping += 1
            
            else:
                self.assertIsInstance(varseq, str)

        print(f"Unable to map {bad_mapping} out of {len(self.clin_gen)}")


# making unittest run from command line
if __name__ == '__main__':
    unittest.main()