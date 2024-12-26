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

            if var is not str:
                continue

            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression)

            if var_obj is not None:
                self.assertIsInstance(var_obj, SequenceVariant)

            else:
                invalid_expressions += 1

        print(f"Found {invalid_expressions} invalid expressions out of {len(self.clin_gen)}")


    def test_retrieve_refseq(self):

        for var in tqdm(self.clin_gen["#Variation"]):

            if var is not str:
                continue
            
            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression)

            if var_obj is not None:
                refseq = self.worker.retrieve_refseq(var_obj)
                self.assertIsInstance(refseq, str)
            else:
                continue


# making unittest run from command line
if __name__ == '__main__':
    unittest.main()