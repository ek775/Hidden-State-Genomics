# test classes and functions in pipelines module
import unittest

from hgvs.sequencevariant import SequenceVariant
import pandas as pd

from HiddenStateGenomics.pipelines.variantmap import DNAVariantProcessor

class TestDNAVariant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.clin_gen = pd.read_csv("~/Downloads/erepo.tabbed.txt", header="infer", sep="\t")
        self.worker = DNAVariantProcessor()

    def test_parse_hgvs(self):
        for var in self.clin_gen["#Variation"]:
            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression)
            self.assertIsInstance(var_obj, SequenceVariant)

    def test_retrieve_refseq(self):
        for var in self.clin_gen["#Variation"]:
            expression = self.worker.clean_hgvs(var)
            var_obj = self.worker.parse_variant(expression)
            refseq = self.worker.retrieve_refseq(var_obj)
            self.assertIsInstance(refseq, str)

if __name__ == '__main__':
    unittest.main()