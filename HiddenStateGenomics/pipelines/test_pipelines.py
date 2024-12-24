# test classes and functions in pipelines module
import unittest

from hgvs.sequencevariant import SequenceVariant

from HiddenStateGenomics.pipelines.variantmap import DNAVariant

class TestDNAVariant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dna_variant = DNAVariant("NM_000551.3:c.1582G>A")

    def test_parse_hgvs(self):
        self.dna_variant.parse_hgvs()
        self.assertIsInstance(self.dna_variant.hgvs_ref, SequenceVariant)

if __name__ == '__main__':
    unittest.main()