# test classes and functions in pipelines module
import unittest
import os

from hgvs.sequencevariant import SequenceVariant
import pandas as pd
from tqdm import tqdm
import difflib

from hsg.pipelines.variantmap import DNAVariantProcessor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class TestDNAVariant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.clin_gen = pd.read_csv(os.environ["CLIN_GEN_CSV"], header="infer", sep="\t")
        self.worker = DNAVariantProcessor()


    def test_parse_hgvs(self):

        print("Testing HGVS Parsing")
        print("====================")

        invalid_expressions = 0

        for var in tqdm(self.clin_gen["#Variation"]):

            var = str(var)
            HGVS = self.worker.clean_hgvs(var)

            var_obj = self.worker.parse_variant(HGVS, return_exceptions=False)  

            if var_obj is None:
                invalid_expressions += 1
                continue
            
            self.assertIsInstance(var_obj, SequenceVariant)

        print(f"Unable to process {invalid_expressions} out of {len(self.clin_gen)}")
        print("This may be due to invalid or uncertain HGVS expressions. See https://hgvs-nomenclature.org/stable/recommendations/summary/ for proper syntax.")
        print("Note that the hgvs package does not currently support all HGVS expression types.")
        print("====================")

    
    def test_sequence_retrieval(self):

        print("Testing Reference Sequence Retrieval & Variant Sequence Mapping")
        print("===================================")

        # current mapping of all 9025 clin gen variants takes hours
        # so we will test a subset of 1000
        sample = self.clin_gen["HGVS Expressions"].sample(n=1000)

        # pick out g. nomenclature, if we have been graced with it
        rand_1000 = []
        for exp_set in sample:
            exp_set = exp_set.split(",")
            for i, exp in enumerate(exp_set):
                if ":g." in exp:
                    rand_1000.append(exp.strip())
                    break
                if i == len(exp_set) - 1:
                    rand_1000.append(exp.strip())
                else:
                    continue

        # count failures
        bad_parse = 0
        bad_retrieval = 0
        bad_projection = 0 
        bad_mapping = 0

        for var in tqdm(rand_1000):

            var = str(var)
            HGVS = self.worker.clean_hgvs(var)

            # parse and project the variant
            var_obj = self.worker.parse_variant(HGVS, return_exceptions=False)
            refseq = ''
            varseq = ''

            if var_obj is None:
                bad_parse += 1
                continue

            else:
                try:
                    var_obj = self.worker.genomic_sequence_projection(var_obj)
                except:
                    bad_projection += 1
                    continue
                
            # get our sequences
            try:
                refseq = self.worker.retrieve_refseq(var_obj)
            except:
                bad_retrieval += 1
                continue
            try:    
                varseq = self.worker.retrieve_variantseq(var_obj)
            except:
                bad_mapping += 1
                continue

            # type checking
            self.assertIsInstance(varseq, str)
            self.assertIsInstance(refseq, str)

            # check that the variant sequence is not the same as the reference sequence
            self.assertNotEqual(varseq, refseq)

            ###############################################
            # check mutations for correctness using difflib
            ###############################################

            diff = [i for i in difflib.ndiff(refseq, varseq) if i[0] != ' ']

            # test conditions for deletions
            if var_obj.posedit.edit.type == "del":
                self.assertTrue("-" in ''.join(diff))
                self.assertFalse("+" in ''.join(diff))
            
            # test conditions for substitutions / SNPs
            if var_obj.posedit.edit.type == "sub":
                self.assertTrue("+" in ''.join(diff))
                self.assertTrue("-" in ''.join(diff))

                additions = len([i for i in diff if i[0] == "+"])
                deletions = len([i for i in diff if i[0] == "-"])

                self.assertEqual(additions, deletions)

            # test conditions for insertions
            if var_obj.posedit.edit.type == "ins":
                self.assertTrue("+" in ''.join(diff))
                self.assertFalse("-" in ''.join(diff))

            # test conditions for duplications
            if var_obj.posedit.edit.type == "dup":
                self.assertTrue("+" in ''.join(diff))
                self.assertFalse("-" in ''.join(diff))

            # test conditions for delins
            if var_obj.posedit.edit.type == "delins":
                self.assertTrue("+" in ''.join(diff))
                self.assertTrue("-" in ''.join(diff))
            
            # other / complex variants not yet supported, should have thrown error earlier
            else:
                continue
            

        print(f"Unable to parse {bad_parse} out of {len(rand_1000)}")
        print(f"Unable to project {bad_projection} out of {len(rand_1000) - bad_parse}")
        print(f"Unable to retrieve {bad_retrieval} out of {len(rand_1000) - bad_parse - bad_projection}")
        print(f"Unable to map {bad_mapping} out of {len(rand_1000) - bad_parse - bad_projection - bad_retrieval}")
        print(f"Total failures: {bad_parse + bad_projection + bad_retrieval + bad_mapping} / {len(rand_1000)}")
        print("===================================")


# making unittest run from command line
if __name__ == '__main__':
    unittest.main()