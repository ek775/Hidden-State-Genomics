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
            except Exception as e:
                print(e)
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

            comparison = difflib.SequenceMatcher(None, refseq, varseq)
            changes = comparison.get_opcodes()

            # test conditions for deletions
            if var_obj.posedit.edit.type == "del":
                try:
                    taga, i1a, i2a, j1a, j2a = changes[0] # deletion is centered in the context window
                    self.assertEqual(taga, "equal")
                    self.assertEqual(i1a, j1a)
                    self.assertEqual(i2a, j2a)

                    # difflib identifies the deletion as a replacement due to our context window
                    tagb, i1b, i2b, j1b, j2b = changes[1]

                    self.assertEqual(tagb, "replace")
                    self.assertEqual(i1b, j1b)

                    # contiguous changes
                    self.assertEqual(i2a, i1b)
                    self.assertEqual(j2a, j1b)

                    # check deletion sizes
                    del_size = var_obj.posedit.pos.end.base - var_obj.posedit.pos.start.base                       
                    self.assertEqual(i2b, j2b + del_size)
                    
                    # match trailing refseq
                    self.assertEqual(varseq[j2b:], refseq[j2b + del_size:])
                        
                except:
                    bad_mapping += 1
                    print(f"Unable to map deletion: {var}")
                    print(changes)
                    continue

            # test conditions for substitutions / SNPs
            if var_obj.posedit.edit.type == "sub":
                try:
                    substitutions = len([i for i in changes if i[0] == "replace"])
                    self.assertEqual(substitutions, 1)
                    self.assertNotIn("delete", [i[0] for i in changes])
                    self.assertNotIn("insert", [i[0] for i in changes])
                except:
                    bad_mapping += 1
                    print(f"Unable to map substitution: {var}")
                    print(changes)
                    continue

            # test conditions for insertions
            if var_obj.posedit.edit.type == "ins":
                try:
                    insertions = len([i for i in changes if i[0] == "insert"])
                    self.assertEqual(insertions, 1)
                    self.assertNotIn("replace", [i[0] for i in changes])
                    self.assertNotIn("delete", [i[0] for i in changes])

                    # check insertion size
                    ins_size = var_obj.posedit.pos.start.base - var_obj.posedit.pos.end.base
                    insertions = [i for i in changes if i[0] == "insert"]
                    i1, i2, j1, j2 = insertions[0][1:]
                    self.assertEqual(i2, i1 + ins_size)
                except:
                    bad_mapping += 1
                    print(f"Unable to map insertion: {var}")
                    print(changes)
                    continue

            # test conditions for duplications
            if var_obj.posedit.edit.type == "dup":
                try:
                    duplication = [i for i in changes if i[0] == "insert"] # tagged as insertion
                    i1, i2, j1, j2 = duplication[0][1:]
                    dup_size = j2-j1
                    stated_dup_size = var_obj.posedit.pos.end.base - var_obj.posedit.pos.start.base
                    self.assertEqual(dup_size, stated_dup_size)
                    self.assertEqual(varseq[j1:j2], refseq[i1-dup_size:i1])
                except:
                    bad_mapping += 1
                    print(f"Unable to map duplication: {var}")
                    print(changes)
                    continue

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