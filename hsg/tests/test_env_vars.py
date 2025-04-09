import os
import unittest

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Test_EnvVarsSet(unittest.TestCase):

    def test_env_vars_set(self):
        """
        Make sure that all the necessary environment variables are set using a .env file
        """

        self.assertIsNotNone(os.getenv('CLIN_GEN_CSV'))
        self.assertIsNotNone(os.getenv('CLIN_VAR_CSV'))
        self.assertIsNotNone(os.getenv('NT_MODEL'))
        self.assertIsNotNone(os.getenv('GCLOUD_BUCKET'))
        self.assertIsNotNone(os.getenv('SEQREPO_PATH'))
        self.assertIsNotNone(os.getenv('REFSEQ_CACHE'))

if __name__ == '__main__':
    unittest.main()