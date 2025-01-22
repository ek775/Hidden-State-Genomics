import os
import unittest

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Test_EnvVarsSet(unittest.TestCase):

    def test_env_vars_set(self):

        self.assertIsNotNone(os.getenv('CLIN_GEN_CSV'))
        self.assertIsNotNone(os.getenv('CLIN_VAR_CSV'))
        self.assertIsNotNone(os.getenv('NT_MODEL'))

if __name__ == '__main__':
    unittest.main()