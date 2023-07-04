import unittest
import io
import sys
from utils import print_ascii

class TestPrintAscii(unittest.TestCase):
    
    def test_output(self):
        # Capture the output printed to stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        print_ascii()
        sys.stdout = sys.__stdout__
        
        # The expected output as a string
        expected_output = """      _____                     __  __ _       
     / ____|                   |  \/  | |      
    | |  __ _ __ ___  ___ _ __ | \  / | |      
    | | |_ | '__/ _ \/ _ \ '_ \| |\/| | |      
    | |__| | | |  __/  __/ | | | |  | | |____  
     \_____|_|  \___|\___|_| |_|_|  |_|______| 
     Creating more efficient and sustainable   
           Machine Learning Pipelines          
                                               \n"""

        # Compare the captured output to the expected output
        self.assertEqual(captured_output.getvalue(), expected_output)

if __name__ == '__main__':
    unittest.main()
