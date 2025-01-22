from hsg.sae.dictionary import Dictionary, AutoEncoder, IdentityDict
from hsg.sae.interleave import get_submodule, intervention_output

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers.models.esm import EsmTokenizer
import torch
import nnsight
from nnsight import NNsight

from tqdm import tqdm
import unittest
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class Test_NT_2_5B_MultiSpecies(unittest.TestCase):

    def test_tokenizer_load(self):

        print("Testing Tokenizer Load")
        print("======================")
        tokenizer = AutoTokenizer.from_pretrained(os.environ["NT_MODEL"])
        self.assertIsInstance(tokenizer, EsmTokenizer)
        print("======================")
        print("Tokenizer Load Passed")
    

    def test_model_load(self):

        print("Testing Model Load")
        print("==================")
        model = AutoModelForMaskedLM.from_pretrained(os.environ["NT_MODEL"])
        self.assertIsInstance(model, EsmForMaskedLM)
        print("==================")
        print("Model Load Passed")


    def test_dummy_encode(self):

        print("Testing Dummy Encoding")
        print("======================")
        # Import the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(os.environ["NT_MODEL"])
        model = AutoModelForMaskedLM.from_pretrained(os.environ["NT_MODEL"])

        # Choose the length to which the input sequences are padded. By default, the 
        # model max length is chosen, but feel free to decrease it as the time taken to 
        # obtain the embeddings increases significantly with it.
        max_length = tokenizer.model_max_length

        # Create a dummy dna sequence and tokenize it
        sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
        token_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

        # Compute the embeddings
        attention_mask = token_ids != tokenizer.pad_token_id
        torch_outs = model(
            token_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Compute sequences embeddings
        embeddings = torch_outs['hidden_states'][-1].detach().numpy()

        # Check that the embeddings have the correct shape
        # Note that 2.5b model has 2560 for neuron width
        self.assertEqual(embeddings.shape, (2, 1000, 1280))

        # Add embed dimension axis
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        self.assertEqual(mean_sequence_embeddings.shape, (2, 1280)) # note that 2.5b model has 2560 for neuron width

        print("======================")
        print("Dummy Encoding Passed")


    def test_nnsight_interleave(self):

        print("Testing NNsight Interleave")
        print("==========================")
        # load huggingface model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["NT_MODEL"])
        model = AutoModelForMaskedLM.from_pretrained(os.environ["NT_MODEL"])

        max_length = tokenizer.model_max_length
        token_ids = tokenizer.encode_plus("ATTCCGATTCCGATTCCG", return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]
        mask = token_ids != tokenizer.pad_token_id

        # test access to each layer of model
        for i in tqdm(range(len(model.esm.encoder.layer))):
            access = intervention_output(
                model=model,
                tokens=token_ids,
                attention_mask=mask,
                patch_layer=i,
            )
            self.assertEqual(len(access), 2)    

        print("==========================")
        print("NNsight Interleave Passed")            

        
class Test_Dictionary_Objects(unittest.TestCase):

    def test_autoencoder_instance(self):

        print("Testing AutoEncoder Initialization")
        print("=================================")

        autoencoder = AutoEncoder(activation_dim=10, dict_size=10)
        self.assertIsInstance(autoencoder, AutoEncoder)
        self.assertEqual(autoencoder.activation_dim, 10)
        self.assertEqual(autoencoder.dict_size, 10)

        print("=================================")
        print("AutoEncoder Initialization Passed")