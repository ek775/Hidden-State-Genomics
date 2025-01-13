from HiddenStateGenomics.sae.dictionary import Dictionary, AutoEncoder, IdentityDict
from HiddenStateGenomics.sae.interleave import get_submodule, intervention_output

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers.models.esm import EsmTokenizer
import torch
import nnsight
from nnsight import NNsight

from tqdm import tqdm
import unittest


class Test_NT_2_5B_MultiSpecies(unittest.TestCase):

    def test_tokenizer_load(self):

        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        self.assertIsInstance(tokenizer, EsmTokenizer)
    

    def test_model_load(self):

        model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        self.assertIsInstance(model, EsmForMaskedLM)


    def test_dummy_encode(self):

        # Import the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

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
        self.assertEqual(embeddings.shape, (2, 1000, 2560))

        # Add embed dimension axis
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)

        # Compute mean embeddings per sequence
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
        self.assertEqual(mean_sequence_embeddings.shape, (2, 2560))

    def test_nnsight_interleave(self):

        # load huggingface model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

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

        
class Test_Dictionary_Objects(unittest.TestCase):

    def test_autoencoder_instance(self):

        autoencoder = AutoEncoder(activation_dim=10, dict_size=10)
        self.assertIsInstance(autoencoder, AutoEncoder)
        self.assertEqual(autoencoder.activation_dim, 10)
        self.assertEqual(autoencoder.dict_size, 10)