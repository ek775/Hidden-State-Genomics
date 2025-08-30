import torch
import math

class Vectorizer():
    def __init__(self, dict_size:int = 4105, vector_size:int = 1280, existing_table_path=None):
        """
        Create 1280 dimensional vectors from token IDs and add positional encoding (sinusoidal). NTv2 500m-human-ref has a vocabulary 
        of 4105 tokens that we need to represent, thus, by we use a binary encoding scheme to identify the tokens while keeping the
        vector length identical to NTv2's encoder embeddings.
        """
        self.dict_size = dict_size
        self.vector_size = vector_size
        if not existing_table_path:
            self.encoding_table = self.create_encoding_table()
        else:
            self.encoding_table = self.load_encoding_table(existing_table_path)

    def create_encoding_table(self):
        """
        Generate a randomized encoding table that will map the vocabulary (dict_size)
        of token IDs to their corresponding n-dimensional vectors.
        """
        table = torch.randint(low=0, high=2, size=(self.dict_size,self.vector_size), dtype=torch.float32)
        table = torch.unique(table, dim=0)
        while len(table) < self.dict_size:
            diff = self.dict_size - len(table)
            fill = torch.randint(low=0, high=2, size=(diff,self.vector_size), dtype=torch.float32)
            table = torch.cat((table, fill), dim=0)
            table = torch.unique(table, dim=0)

        table = {i: vec for i, vec in enumerate(table)}
        return table

    def save_encoding_table(self, file_path: str):
        """
        Save the encoding table to a file.
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.encoding_table, f)

    def load_encoding_table(self, file_path: str):
        """
        Load the encoding table from a file.
        """
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        """
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_len, d_model, dtype=torch.float32)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def vectorize_tokens(self, token_ids: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Creates 1280 dimensional vectors from token IDs and add positional encoding (sinusoidal).
        """
        batch = []
        for sequence in token_ids:
            tok_base_vecs = [self.encoding_table[int(tok_id)] for tok_id in sequence]
            tensor = torch.stack(tok_base_vecs)
            tensor = tensor + self.positional_encoding(tensor.size(0), tensor.size(1))
            batch.append(tensor)

        return batch