import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHead(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim, dropout_rate=0.5):
        super(CNNHead, self).__init__()
        kernel_size = max(int((input_dim)**0.2), 2) # minimum kernel size of 2
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(seq_length, input_dim//2, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.conv2 = nn.Conv1d(input_dim//2, 64, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.pooling1 = nn.MaxPool1d(kernel_size=kernel_size, stride=6)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling2 = nn.MaxPool1d(kernel_size=4, stride=input_dim//self.pooling1.stride)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.pad_sequence(x, max_length=self.seq_length)  # NTv2 tokenizer max length
        x = x.squeeze(0)
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.pooling1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.pooling2(x))
        x = self.fc(torch.mean(x, dim=1))
        return x
    
    def pad_sequence(self, x, max_length):
        """
        Pads the input sequence to the specified maximum length. Assumes input from NTv2-500m Human model with shape (seq_length, embedding_dim),
        or (seq_length, feature_dim) for CNNHead.
        """
        if x.size(0) < max_length:
            padding = torch.zeros((max_length - x.size(0), x.size(1)), device=x.device)
            return torch.cat((x, padding), dim=0)
        return x[:max_length, :]