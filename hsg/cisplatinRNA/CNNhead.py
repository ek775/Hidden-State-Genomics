import torch
import torch.nn as nn


class CNNHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(CNNHead, self).__init__()
        kernel_size = max(int((input_dim) ** 0.2), 2)  # minimum kernel size of 2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(input_dim, input_dim // 8, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.conv2 = nn.Conv1d(input_dim // 8, 64, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.pooling1 = nn.MaxPool1d(kernel_size=kernel_size, stride=6)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, seq_length, feature_dim] -> [batch, feature_dim, seq_length]
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.pooling1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.pooling2(x))
        x = self.fc(x.squeeze(2))
        return x

    def pad_sequence(self, x, max_length):
        """
        Pads the input sequence to the specified maximum length. Assumes input
        shape is (seq_length, embedding_dim) or (seq_length, feature_dim).
        """
        if x.size(0) < max_length:
            padding = torch.zeros((max_length - x.size(0), x.size(1)), device=x.device)
            return torch.cat((x, padding), dim=0)
        return x[:max_length, :]

    @staticmethod
    def from_pretrained(path, device=None) -> "CNNHead":
        """Load a pretrained CNNHead from a file."""
        import io

        try:
            state_dict = torch.load(path)
        except Exception:
            state_dict = torch.load(io.FileIO(path))

        input_dim = state_dict["conv1.weight"].shape[1]
        output_dim = state_dict["fc.weight"].shape[0]
        cnn_head = CNNHead(input_dim=input_dim, output_dim=output_dim)
        cnn_head.load_state_dict(state_dict)
        if device is not None:
            cnn_head.to(device)
        return cnn_head
