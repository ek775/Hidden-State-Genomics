import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHead(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim, dropout_rate=0.5):
        super(CNNHead, self).__init__()
        kernel_size = max(int((input_dim)**0.2), 2) # minimum kernel size of 2
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv1d(seq_length, input_dim//2, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.conv2 = nn.Conv1d(input_dim//2, 64, kernel_size=kernel_size, padding=int(0.5 * kernel_size))
        self.pooling1 = nn.MaxPool1d(kernel_size=kernel_size, stride=6)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling2 = nn.MaxPool1d(kernel_size=4, stride=input_dim//self.pooling1.stride)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
#        print(f"Input shape: {x.shape}")
        x = nn.ReLU()(self.conv1(x))
#        print(f"After conv1 shape: {x.shape}")
        x = nn.ReLU()(self.conv2(x))
#        print(f"After conv2 shape: {x.shape}")
        x = nn.ReLU()(self.pooling1(x))
#        print(f"After pooling1 shape: {x.shape}")
        x = self.dropout(x)
        x = nn.ReLU()(self.pooling2(x))
#        print(f"After pooling2 shape: {x.shape}")
        x = self.fc(torch.mean(x, dim=2)) # global average pooling
#        print(f"After fc shape: {x.shape}")
        x = F.softmax(x, dim=1)
#        print(f"Output shape: {x.shape}")
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
    
    def from_pretrained(path, device=None) -> "CNNHead":
        """
        Load a pretrained CNNHead from a file.
        """
        import io
        try:
            state_dict = torch.load(path)
#            print("Loaded model directly")
        except:
            state_dict = torch.load(io.FileIO(path))
#            print("Loaded model from buffer")

        input_dim = state_dict["conv1.weight"].shape[0]
        seq_length = state_dict["conv1.weight"].shape[2]
        output_dim = state_dict["fc.weight"].shape[0]
        cnn_head = CNNHead(input_dim=input_dim, seq_length=seq_length, output_dim=output_dim)
        cnn_head.load_state_dict(state_dict)
        if device is not None:
            cnn_head.to(device)
        return cnn_head