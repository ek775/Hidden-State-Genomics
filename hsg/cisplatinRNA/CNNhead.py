import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(CNNHead, self).__init__()
        kernel_size = max(int((input_dim)**0.25), 2) # minimum kernel size of 2
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=kernel_size, padding=0.5 * kernel_size)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=kernel_size, padding=0.5 * kernel_size)
        self.fc = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return x