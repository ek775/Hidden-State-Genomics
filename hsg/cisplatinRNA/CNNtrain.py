import torch
import torch.nn as nn
import torch.nn.functional as F

from hsg.cisplatinRNA.CNNhead import CNNHead
from tap import tapify
from tqdm import tqdm
import os