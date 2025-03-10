import os
import shutil
import numpy as np
import pickle
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collections import defaultdict
from pyg_dataset_sub import NetlistDataset as NetlistDatasetSub

dataset = NetlistDatasetSub(data_dir=os.path.join('..', '..', 'data', 'superblue'), load_pe = True, load_pd = False, pl = True, processed = False, load_indices=None, density = True)

print(dataset[0])
