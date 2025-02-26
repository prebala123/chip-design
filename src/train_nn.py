import numpy as np
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from pyg_dataset import NetlistDataset

dataset = NetlistDataset(data_dir="../data/superblue", load_pd = True, load_pe = True, pl = True, processed = True, load_indices=None)

node_features = []
node_demand = []
for i in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]:
    node_features.append(dataset[i].node_features.cpu().numpy())
    node_demand.append(dataset[i].node_demand.cpu().numpy())

train_features = np.concatenate(node_features)
train_demand = np.concatenate(node_demand)

test_features = []
test_demand = []
for i in [4, 5]:
    test_features.append(dataset[i].node_features.cpu().numpy())
    test_demand.append(dataset[i].node_demand.cpu().numpy())

test_features = np.concatenate(test_features)
test_demand = np.concatenate(test_demand)

# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)
# train_demand = (train_demand - np.mean(train_demand)) / np.std(train_demand)
# test_demand = (test_demand - np.mean(test_demand)) / np.std(test_demand)

num_features = train_features.shape[1]
hidden_dim = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
learning_rate = 0.0001
batch_size = 512


X_train = torch.Tensor(train_features).to(device)
y_train = torch.Tensor(train_demand.reshape(-1, 1)).to(device)
X_test = torch.Tensor(test_features).to(device)
y_test = torch.Tensor(test_demand.reshape(-1, 1)).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_train_batches = len(train_dataloader)

test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
num_test_batches = len(test_dataloader)

model = nn.Sequential(
    nn.Linear(num_features, hidden_dim),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    # nn.Linear(hidden_dim, hidden_dim),
    # nn.LeakyReLU(0.01),
    # nn.Dropout(0.3),
    nn.Linear(hidden_dim, 1)
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.0001)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filepath = f"../results/baselines/{timestamp}_baseline.csv"
with open(filepath, 'w') as f:
    f.write('Epoch,TrainLoss,ValidLoss,Time\n')

start_time = time.time()
for epoch in range(epochs):
    model.train()
    train_loss = 0
    valid_loss = 0
    for batch_X, batch_y in tqdm(train_dataloader):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.float()) 
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    for test_X, test_y in tqdm(test_dataloader):
        y_pred = model(test_X)
        mse = criterion(y_pred, test_y)
        valid_loss += mse.item()

    train_loss /= num_train_batches
    valid_loss /= num_test_batches
    cur_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Time: {cur_time:.2f}\n")
    with open(filepath, 'a') as f:
        f.write(f'{epoch+1},{train_loss},{valid_loss},{cur_time}\n')