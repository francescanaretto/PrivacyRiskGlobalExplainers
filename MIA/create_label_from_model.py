import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import tensor, from_numpy
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
HOMEDIR = "../adult/"
dir = "original"
classifier = "nn"
mode = "adult"
kind = "original"
class_name = "class"
data_kind = 'stat'
num_features = 108
device = torch.device('cpu')

class MyDataset(Dataset):
    def __init__(self, xy):
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1]).type(torch.float)
        self.y_data = from_numpy(xy[:, [-1]]).type(torch.LongTensor)
        self.y_data = torch.squeeze(self.y_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
class MyNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = 0

        self.lin1 = torch.nn.Linear(self.num_features, 150)
        self.lin2 = torch.nn.Linear(50, 50)
        self.lin3 = torch.nn.Linear(50, 50)

        self.lin4 = torch.nn.Linear(150, 150)

        self.lin5 = torch.nn.Linear(50, 50)
        self.lin6 = torch.nn.Linear(50, 50)
        self.lin10 = torch.nn.Linear(150, self.num_classes)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, xin):
        self.layers = 0

        x = F.relu(self.lin1(xin))
        self.layers += 1

        # x = F.relu(self.lin2(x))
        # self.layers += 1
        for y in range(8):
            x = F.relu(self.lin4(x))
            self.layers += 1

        x = self.dropout(x)

        x = F.relu(self.lin10(x))
        self.layers += 1
        return x

class NetCopy(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(236, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 30)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = self.fc4(self.drop(x))
        return F.softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ### FOR XAVIER INITIALIZATION
        self.fc1 = nn.Linear(30, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc1.weight)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc2.weight)
        self.drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)  # fc stays for 'fully connected'
        nn.init.xavier_normal_(self.fc3.weight)
        self.drop = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 15)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(self.drop(x))
        return F.softmax(x, dim=1)


filename: str = f"{HOMEDIR}{dir}/{classifier}_{mode}_{kind}.sav"
if classifier == 'rf':
    bb = pickle.load(open(filename, 'rb'))
else:
    bb = MyNet(num_features, 2).to(device)
    bb.load_state_dict(torch.load(HOMEDIR+'blackbox/nn_'+mode+'_original'))
    bb.eval()

print(bb)
filename: str = f"{HOMEDIR}data/{mode}_{data_kind}_shadow.csv"
infile = open(filename,'rb')
new_dict = pickle.load(infile)
print('this is the new el ', new_dict[0].shape)
dataset_shadow = pd.DataFrame(new_dict[0])
print('dataset shadow ', dataset_shadow.shape)
train_set1 = MyDataset(dataset_shadow.values)
train_loader = DataLoader(dataset=train_set1,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
#dataset_shadow = pd.read_csv(filename, index_col=0)
#dataset_shadow.pop(class_name)
#dataset_shadow.pop('UserID')
print('this is the dataset ', dataset_shadow)
if classifier == 'rf':
    labels_bb = bb.predict(dataset_shadow)
else:
    with torch.no_grad():
        labels_bb = []
        for inputs in train_loader:
            print('la shape dell input', inputs.shape)
            inputs = inputs.to(device)
            output = bb(inputs)
            print('output ', output)
            pred = output.max(1, keepdim=True)[1]
            temp = pred.squeeze().numpy()
            for i in temp:
                labels_bb.append(i)

print(len(labels_bb), dataset_shadow.shape)
dataset_shadow[class_name] = labels_bb
print(dataset_shadow.head())
filename: str = f"{HOMEDIR}data/{mode}_{data_kind}_{kind}_{classifier}_shadow_labelled.csv"
dataset_shadow.to_csv(filename)
#pickle.dump(dataset_shadow, open(filename, 'wb'))
