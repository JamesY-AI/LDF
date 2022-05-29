import torch
import torch.nn.functional as F

#define MLP model
class MLP(torch.nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(MLP, self).__init__()
        self.input_size = in_size
        self.hidden1_size = h1_size
        self.hidden2_size = h2_size
        self.output_size = out_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden1_size)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden1_size) #batchnorm layer, test doing before ReLU
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.5) #drop out layer, to try improving accuracy
        self.fc2 = torch.nn.Linear(self.hidden1_size, self.hidden2_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden2_size) #batchnorm layer, test doing before ReLU
        self.drop2 = torch.nn.Dropout(0.5) #drop out layer, to try improving accuracy
        self.fc3 = torch.nn.Linear(self.hidden2_size, self.output_size)
        
    def forward(self, X, mode):
        h1 = self.fc1(X)
        bn1 = self.bn1(h1)
        #drop1 = self.drop1(bn1)
        relu1 = self.relu(bn1)
        h2 = self.fc2(relu1)
        bn2 = self.bn2(h2)
        #drop2 = self.drop2(bn2) 
        relu2 = self.relu(bn2)
        
        if mode == "inference":
            out = self.fc3(relu2)
            out = F.softmax(out, dim=1)
        else:
            out = self.fc3(relu2)
        
        return out
    