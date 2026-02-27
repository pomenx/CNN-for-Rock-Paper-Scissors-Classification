
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 50 * 75 , 32)
        self.fc2 = nn.Linear(32, 3)  # Assuming 3 classes for Rock-Paper-Scissors

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), kernel_size=2, stride=2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), kernel_size=2, stride=2)
        out = out.view(-1, 4 * 50 * 75)  # Flatten the output for the fully connected layer
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

# model = Net()
# numel_list = [p.numel() for p in model.parameters()]
# print('Net: ', sum(numel_list), numel_list)
# print(model)

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 25 * 37 , 32)
        self.fc2 = nn.Linear(32, 3)  # Assuming 3 classes for Rock-Paper-Scissors

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), kernel_size=2, stride=2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), kernel_size=2, stride=2)
        out = out.view(-1, 4 * 25 * 37)  # Flatten the output for the fully connected layer
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
# model = DeepNet()
# numel_list = [p.numel() for p in model.parameters()]
# print('DeepNet: ', sum(numel_list), numel_list)
# print(model)

class NetDropout(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.3)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.3)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv3_dropout = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(4 * 25 * 37, 32)
        self.fc2 = nn.Linear(32, 3)  # Assuming 3 classes for Rock-Paper-Scissors
    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), kernel_size=2, stride=2)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = self.conv2_dropout(out)
        out = F.max_pool2d(torch.relu(self.conv3(out)), kernel_size=2, stride=2)
        out = self.conv3_dropout(out)
        out = out.view(-1, 4 * 25 * 37)  # Flatten the output for the fully connected layer
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
# model = NetDropout()
# numel_list = [p.numel() for p in model.parameters()]
# print('NetDropout: ', sum(numel_list), numel_list)
# print(model)