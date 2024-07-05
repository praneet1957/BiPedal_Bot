import torch
from torch import nn
import torch.nn.functional as F


class valueFunction:
    def __init__(self):
        #input layer is 4 for 4 motors and 2 positional variables
        self.layer1 = nn.Linear(6,16) 
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)
        #output layer is the value function

    def evaluate(self, state):
        # convert state to tensor
        stateTensor = torch.tensor(state, dtype=float)

        activation1 = F.linear(self.layer1(stateTensor))
        activation2 = F.sigmoid(self.layer2(activation1))
        activation3 = F.sigmoid(self.layer3(activation2))
        activation4 = F.linear(self.layer4(activation3))

        return activation4
    


