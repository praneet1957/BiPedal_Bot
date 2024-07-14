import torch
from torch import nn
import torch.nn.functional as F


class valueFunction(nn.Module):
    def __init__(self):
        #input layer is 4 for 4 motors and 2 positional variables and 1 velocity
        super(valueFunction, self).__init__()
        self.layer1 = nn.Linear(7,16) 
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)
        #output layer is the value function

    def evaluate(self, state):
        # convert state to tensor
        
        stateTensor = state

        activation1 = F.leaky_relu(self.layer1(stateTensor),negative_slope=1)
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        activation4 = self.layer4(activation3)

        return activation4
    


