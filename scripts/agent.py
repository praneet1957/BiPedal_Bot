import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class actionFunction(nn.Module):
    def __init__(self):
        #input layer is 4 for 4 motors and 2 positional variables and 1 velocity
        super(actionFunction, self).__init__()
        self.layer1 = nn.Linear(7,32) 
        self.layer2 = nn.Linear(32,64)
        self.layer3 = nn.Linear(64,4)
        #output layer is the value function

    def evaluateMean(self, state):
        # convert state to tensor
        stateTensor     = torch.tensor(state, dtype=torch.float)
        activationMean1 = F.leaky_relu(self.layer1(stateTensor),negative_slope=1)
        activationMean2 = F.sigmoid(self.layer2(activationMean1))
        activationMean3 = F.leaky_relu(self.layer3(activationMean2), negative_slope=1)

        return activationMean3
    
    def evaluateVariance(self, state):
        # convert state to tensor
        stateTensor = torch.tensor(state, dtype=torch.float)

        # relu to ensure variance is positive
        activationVariance1 = F.relu(self.layer1(stateTensor))
        activationVariance2 = F.relu(self.layer2(activationVariance1))
        activationVariance3 = F.sigmoid(self.layer3(activationVariance2))

        return activationVariance3
    


