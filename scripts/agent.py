import torch
from torch import nn
import torch.nn.functional as F


class actionFunction:
    def __init__(self):
        #input layer is 4 for 4 motors
        self.layer1 = nn.Linear(6,32) 
        self.layer2 = nn.Linear(32,64)
        self.layer3 = nn.Linear(64,4)
        #output layer is the value function

    def evaluateMean(self, state):
        # convert state to tensor
        stateTensor = torch.tensor(state, dtype=float)

        activationMean1 = F.linear(self.layer1(stateTensor))
        activationMean2 = F.sigmoid(self.layer2(activationMean1))
        activationMean3 = F.linear(self.layer3(activationMean2))

        return activationMean3
    
    def evaluateVariance(self, state):
        # convert state to tensor
        stateTensor = torch.tensor(state, dtype=float)

        # relu to ensure variance is positive
        activationVariance1 = F.sigmoid(self.layer1(stateTensor))
        activationVariance2 = F.relu(self.layer2(activationVariance1))
        activationVariance3 = F.relu(self.layer3(activationVariance2))

        return activationVariance3
    


