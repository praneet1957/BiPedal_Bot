import numpy as np
import pygame
import os
import torch
from torch import nn
import torch.nn.functional as F

class Policy:
    def __init__(self,agent):
        self.numStates  = 0
        self.numActions = #e
        self.agent      = agent  #Either  policy
        self.batchTime  = 10





    def valueFunction(self):
        #Calculate the value function of current policy


    def generateAction(self, state):
        # generate action for all motors

    def getProbability(self, state, action):
        # get probability of chosing a certain action when in a state


    def calculateObj(self):
        # calculate the clip and value network 

    def computeGrad(self):
        # compute gradient

    def updateParam(self):
        # policy parameter update

    def evaluatePolicy(self):
        # the final part of code to run the policy

        #Storing data for updation
        data = []

        #initialize env
        self.env.initialize()
        state = self.env.state

        for t in range(self.batchTime):
            action   = self.generateAction(state)
            stateNew,reward = self.env.step(state,action)
            probAction = self.getProbability(state,action)
            data.append([state,action,stateNew,reward,probAction])
            state = stateNew

        # Value function
        Valuefunction = []*self.numStates
        for t in range(self.batchTime):













