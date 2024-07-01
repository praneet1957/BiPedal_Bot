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
        self.motors    = 4
        self.tiles     = 25
        self.discountFactor = 0.7
        self.epochs    = 5





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
        globalTime = 0

        # till a certain accuracy is not met
        while(err):
            #Storing data for updation
            data = []

            #initialize env
            self.env.initialize()
            state = self.env.state

            # run for batchTime 
            for t in range(self.batchTime):
                action   = self.generateAction(state)
                stateNew,reward = self.env.step(state,action)
                probAction = self.getProbability(state,action)
                data.append([state,action,probAction,stateNew,reward,probAction])
                state = stateNew


            globalTime += self.batchTime
            # Reshape data as tensors


            # Value function
            
            traindata = []
            # Tile Coding 

            # Every state is a combination of 4 tiling for 4 motors
            for t in range(self.batchTime):
                valueFunction = np.zeros(100)
                state  = data[t][0]
                
                for p in range(0,self.batchTime-t-1,1):
                    reward = data[p][3]
                    valueFunction[state] += reward*self.discountFactor**p

                terminalState = data[self.batchTime-t][0]
                valueFunction[state] += stateValueNetwork(terminalState)*self.discountFactor**(self.batchTime-t)

                #need to see whether to normalize the advantageEst
                advantageEst = valueFunction[state] -  stateValueNetwork(state)*self.discountFactor**(self.batchTime-t)

                #Reshape valuefunction as tensors
                valueFunction = torch.tensor(valueFunction, dtype=torch.float)
                advantageEst  = torch.tensor(advantageEst, dtype=torch.float)
                probActionTensor = torch.tensor(data[:][2],dtype=float)

                # append training data
                traindata.append([valueFunction, advantageEst])


            for e in range(self.epochs):
                #need to calculate phi(a_t/s_t)

                #curr log probs

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - probActionTensor)

                # Calculate losses
                



























