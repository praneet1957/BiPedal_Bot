import numpy as np
import pygame
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal


class Policy:
    def __init__(self,agent,network):
        self.numStates  = 0
        self.numActions = #e
        self.agent      = agent  #Either  policy
        self.batchTime  = 10
        self.motors    = 4
        self.tiles     = 25
        self.discountFactor = 0.7
        self.epochs    = 5
        self.actorOptimizer = 4     #define using Adam
        self.networkOptimizer = 1   #define using Adam
        self.clip = 0.1
        self.horizonTime = 4
        self.stateValueNetwork = network


    def generateAction(self, state):
        # generate action for all motors

        # policy defined in agent.py
        actionParamsMean = self.agent.evaluateMean(state)
        actionParamsVar = self.agent.evaluateVaraiance(state)


        # define the distribution based on actions to be taken from particular state
        Covariance = torch.diag(actionParamsVar)
        disMotors = MultivariateNormal(actionParamsMean, Covariance)

        # disMot1 = Normal(actionParamsMean[0],actionParamsVar[0])
        # disMot2 = Normal(actionParamsMean[1],actionParamsVar[1])
        # disMot3 = Normal(actionParamsMean[2],actionParamsVar[2])
        # disMot4 = Normal(actionParamsMean[3],actionParamsVar[3])

        # sample the action from distribution

        # Act1 =  disMot1.sample()
        # Act2 =  disMot2.sample()
        # Act3 =  disMot3.sample()
        # Act4 =  disMot4.sample()
        Action = disMotors.sample()

        # calculate log of probability of chosing the action

        probAction = disMotors.log_prob(Action)

        return Action, probAction


    # def getProbability(self, state, action):
    #     # get probability of chosing a certain action when in a state


    # def calculateObj(self):
    #     # calculate the clip and value network 

    # def computeGrad(self):
    #     # compute gradient

    # def updateParam(self):
    #     # policy parameter update


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
                action, probAction  = self.generateAction(state)
                stateNew,reward = self.env.step(state,action)
                # probAction = self.getProbability(state,action)
                data.append([state,action,probAction,stateNew,reward])
                state = stateNew


            globalTime += self.batchTime
            # Reshape data as tensors

            # Need to check if this is necessary

            # Value function
            
            traindata = []
            # Tile Coding 

            # Every state is a combination of 4 tiling for 4 motors
            visitedState = [0]*self.numStates
            sumVisitedState = 0

            for t in range(self.batchTime):
                valueFunctionTarget = 0
                state  = data[t][0]

                # if sumVisitedState == self.numStates:
                #     break
                # else:
                #     if visitedState[state]==0:
                #         visitedState[state] = 1
                #         sumVisitedState += 1

                maxTime = max(self.horizonTime + t, self.batchTime)
                for p in range(t,maxTime,1):
                    reward = data[p][3]
                    valueFunctionTarget += reward*self.discountFactor**(p-t)

                lastState = data[maxTime][0]
                valueFunctionTarget += self.stateValueNetwork(lastState)*self.discountFactor**(maxTime-t)

                #need to see whether to normalize the advantageEst
                advantageEst = valueFunctionTarget -  self.stateValueNetwork(state)

                # append training data
                traindata.append([valueFunctionTarget, advantageEst])
            
            #Reshape valuefunction as tensors
            valueFunctionTargetTensor = torch.tensor(traindata[:][0], dtype=torch.float)
            advantageEstTensor        = torch.tensor(traindata[:][1], dtype=torch.float)
            probActionTensor          = torch.tensor(traindata[:][2], dtype=float)


            for e in range(self.epochs):
                # current log probs vary as the parameters update

                #initialize env
                self.env.initialize()
                state = self.env.state

                if e!=0:
                    # run for batchTime 
                    for t in range(self.batchTime):
                        actionNew, probActionNew          = self.generateAction(state)
                        stateNewUpdate,rewardNew = self.env.step(state,actionNew)
                        # probActionNew = self.getProbability(state,actionNew)
                        data.append([state,actionNew,probActionNew,stateNewUpdate,rewardNew])
                        state = stateNewUpdate
                    probActionTensorNew  = torch.tensor(data[:][2], dtype=float)
                else:
                    probActionTensorNew = probActionTensor
                

                # Calculate ratios
                ratios = torch.exp(probActionTensorNew - probActionTensor)

                # Calculate losses

                # Clip loss
                lossClipTerm1  = ratios*advantageEstTensor
                lossClipTerm2  = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)*advantageEstTensor
                lossClip       = (-torch.min(lossClipTerm1,lossClipTerm2)).mean()

                #loss of value function when approximated as NN
                lossValueFunction = nn.MSELoss()
                lossValueNetwork  = lossValueFunction(self.stateValueNetwork, valueFunctionTargetTensor)

                #Use the defined optimizer to go back in both clip loss + value network loss
                self.actorOptimizer.zero_grad()
                lossClip.backward(retain_graph=True)
                self.actorOptimizer.step()


                self.networkOptimizer.zero_grad()
                lossValueNetwork.backward(retain_graph= True)
                self.networkOptimizer.step()






































