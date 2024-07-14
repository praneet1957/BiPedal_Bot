import numpy as np
# import pygame
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
from policyNetwork import valueFunction
from agent import actionFunction
from env import env
torch.set_grad_enabled(True)  # Context-manager 


class Policy:
    def __init__(self,agent,network, environment):
        self.numStates  = 7
        self.numActions = 4
        self.agent      = agent()  #Either  policy
        self.stateValueNetwork = network()
        self.learningRate      = 0.001
        self.batchTime  = 100
        self.motors     = 4
        self.tiles      = 25
        self.discountFactor = 0.7
        self.epochs         = 5
        self.actorOptimizer   = Adam(self.agent.parameters(), lr=self.learningRate)     #define using Adam
        self.networkOptimizer = Adam(self.stateValueNetwork.parameters(), lr=self.learningRate)     #define using Adam
        self.clip        = 0.1
        self.horizonTime = 4
        self.saveHorizon = 4
        self.logger = {
			'actorClipLosses': [],     # losses of actor network in current iteration
            'networkLosses':[],
            'avgRewardperEps': []
		}
        
        
        self.env = environment()


    def generateAction(self, state):
        # generate action for all motors

        # policy defined in agent.py
        actionParamsMean = self.agent.evaluateMean(state)
        actionParamsVar = self.agent.evaluateVariance(state)


        # define the distribution based on actions to be taken from particular state
        Covariance = torch.diag(actionParamsVar)
        disMotors = MultivariateNormal(actionParamsMean, Covariance)

        # disMot1 = Normal(actionParamsMean[0],actionParamsVar[0])
        # disMot2 = Normal(actionParamsMean[1],actionParamsVar[1])
        # disMot3 = Normal(actionParamsMean[2],actionParamsVar[2])
        # disMot4 = Normal(actionParamsMean[3],actionParamsVar[3])

        # sample the action from distribution

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
        N = 0
        while(1):
            N = N +1 
            print("Current Iteration", N)
            #GUI update
            self.env.guiUpdate()

            #Storing data for updation
            data = []
            probActionList = []
            stateList = []

            #initialize env
            self.env.initialize()
            state = self.env.state


            # run for batchTime 
            for t in range(self.batchTime):
                action, probAction  = self.generateAction(state)
                stateNew,reward = self.env.step(state,action)
                # probAction = self.getProbability(state,action)
                stateList.append(state)
                data.append([state,action,probAction,stateNew,reward])
                state = stateNew
                probActionList.append(probAction)
                self.env.guiUpdate()
                
                

            globalTime += self.batchTime
            # Reshape data as tensors
            stateListTensor = torch.tensor(stateList, dtype=torch.float)
            # Need to check if this is necessary

            # Value function
            
            traindata = []
            valueFunctionTargetLis = []
            advantageEstLis = []
            # Tile Coding 

            # Every state is a combination of 4 tiling for 4 motors
            # Neglect tiling
            # Every state is a combination of 4 motors + z (height) + x(longitudinal distance)

            # visitedState = [0]*self.numStates
            # sumVisitedState = 0

            for t in range(self.batchTime):
                valueFunctionTarget = 0
                state  = stateListTensor[t]


                # if sumVisitedState == self.numStates:
                #     break
                # else:
                #     if visitedState[state]==0:
                #         visitedState[state] = 1
                #         sumVisitedState += 1

                # maxTime = max(self.horizonTime + t-1, self.batchTime-1)
                maxTime = self.batchTime - 1
                for p in range(t,maxTime,1):
                    reward = data[p][4]
                    valueFunctionTarget += reward*self.discountFactor**(p-t)



                lastState = data[maxTime-1][0]
                lastStateTensor = torch.tensor(lastState, dtype = torch.float)
                
                valueFunctionTarget += (self.stateValueNetwork.evaluate(lastStateTensor).item())*(self.discountFactor**(maxTime-t-1))

                # need to see whether to normalize the advantageEst

                advantageEst = torch.tensor(valueFunctionTarget, dtype= torch.float) -  self.stateValueNetwork.evaluate(state)

                # append training data
                advantageEstLis.append(advantageEst)
                valueFunctionTargetLis.append(valueFunctionTarget)
                


            
            #Reshape valuefunction as tensors
            valueFunctionTargetTensor = torch.tensor(valueFunctionTargetLis, dtype=torch.float)
            advantageEstTensor        = torch.tensor(advantageEstLis, dtype=torch.float)
            probActionTensor          = torch.tensor(probActionList, dtype=torch.float)


            for e in range(self.epochs):
                # current log probs vary as the parameters update

                epochData = [] 
                probActionNewList = []
                cumReward = 0

                #initialize env
                self.env.initialize()
                state = self.env.state

                if e!=0:
                    # run for batchTime 
                    for t in range(self.batchTime):
                        actionNew, probActionNew = self.generateAction(state)
                        stateNewUpdate,rewardNew = self.env.step(state,actionNew)
                        # probActionNew = self.getProbability(state,actionNew)
                        epochData.append([state,actionNew,probActionNew,stateNewUpdate,rewardNew])
                        probActionNewList.append(probActionNew)
                        state = stateNewUpdate
                        cumReward += rewardNew
                        self.env.guiUpdate()

                    probActionTensorNew  = torch.tensor(probActionNewList, dtype=torch.float)

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

                #Need to pass again through New netowrk
                stateValueNetworkNew = []
                for t in range(self.batchTime):
                    state = stateListTensor[t]
                    stateValueNetworkNew.append(self.stateValueNetwork.evaluate(state))



                stateValueNetworkNewTensor = torch.tensor(stateValueNetworkNew, dtype = torch.float)
                lossValueNetwork  = lossValueFunction(stateValueNetworkNewTensor, valueFunctionTargetTensor)

                #Use the defined optimizer to go back in both clip loss + value network loss
                lossClip.requires_grad = True
                lossValueNetwork.requires_grad = True

                self.actorOptimizer.zero_grad()
                lossClip.backward(retain_graph=True)
                self.actorOptimizer.step()

                self.networkOptimizer.zero_grad()
                lossValueNetwork.backward(retain_graph= True)
                self.networkOptimizer.step()

                # Log  loss
                self.logger['actorClipLosses'].append(lossClip.item())
                self.logger['networkLosses'].append(lossValueNetwork.item())
                self.logger['avgRewardperEps'].append(cumReward/self.batchTime)

            #save the weights
            if N % self.saveHorizon == 0:
                torch.save(self.agent.state_dict(), './ppo_actor.pth')
                torch.save(self.stateValueNetwork.state_dict(), './ppo_valuefunction.pth')
    


# To Run the PPO we start here
model = Policy(actionFunction, valueFunction, env)

# Run Simulation
model.evaluatePolicy()

    





































