import numpy as np
import pygame
import os

class biBot():
    def __init__(self, linkArm = 6, linkLeg= 4):
        self.linkArm = linkArm
        self.linkLeg = linkLeg
        # Actuation based on PWM for dc, angle for servo motors 
        self.motLeftArm  = 0                  
        self.motLeftLeg  = 0
        self.motRightArm = 0
        self.motRightLeg = 0
        self.states = [np.pi/2,0,np.pi/2,0];


    def actuate(self,action):
        if action is None:
            self.motLeftArm  = 0
            self.motLeftLeg  = 0
            self.motRightArm = 0
            self.motRightLeg = 0
        else:
            self.motLeftArm  = action[0]
            self.motLeftLeg  = action[1]
            self.motRightArm = action[2]
            self.motRightLeg = action[3]

    

