import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os


class env:
    def __init__(self):
        self.state = 0
        self.model = mj.MjModel.from_xml_path(r"D:\Github\RLbot\scripts\Biped.xml") 
        self.data  = mj.MjData(self.model)
        self.cam   = mj.MjvCamera()                        
        self.opt   = mj.MjvOption() 
        self.numStates = 7
        self.timestepReward = 0.01
        self.COM = 0.4    # Need to check

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam )
        mj.mjv_defaultOption(self.opt)
        scene   = mj.MjvScene(self.model, maxgeom=10000)
        context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # cam settings
        self.cam.azimuth     = 60 
        self.cam.elevation   = -20 
        self.cam.distance    = 6
        self.cam.lookat      = np.array([ 0.0 , 0.0 , 0.0 ])
        self.cam.trackbodyid = 0


    def initialize(self):
        # randomly initialize state

        # standing pose
        for i in range(self.numStates):
            self.data.qpos[i] = 0



    def initializeState(self,state):
        for i in range(self.numStates):
            self.data.qpos[i] = state[i]




    def rewardFunction(self,stateNew, action, StateOld):
        reward = -10*(stateNew[1]- self.COM)**2 + self.timestepReward + stateNew[2]
        return reward




    def step(self,state,action):
        self.initializeState(state)
        for i in range(self.numActions):
            self.data.act[i] = action[i]
        
        mj.mj_step(self.model, self.data)
        
        stateNew = np.zeros(self.numStates)

        for i in range(self.numStates):
            stateNew[i] = self.data.qpos[i]


        reward = self.rewardFunction(stateNew, action, state)

        return stateNew, reward









    
