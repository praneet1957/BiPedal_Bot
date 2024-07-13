import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os


class env:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(r"D:\Github\RLbot\scripts\Biped.xml") 
        self.data  = mj.MjData(self.model)
        self.state = [self.data.qpos[7],self.data.qpos[8],self.data.qpos[9],self.data.qpos[10],self.data.qpos[0],self.data.qpos[2],0]
        self.cam   = mj.MjvCamera()                        
        self.opt   = mj.MjvOption() 
        self.numStates = 7
        self.numActions  = 4
        self.timestepReward = 0.01
        self.COM = 3.38    # Need to check

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
        self.cam.azimuth     = 45
        self.cam.elevation   = -20 
        self.cam.distance    = 25
        # self.cam.lookat      = np.array([ 0.0 , 0.0 , 0.0 ])
        self.cam.trackbodyid = 0


    def initialize(self):
        # randomly initialize state

        # standing pose
        for i in range(4):
            self.data.qpos[i+7] = 0

        self.data.qpos[0] = 0
        self.data.qpos[2] = 4


    def initializeState(self,state):
        for i in range(4):
            self.data.qpos[i+7] = state[i]
        self.data.qpos[0] = state[4]
        self.data.qpos[2] = state[5]
        #neglecting velocity


    def rewardFunction(self,stateNew, action, StateOld):
        reward = -10*(stateNew[5]- self.COM)**2 + self.timestepReward + stateNew[6]
        return reward 


    def step(self,state,action):
        # self.initializeState(state)
        for i in range(self.numActions):
            self.data.ctrl[i] = action[i]
        
        mj.mj_step(self.model, self.data)
        
        stateNew = np.zeros(self.numStates)

        for i in range(4):
            stateNew[i] = self.data.qpos[i+7]

        #add position and vel of COM
        stateNew[4] = self.data.sensordata[0]
        stateNew[5] = self.data.sensordata[2]
        stateNew[6] = self.data.sensordata[4]


        reward = self.rewardFunction(stateNew, action, state)
        return stateNew, reward









    
