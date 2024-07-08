import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os


#create your own
xmlPath   = 'Biped.xml' #xml file (assumes this is in the same folder as this file) 
simend    = 3 #simulation time
camConfig = 0 #set to 1 to print camera config


#starting the sim with loading model
model = mj.MjModel.from_xml_path(xmlPath) 
data  = mj.MjData(model)
cam   = mj.MjvCamera()                        
opt   = mj.MjvOption() 

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene   = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)