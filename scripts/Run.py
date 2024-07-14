import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os


#create your own
xmlPath        = 'Biped.xml' #xml file (assumes this is in the same folder as this file) 
simend         = 3 #simulation time
printCamConfig = 1 #set to 1 to print camera config

#starting the sim with loading model
model = mj.MjModel.from_xml_path(r"D:\Github\RLbot\scripts\Biped.xml") 
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

# cam settings
cam.azimuth = 45; cam.elevation = -20 ; cam.distance =  25
cam.lookat  = np.array([ data.qpos[0] , data.qpos[1] , data.qpos[2] ])
# cam.trackbodyid = 0

# standing pose
data.qpos[0] = 0
data.qpos[1] = 0
data.qpos[2] = 4
data.qpos[3] = 0.707
data.qpos[4] = 0
data.qpos[5] = 0.707
data.qpos[6] = 0
data.qpos[7] = 0
data.qpos[8] = 0
data.qpos[9] = 0.5
data.qpos[10] = 0.5

time_prev    = 0

# Run Simulation
while not glfw.window_should_close(window):
    mj.mj_step(model, data)
    if (data.time - time_prev > 1.0/10.0):
        mj.mj_step(model, data)
        time_prev = data.time
        print("time" , data.time) 
        print("sensor", data.sensordata)
        print("data" , data.qpos)


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    # if (printCamConfig ==1):
    #     print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
    #     print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')
        # print('testing states', data.State )

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()


