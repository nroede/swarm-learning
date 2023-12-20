"""" The script uses the control defined in file `swarm_motion_ctrl.py`.

"""
import time
import random
import numpy as np
import pybullet as p

#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
import sys
sys.path.append('../')

from gym_pybullet_drones.envs.GroupAviary import GroupAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from swarm_motion_ctrl import HW2Control

DURATION = 30
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    #### Create the ENVironment ################################
    ENV = GroupAviary(num_drones=9,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, .0, .15], [-.3, .0, .15], [.3, .0, .15],
                                             [.0, -.3, .15], [-.3, -.3, .15], [.3, -.3, .15],
                                             [.0, .3, .15], [-.3, .3, .15], [.3, .3, .15] ]),
                     gui=GUI,
                     record=RECORD
                     )
    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.PYB_FREQ,
                    num_drones=9,
                    )

    #### Initialize the CONTROLLERS ############################
    CTRL_0 = HW2Control(env=ENV,
                        control_type=0
                        )
    CTRL_1 = HW2Control(env=ENV,
                        control_type=1
                        )
    CTRL_2 = HW2Control(env=ENV,
                        control_type=2
                        )
    CTRL_3 = HW2Control(env=ENV,
                        control_type=0
                        )
    CTRL_4 = HW2Control(env=ENV,
                        control_type=1
                        )
    CTRL_5 = HW2Control(env=ENV,
                        control_type=2
                        )
    CTRL_6 = HW2Control(env=ENV,
                        control_type=0
                        )
    CTRL_7 = HW2Control(env=ENV,
                        control_type=1
                        )
    CTRL_8 = HW2Control(env=ENV,
                        control_type=2
                        )

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()
    STATE = OBS[0][0]
    ACTION[0] = CTRL_0.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][1]
    ACTION[1] = CTRL_1.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][2]
    ACTION[2] = CTRL_2.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][3]
    ACTION[3] = CTRL_3.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][4]
    ACTION[4] = CTRL_4.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][5]
    ACTION[5] = CTRL_5.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][6]
    ACTION[6] = CTRL_6.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][7]
    ACTION[7] = CTRL_7.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS[0][8]
    ACTION[8] = CTRL_8.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    #### Initialize the target trajectory ######################
    TARGET_POSITION = np.array([[0.0, 2.0*np.sin(0.006*i), 1.0] for i in range(DURATION*ENV.PYB_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * ENV.PYB_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * ENV.PYB_FREQ, 3])

    #### Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / ENV.PYB_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / ENV.PYB_FREQ

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, DURATION*ENV.PYB_FREQ):

        ### Secret control performance booster #####################
        # if i/ENV.PYB_FREQ>3 and i%30==0 and i/ENV.PYB_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        OBS, _, _, _, _ = ENV.step(ACTION) #!!

        #### Compute control for drone 0 ###########################
        STATE = OBS[0]
        ACTION[0] = CTRL_0.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :],
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 0 ###########################################
        LOGGER.log(drone=0, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 1 ###########################
        STATE = OBS[1]
        ACTION[1] = CTRL_1.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([-.3, .0, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 1 ###########################################
        LOGGER.log(drone=1, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 2 ###########################
        STATE = OBS[2]
        ACTION[2] = CTRL_2.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([.3, .0, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 2 ###########################################
        LOGGER.log(drone=2, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 3 ###########################
        STATE = OBS[3]
        ACTION[3] = CTRL_3.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([.0, -.3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 3 ###########################################
        LOGGER.log(drone=3, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 4 ###########################
        STATE = OBS[4]
        ACTION[4] = CTRL_4.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([-.3, -.3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 4 ###########################################
        LOGGER.log(drone=4, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 5 ###########################
        STATE = OBS[5]
        ACTION[5] = CTRL_5.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([.3, -.3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 5 ###########################################
        LOGGER.log(drone=5, timestamp=i/ENV.PYB_FREQ, state=STATE)
        
        #### Compute control for drone 6 ###########################
        STATE = OBS[6]
        ACTION[6] = CTRL_6.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([.0, .3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 6 ###########################################
        LOGGER.log(drone=6, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 7 ###########################
        STATE = OBS[7]
        ACTION[7] = CTRL_7.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([-.3, .3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 7 ###########################################
        LOGGER.log(drone=7, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Compute control for drone 8 ###########################
        STATE = OBS[8]
        ACTION[8] = CTRL_8.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION[i, :] + np.array([.3, .3, .0]),
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        #### Log drone 8 ###########################################
        LOGGER.log(drone=8, timestamp=i/ENV.PYB_FREQ, state=STATE)

        #### Printout ##############################################
        if i%ENV.PYB_FREQ == 0:
            ENV.render()

        #### Sync the simulation ###################################
        if GUI:
            sync(i, START, ENV.PYB_TIMESTEP)

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    LOGGER.save()

    #### Plot the simulation results ###########################
    LOGGER.plot()
