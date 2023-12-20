import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class GroupAviary(BaseRLAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=3,
                 neighbourhood_radius: float=0.8,
                 initial_xyzs=np.array([ [.0, .0, 0.15], [-.4, .0, 0.15], [.4, .0, 0.15] ]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.PID
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.CROWD_RADIUS = 0.3
        # self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1] for i in range(num_drones)])

    ################################################################################

    # def _actionSpace(self):
    
    ################################################################################

    # def _observationSpace(self):

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        sum = 0
        for i in range(self.NUM_DRONES):
            dist = abs(self.pos[i,2] - 0.5)
            if dist < 0.5:
                sum += 0.5 - dist
        ret = sum / self.NUM_DRONES
        
        sum1 = 0
        total = 0
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                total += 1
                dist = np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :])
                if dist < self.NEIGHBOURHOOD_RADIUS:
                    sum1 += 2
                elif dist - self.NEIGHBOURHOOD_RADIUS < 2.0:
                    sum1 += 2 - (dist - self.NEIGHBOURHOOD_RADIUS)
                if dist < self.CROWD_RADIUS:
                    sum1 -= 2 * (self.CROWD_RADIUS - dist) / self.CROWD_RADIUS
                    
        return ret + sum1/total
        # print(self._getAdjacencyMatrix())
        # print(self._getCrowdingMatrix())
        # return ret + np.linalg.norm(self._getAdjacencyMatrix()) - np.linalg.norm(self._getCrowdingMatrix())

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # dist = 0
        # for i in range(self.NUM_DRONES):
        #     dist += np.linalg.norm(self.TARGET_POS[i,:]-states[i][0:3])
        # if dist < .0001:
        #     return True
        # else:
        #     return False
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _getCrowdingMatrix(self):
            """Computes the crowding matrix of a multi-drone system.

            Attribute CROWD_RADIUS is used to determine neighboring relationships.

            Returns
            -------
            ndarray
                (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the crowding matrix 
                of the system: crowd_mat[i,j] == 1 if (i, j) are crowded; == 0 otherwise.

            """
            crowd_mat = np.identity(self.NUM_DRONES)
            for i in range(self.NUM_DRONES-1):
                for j in range(self.NUM_DRONES-i-1):
                    if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.CROWD_RADIUS:
                        crowd_mat[i, j+i+1] = crowd_mat[j+i+1, i] = 1
            return crowd_mat
    
    ################################################################################