import numpy as np
from scipy.spatial.transform import Rotation as R


import numpy as np
from tools.SE3_tools import exponential_map

    
class SimulatorCamera:
    def __init__(self, delta_t=0.04):
        self.delta_t = delta_t
        self.state = 0  # STATE_STOP
        self.wMc = np.eye(4)

    def setSamplingTime(self, delta_t):
        self.delta_t = delta_t

    def setPosition(self, wMc):
        self.wMc = wMc.copy()

    def getPosition(self):
        return self.wMc.copy()

    def setRobotState(self, state):
        self.state = state

    def getRobotState(self):
        return self.state

    def setVelocity(self, frame, v):
        if self.getRobotState() != 1:
            self.setRobotState(1)
            
        T = exponential_map(v, self.delta_t)
        self.wMc = self.wMc @ T

    '''    
    def twist_to_transform(self, v, dt):
        t = v[:3] * dt
        w = v[3:] * dt
        theta = np.linalg.norm(w)

        wx = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])

        R = np.eye(3)
        if theta > 1e-6:
            R += (np.sin(theta)/theta)*wx + ((1-np.cos(theta))/theta**2)*(wx @ wx)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    '''
