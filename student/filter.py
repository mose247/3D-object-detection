# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import misc.params as params 
from student.trackmanagement import Trackmanagement

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = 6  # state dimension [x, y, z, vx, vy, vz]
        self.dt = params.dt # time increment
        self.q = params.q   # process noise variable 

        self.F = self.F()   # process state matrix
        self.Q = self.Q()   # process covariance matrix


    def F(self):
        # implement and return system matrix F, where state 
        F = np.matrix([[1, 0, 0, self.dt,  0,      0    ],
                       [0, 1, 0,     0,  self.dt,  0    ],
                       [0, 0, 1,     0,      0,  self.dt],
                       [0, 0, 0,     1,      0,    0    ],
                       [0, 0, 0,     0,      1,    0    ],
                       [0, 0, 0,     0,      0,    1    ]])
        
        return F
        

    def Q(self):
        # implement and return process noise covariance Q
        q1 = 1/3*(self.dt**3)*self.q
        q2 = 1/2*(self.dt**2)*self.q
        q3 = self.dt*self.q

        Q = np.matrix([[q1, 0, 0, q2,  0,  0],
                       [0, q1, 0,  0, q2,  0],
                       [0, 0, q1,  0,  0, q2],
                       [q2, 0, 0, q3,  0,  0],
                       [0, q2, 0,  0, q3,  0],
                       [0, 0, q2,  0,  0, q3]])
        
        return Q
    

    def predict(self, track):
        # predict state x and estimation error covariance P to next timestep
        x = self.F * track.x                                  # state prediction
        P = self.F * track.P * self.F.transpose() + self.Q    # covariance prediction

        # save x and P in track
        track.set_x(x)
        track.set_P(P)
        

    def update(self, track, meas):
        # get matrices for update step
        P = track.P                     # state estimation covariance
        H = meas.sensor.get_H(track.x)  # Jacobian measurement matrix (evaluated in current state)
        S = self.get_S(track, meas, H)  # residual covariance

        # compute residual
        gamma = self.gamma(track, meas)

        # update state x and covariance P with associated measurement
        K = P*H.transpose()*np.linalg.inv(S)    # Kalman gain
        x = track.x + K*gamma                   # state update
        P = (np.eye(self.dim_state) - K*H) * P  # covariance update

        # save x and P in track
        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

    
    def gamma(self, track, meas):
        # compute residual (error between actual measurement and expected measurement)
        gamma = meas.z - meas.sensor.get_hx(track.x)
  
        return gamma
        

    def get_S(self, track, meas, H):
        # calculate and return covariance of residual S
        P = track.P     # state estimation covariance
        R = meas.R      # measurement covariance
        S = H*P*H.transpose() + R  

        return S
        
