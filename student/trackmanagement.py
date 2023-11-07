# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        trans = meas.sensor.sens_to_veh[0:3, 3]   # translation vector from sensor to vehicle coordinates
        
        # initialize position with the first measurement and velocity to zero
        x = np.zeros((6,1))
        x[:3] = rot * meas.z + trans
        self.x = np.matrix(x)
        
        # initialize position covariance with measurement covariance and velocity covariance to a high value
        P = np.eye(6) * params.sigma_p44
        P[:3,:3] = rot * meas.R * rot.transpose()
        self.P = np.matrix(P)
        
        # initialize track state 
        self.state = 'initialized'

        # initialize a list to store whether the track was detected in the last n frames (1-detected, 0-undetected)
        self.last_n_detections = collections.deque([0]*params.window, params.window)
        self.last_n_detections.append(1)    

        # initialize track score
        self.update_score()
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(rot[0,0]*np.cos(meas.yaw) + rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(rot[0,0]*np.cos(meas.yaw) + rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
    
    def update_score(self):
        # update track score based on the number of detection is the last n frames
        self.score = sum(self.last_n_detections) / len(self.last_n_detections)
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        
        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # mark track as undetected and decrease score
                    track.last_n_detections.append(0)
                    track.update_score()                                 
                    
        # delete old tracks   
        for track in self.track_list:
            # use different score threshlolds for different track states
            if track.state == 'confirmed':
                if track.score < params.delete_threshold_conf: 
                    self.delete_track(track)
            else: 
                if track.score < params.delete_threshold_tent: 
                    self.delete_track(track)

            # delete if covariance is too high
            if track.P[0,0]>params.max_P or track.P[1,1]>params.max_P:
                self.delete_track(track)

        # initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])

            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        # increase track score
        track.last_n_detections.append(1)
        track.update_score()

        # set track state to 'tentative' or 'confirmed'
        if track.score > params.confirmed_threshold:
            track.state = 'confirmed'
        else:
            track.state = 'tentative'