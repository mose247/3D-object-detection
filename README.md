# Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213): Sensor Fusion and Tracking. 

The goal of this project is to fuse LiDAR and camera measurements in order to track vehicles over time. This is accomplished following two steps, consisting in detecting objects in 3D point clouds and applying an Extended Kalman Filter (EKF) for sensor fusion and tracking. Real-world data from the [Waymo Open Dataset](https://waymo.com/open/) is used to test the algorithms.

The project consists of two major parts: 
1. **Object detection**: In this part, two deep-learning approaches (Complex YOLOv4 and FPN ResNet18) are used to detect vehicles in LiDAR data based on a birds-eye view of the 3D point-cloud. Also, standard metrics are used to evaluate the performance of the detection approaches. 
2. **Object tracking**: In this part, an extended Kalman filter is used to track vehicles over time, based on the LiDAR detections fused with camera detections. Data association and track management are implemented as well.

## Table of Contentsor's resolution 
1. [Install & Run](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#install--run)
2. [Object Detection](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-detection)
   1. [Lidar Point-Cloud from Range Image](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#lidar-point-cloud-from-range-image)
   2. [Birds-Eye View from Lidar Point-Cloud](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#birds-eye-view-from-lidar-point-cloud)
   3. [Object Detection in BEV Image](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-detection-in-bev-image)
   4. [Performance Evaluation](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#performance-evaluation)
4. [Object Tracking](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-tracking)
   1. [Extended Kalman Filter](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#extended-kalman-filter)
   2. [Track Management](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#track-managemnent)
   3. [Data Association](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#data-association)
   4. [Camera and LiDAR dusion](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#camera-and-lidar-fusion)
   5. [Results](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#results)

## Install & Run
To setup the project, start by creating a local copy of the current repository:
```
git clone https://github.com/mose247/sensor-fusion-and-tracking.git
```
All the required dependencies are listed in the file `requirements.txt`, you may either install them one-by-one using `pip` or you can use the following command to install them all at once: 
```
pip3 install -r requirements.txt
```

Moreover, the project makes use of three different sequences from the Waymo Open Dataset to test the concepts of object detection and tracking. These are:

- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to [register](https://waymo.com/open/terms) to Waymo Open Dataset first and click [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) to access the Google Cloud Container that holds all the sequences. Once downloaded, please move the tfrecord-files into the `/dataset` folder of this project.

To execute the project, run the code within `loop_over_dataset.py`. You can choose which steps of the algorithm should be executed by adding the string literal associated to the selected function to one of the following lists: `exec_detection`, `exec_tracking` and `exec_visualization` (more details in the code). 

In case you do not include a specific step into the lists, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be downloaded using this [link](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7), unzipped and moved in the `/results` folder.

## Object Detection

### Lidar Point-Cloud from Range Image
The Waymo Open Dataset uses range images to store LiDAR data. This data structure holds 3D points as a 360° photo of the scanning environment with the row dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle. Each image consists in four channels:

- _range_: encodes the distances from the LiDAR sensor to objects in the environment.
- _intensity_: captures the intensity of the laser beam's return signal.
- _eligibility_: contains binary values indicating whether a given pixel in the range image is considered valid.
- _camera projection_: provides information about the 2D projection of the LiDAR data onto a camera image.
  
Below, the range and intensity channels are stack upon each others and converted in 8-bit grey-scale images. To simplify the interpretability, the images are cropped to show just +/-60° around the forward-facing x-axis.

<p align="center">
<img src="https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/range_intensity_image.jpg" title="Range and Intensity channels" width=100% height=100%>
</p>

Range images can be converted into 3D Point-Clouds by leveraging the sensor intrinsic and estrinsic parameters. Below, vehicles approaching from different directions and with varying degrees of visibility are shown in the 3D space. It is worth to notice that vehicles close to the LiDAR appear as dense distribution of points, making it possible to distinguish  features as tires, side mirrors or windshields. In contrast, vehicles located farther from the sensor are covered with fewer points due to occlusion and the sensor's limited resolution, making it more challenging to identify finer details.

| Vehicle front view            |  Vehicle rear view |
:-------------------------:|:-------------------------:
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/vehicle_front_pcl.png)  |  ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/vehicle_rear_pcl.png)
| **Parked vehicle**           |  **Distant vehicles** |
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/parked_vehicle_pcl.png)  |  ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/far_vehicle_pcl.png)


### Bird's-Eye View from Lidar Point-Cloud
LiDAR Point-Clouds are an unstructured assortment of data points, which are distributed unevenly over the measurement range. With the prevalence of convolutional neural networks (CNN) in object detection, Point-Clouds are typically converted into a more convenient structure before being fed to the model. In the literature, Bird's-Eye View (BEV) is the most widely used projection scheme. 

The BEV representation of a Point-Cloud is achieved by firstly flatten the points along the upward-facing axis. The 2D Cloud is then divided into a regular grid and three values (i.e. max height, max intensity, density) are stored for each region of the road surface. 

<p align="center">
<img src="https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/bev_map.jpg" title="BEV map" width=35% height=35%>
</p>

This enables us to treat the resulting Point-Cloud as a RGB image, where each pixel corresponds to a road patch. Above, it is shown an example of 3-channels BEV map where:
- _R-channel_: encodes density values.
- _G-channel_: encodes height values.
- _B-channel_: encodes intensity values.

As expected, pixels corresponding to vehicles' roof or other high objects appear greenish. In contrast, highly reflective objects, as vehicles' rear lights or plates, take a more bluish color. Finally, the red channel is more intense near the LiDAR sensor, representing the fact that point density is inversely proportional to the range.

### Object Detection in BEV Image
In this project, two pre-trained architectures are tested: Complex YOLOv4 and FPN ResNet18. Below, it is provided a qualitative comparison of these models using the frame 171 in Sequence 1. It is evident that both models successfully detect the three vehicles on the road, while the two parked vehicles go undetected. Nevertheless, this outcome can be considered satisfactory. This is because, upon examining the corresponding camera image for the same frame, a substantial portion of the parked vehicles is obscured by foreground objects, such as a wall and a tree. Consequently, it is unreasonable to expect them to appear clearly in the LiDAR Point-Cloud.

 BEV labels | YOLOv4 | FPN ResNet18 |
:------------:|:----------:|:-----------:
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/bev_labels.jpg) | ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/bev_detections_yolov4.jpg)  |  ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/bev_detections_resnet.jpg) 

| Camera labels |
:------------: |
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/camera_labels.jpg)

### Performance Evaluation 
In this section a more objective evaluation of the models' performance is provided by using the precision and recall metrics. Precision measures the accuracy of positive predictions made by a model, while  recall evaluates the model's ability to identify all relevant instances of the positive class in the dataset. The results below are obtained using 0.5 as IoU threshold for identifying true positives detections.

| YOLOv4 | FPN ResNet18 |
:------------: | :------------: 
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/yolov4_eval.png) | ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/resnet_eval.png)

## Object Tracking

### Extended Kalman Filter
To track the position of targets in the 3D space an Extended Kalman Filter (EKF) is used. In the project, the state vector `x = [x, y, z, vx, vy, vz]` is a 6-dimension array, where the three initial entries represent the object's position and the remaining ones its velocity. The state estimation involves two steps:

1. _Prediction step_: predicts the next state `x` and covariance `P` matrix (i.e. estimation uncertainty) using the state transition equation `f().`
   ```
   x = f(x)
   P = F*P*F.T + Q      # Q is the process covariance matrix and models state equation uncertainty
   ```
   The EKF extends the KF for handling nonlinear systems where the state transition function `f()` is nonlinear. It does this by linearizing the model using its
   first-order Taylor expansion, `F`, evaluated in a neighbourhood of the current state.

   In our case, the targets are assumed to follow constant velocity motion. Therefore, the state transition equation is already linear with respect to the state       and it can be expressed exactly in matrix form `f(x) = F*x`. 
    
3. _Update_step_: corrects the state `x` and covariance `P` estimation using sensor measurements. Their values are updated on the basis of the error between what the expected measurement (i.e. what the sensor is expected to "see" if the true state was the one estimated in the previous step) and the actual data.
   ```
   gamma = z - h(x)    # residual
   S = H*P*H.T + R     # residual variance, R is the process covariance matrix and models measurements uncertainty
   K = P*H.T*S.I       # Kalman gain
   x = x + K*gamma     # state update
   P = (I - K*H)*P     # covariance update
   ```
   Again, the EKF allows us to extend the KF for handling nonlinear systems where the measurement equation `h()` is nonlinear. It does this by linearizing the         model using its first-order Taylor expansion, `H`, evaluated in a neighbourhood of the current state.

   In our case, it is necessary to distinguish between LiDAR and camera measurements. Indeed, while the LiDAR measuerment equation is linear with respect the       state (i.e. can be expressed exactly as `h(x) = H*x`), the camera one is not. Therefore, in this second case it is necessary to linearize the system by evaluating the camera Jacobian `H` in the proximity of the current state.

### Track Management
Every tracking system needs a set of rules to initialize new tracks, update current ones and delete invalid/old ones. In order to simplify tracks management, it is useful to assign them a score. In this project, we scored them based on the number of successful detection (i.e. number of times a track is matched to a measurement) in the last `n` frames:
```
score = n° detections in last n frames / n
```
Using a score threshold, tracks are then classified into three states `initialized`, `tentative` and `confirmed`. This enables to set different track removal policies on the basis of their state. For instance, the following management logic is used:

```
for track in assigned_tracks:         # tracks matched to a measurement
   update track        
   update track.score
   update track.state

for track in unassigned_tracks:       # tracks not assigned to a measurement
   if track is in sensor FOV:
      update track.score

for track in all_tracks:              # check if any track has to be deleted
   if track.state == 'confirmed':
      delete(track) if track.score < 0.6                             # delete track if it is not visible anymore
      delete(track) if track.P[0,0] > 3**2 or track.P[1,1] > 3**2    # delete track if its (x,y) positioning uncertainty is too high
   else:
      delete(track) if track.score < 0.17                            # delete 'ghost' track as soon as possible
      delete(track) if track.P[0,0] > 3**2 or track.P[1,1] > 3**2    # delete track if its (x,y) positioning uncertainty is too high

for meas in unassigned_measurements:   # initialize new tracks
   init new track with meas
```

### Data Association
Deciding which measurement should be used to update which track is not trivial and special care should be put in order to avoid using wrong measurements that can lead the divergence of the state. Instead of the simple Euclidean distance, the Mahalanobis distance is more suitable in this context to measure the proximity of a data point `z` to a tracked state `x`:
```
mhd = (z-h(x)).T*S.I*(z-h(x))
```
As you can see, this distance not only takes into account the mean of the residual but also its uncertainty, given by the covariance matrix `S`. Specifically, since the Mahalanobis distance is inversely proportional to `S`, an higher covariance corresponds to a smaller distance. In practise, given two states with the same Euclidean distance from a measurement, the Mahalanobis measure results closer for the most uncertain track.

In the project, Simple Nearest Neighbor (SNN) is used as data association method. It consists in calculating all Mahalanobis distances between tracks and measurements and iteratively updating the closest association pair.

### Camera and LiDAR fusion
As pointed out in the Extended Kalman Filter [section](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#extended-kalman-filter), LiDAR and camera data are represented by two different measurement models. 

The positional part of the state is mapped to the LiDAR frame through the linear relation:
```
z = R*x_pos   # where H=R is the rotation matrix from the vehicle to the sensor frame
```
Contrarialy, the mapping to image coordinates is nonlinear:
```
z[0] = c_i - f_i * (R*x_pos[1])/(R*x_pos[0])      # c_i principal point horizontal coord, f_i focal length in horizontal dim
z[1] = c_j - f_j * (R*x_pos[2])/(R*x_pos[0])      # c_j principal point vertical coord, f_j focal length in vertical dim
```
Thus, when a measurement arrives from the camera sensor, it is necessary to linearize the equation before using it in the Kalman Filter.

### Results
The tracking system accurately traces all valid objects within the chosen sample sequences by fusing both LiDAR and camera measurements. It achieves an RMSE (Root Mean Square Error) below 0.2 for all the 'confirmed' tracks. However, some false detections are still present. Nevertheless, these 'ghost' tracks are correctly removed after a few frames due to a lack of supporting measurements. 

| Camera + LiDAR fusion 
:------------: |
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/lidar_camera_tracking.gif) 
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/RMSE_lidar_camera.png) 
