# Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213): Sensor Fusion and Tracking. 

The goal of this project is to fuse LiDAR and camera measurements in order to track vehicles over time. This is accomplished following two steps, consisting in detecting objects in 3D point clouds and applying an Extended Kalman Filter (EKF) for sensor fusion and tracking. Real-world data from the [Waymo Open Dataset](https://waymo.com/open/) is used to test the algorithms.

The project consists of two major parts: 
1. **Object detection**: In this part, two deep-learning approaches (Complex YOLOv4 and FPN ResNet18) are used to detect vehicles in LiDAR data based on a birds-eye view of the 3D point-cloud. Also, standard metrics are used to evaluate the performance of the detection approaches. 
2. **Object tracking**: In this part, an extended Kalman filter is used to track vehicles over time, based on the LiDAR detections fused with camera detections. Data association and track management are implemented as well.

## Table of Contents
1. [Table of Contents](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#install--run)
2. [Object Detection](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-detection)
   1. [Lidar Point-Cloud from Range Image](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#lidar-point-cloud-from-range-image)
   2. [Birds-Eye View from Lidar Point-Cloud](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#birds-eye-view-from-lidar-point-cloud)
   3. [Object Detection in BEV Image](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-detection-in-bev-image)
   4. [Performance Evaluation](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#performance-evaluation)
4. [Object Tracking](https://github.com/mose247/sensor-fusion-and-tracking/tree/main#object-tracking)

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
The Waymo Open Dataset uses range images to store LiDAR data. This data structure holds 3D points as a 360° "photo" of the scanning environment with the row dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle. Each image consists in four channels:

- _range_: encodes the distances from the LiDAR sensor to objects in the environment.
- _intensity_: captures the intensity of the laser beam's return signal.
- _eligibility_: contains binary values indicating whether a given pixel in the range image is considered valid.
- _camera projection_: provides information about the 2D projection of the LiDAR data onto a camera image.
  
Below, the range and intensity channels are stack upon each others and converted in 8-bit grey-scale images. To simplify the interpretability, the images are cropped to show just +/-60° around the forward-facing x-axis.

<p align="center">
<img src="https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/range_intensity_image.jpg" title="Range and Intensity channels" width=100% height=100%>
</p>

| Approaching vehicle            |  Ahead vehicles |
:-------------------------:|:-------------------------:
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/vehicle_front_pcl.png)  |  ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/vehicle_rear_pcl.png)

| Parked vehicle            |  Far vehicle |
:-------------------------:|:-------------------------:
![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/parked_vehicle_pcl.png)  |  ![](https://github.com/mose247/sensor-fusion-and-tracking/blob/main/img/far_vehicle_pcl.png)




### Birds-Eye View from Lidar Point-Cloud

### Object Detection in BEV Image

### Performance Evaluation 

## Object Tracking
> TO DO
