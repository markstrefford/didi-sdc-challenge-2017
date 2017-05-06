# Team Timelaps: Udacity Challenge 2017

This code is designed to work with the data provided by Udacity for the [2017 Self Driving Car Challenge](https://github.com/udacity/didi-competition). 

## Dependencies

This code has been developed on a mix of OS/X Sierra and Ubuntu 16.04 LTS hosted on a MacBook Pro, a VirtualBox VM and run on Google Cloud Platform.

The following packages are dependencies:

* Ubuntu 16.04
* ROS Kinetic
* Python 2.7
* OpenCV (ROS Provided)
* Numpy (Latest)
* Pandas (Latest)
* MayAvi (Provided in Ubuntu 16.04)
* matplotlib
* Catkin (Provided by ROS Kinetic)
* Velodyne drivers (see [here](velodybe-tutorials/Installing-Velodyne-Drivers-On-Ros-Kinetic-Ubuntu-16.04-LTS-Xenial.md) for installation)


## Training

To train the model, perform the following steps:

### Prepare the data

For each training bag, follow the following steps:

1. Run the `bag-to-kitti.sh` script from the [Udacity provided code](https://github.com/udacity/didi-competition/tree/master/tracklets)
1. Follow the [ROS instructions here](https://github.com/udacity/didi-competition/tree/master/tracklets) to create a bag containing Velodyne points 
1. Run the script `extract_pointclouds.py` with the pointcloud bag generated above and the same output directory used for `bag_to_kitti.sh` previously.  Note this will also extract Radar points if they are included in the bag.

### Training the model

To train the model, run the following commands:

TODO!!!


## Predicting

The following steps will set up the ROS environment to perform predictions on incoming sensor messages:

1. Run `roscore`
1. In a seperate terminal, run `rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet` to convert Velodyne scans messages into PointCloud2 messages
1. In a seperate window, run the `tracklet generator todo...`
1. In a seperate terminal, run predictor`todo...`
1. Play the test bag, or data from your car's sensors if you have one!!


## Improvements / Next steps

1. Rewrite the code in C
1. Write up a report on my findings
