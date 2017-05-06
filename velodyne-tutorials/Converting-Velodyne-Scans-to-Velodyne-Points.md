# Convert Velodyne Scans to PointCloud2

In separate terminal sessions run the following:

    roscore

Run the nodelet:

    rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet
    
Use rosbag record to record the point cloud messages:

    rosrun rosbag record -O pointcloud.bag /velodyne_points

And finally play your bag containing the VelodyneScan:

    rosbag play scans.bag




This is also provided in the following [answers.ros.org](http://answers.ros.org/question/191972/convert-velodynescan-to-pointcloud2-from-a-rosbag-file/?answer=259394#post-id-259394) question.


