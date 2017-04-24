#!/usr/bin/env bash
#
# Run the rosbag viewer from https://github.com/jokla/didi_challenge_ros
#
# TODO: Check there's a bagfile on the command line
roslaunch didi_challenge_ros display_rosbag_rviz.launch rosbag_file:=$1
