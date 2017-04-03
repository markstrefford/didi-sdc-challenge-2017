
import rosbag
import cv2
from cv_bridge import CvBridge
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from mayavi import mlab
import numpy as np

bag_file_dir = '/media/sf_vol/didi/Didi-Training-Release-1'
image_size = (512, 1400, 3)
lidar_size = (42034, 5)

fig = mlab.figure(bgcolor=(0, 0, 0), size=(500, 300))
lidar = np.zeros(lidar_size)
plt = mlab.points3d(lidar[:, 0], lidar[:, 1], lidar[:, 2],
                    np.ones(lidar.shape[0]), #lidar[:, 0] ** 2 + lidar[:, 1] ** 2,
                    mode="point",
                    colormap='spectral',
                    figure=fig,
                    )
msplt = plt.mlab_source

# Load images from the ROS databag, resize accordingly and ensure orientation (w x h instead of h x w)
def load_rosbag_data(bag_file):
    print ('Loading databag {}'.format(bag_file))
    cvbridge = CvBridge()
    bag = rosbag.Bag(os.path.join(bag_file_dir, bag_file))

    img = np.zeros(image_size)
    #lidar = np.zeros(lidar_size)
    _obs1_gps_fix = []
    _gps_fix = []

    for topic, msg, t in bag.read_messages(
        topics=[
            # '/gps/time',
            '/gps/fix',
            '/gps/rtkfix',
            '/obs1/gps/fix',
            '/image_raw',
            '/velodyne_points'
        ]):
        print ('Topic: {}'.format(topic))
        if topic == '/image_raw':
            img = cvbridge.imgmsg_to_cv2(msg, "bgr8")

        elif topic == '/velodyne_points':
            lidar_msg = pc2.read_points(msg)
            lidar = np.array(list(lidar_msg))
            print ('LIDAR: {}'.format(lidar.shape))
            yield img, lidar

        elif topic == '/obs1/gps/fix':
            latitude = msg.latitude
            longitude = msg.longitude
            altitude = msg.altitude
            _obs1_gps_fix.append((latitude, longitude, altitude))
            # gps_count +=1

        elif topic == '/gps/fix':
            latitude = msg.latitude
            longitude = msg.longitude
            altitude = msg.altitude
            _gps_fix.append((latitude, longitude, altitude))

    bag.close()

# Main loop
# TODO - Make sure it's main (if '__main__'... etc) and also add in argparse...

for image, lidar in load_rosbag_data('overtake.bag'):
    msplt.reset(x=lidar[:, 0], y=lidar[:, 1], z=lidar[:, 2],
              scalars=lidar[:, 0] ** 2 + lidar[:, 1] ** 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)

