# timestamp_utils.py
"""
Cater for timestamp mismatch in Udacity dataset 2

Provide an offset between the camera/RTK data and the pointcloud data.  This is determined through trial and error, and for training
purposes is located in data/training_data.csv
"""

def get_camera_timestamp_and_index(camera_data, pointcloud_timestamp, timestamp_offset):
    camera_index = camera_data.ix[(camera_data.timestamp - pointcloud_timestamp).abs().argsort()[:1]].index[0]

    proposed_index = camera_index + timestamp_offset
    if timestamp_offset > 0:
        actual_index = min(proposed_index,len(camera_data))
    elif timestamp_offset < 0:
        actual_index = max(proposed_index, 0)
    else:
        actual_index = proposed_index

    camera_timestamp = camera_data.ix[actual_index].timestamp
    return camera_timestamp, actual_index
