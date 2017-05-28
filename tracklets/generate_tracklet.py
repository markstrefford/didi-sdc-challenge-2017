""" Tracklet XML file generation
"""

import numpy as np
import pandas as pd
import os

track_raw = 'data/tracklet_raw.txt'  # raw track_let generated above: one lidar frame => one location
lidar_dir = '/vol/dataset2/Didi-Release-2/Predict/pointcloud/'

header = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?> \n\
<!DOCTYPE boost_serialization> \n\
<boost_serialization signature="serialization::archive" version="9"> \n\
<tracklets class_id="0" tracking_level="0" version="0"> \n\
    <count>1</count> \n\
	<item_version>1</item_version> \n\
	<item class_id="1" tracking_level="0" version="1"> \n\
		<objectType>Car</objectType> \n\
		<h>1.574800</h> \n\
		<w>1.447800</w> \n\
		<l>4.241800</l> \n\
		<first_frame>0</first_frame> \n\
		<poses class_id="2" tracking_level="0" version="0">\n'

footer = '		</poses> \n\
		<finished>1</finished> \n\
	</item> \n\
</tracklets> \n\
</boost_serialization>'


def itemize(position):
    tx, ty, tz = position

    item = '			<item class_id="3" tracking_level="0" version="2"> \n\
				<tx>' + str(tx) + '</tx> \n\
				<ty>' + str(ty) + '</ty> \n\
				<tz>' + str(tz) + '</tz> \n\
				<rx>0.000000</rx> \n\
				<ry>-0.000000</ry> \n\
				<rz>0.000000</rz> \n\
				<state>1</state> \n\
				<occlusion>-1</occlusion> \n\
				<occlusion_kf>-1</occlusion_kf> \n\
				<truncation>-1</truncation> \n\
				<amt_occlusion>0.0</amt_occlusion> \n\
				<amt_occlusion_kf>-1</amt_occlusion_kf> \n\
				<amt_border_l>0.0</amt_border_l> \n\
				<amt_border_r>0.0</amt_border_r> \n\
				<amt_border_kf>-1</amt_border_kf> \n\
			</item>\n'

    return item

lidar_files = np.array([int(lidar.split('.')[0]) for lidar in sorted(os.listdir(lidar_dir))])

#tracklet_raw = [line.strip() for line in open(track_raw)]
#tracklet_raw = np.array([[float(number) for number in line.split(',')] for line in tracklet_raw])
tracklet_raw =

frame=0
with open('data/tracklet_preds.xml', 'w') as tracklet:
    tracklet.write(header)

    lidar_files = sorted(os.listdir(lidar_dir))

    tracklet.write('			<count>' + str(len(lidar_files)) + '</count>\n')
    tracklet.write('			<item_version>2</item_version>\n')

    for image in image_files:

        """
        Create an xml entry for each image
        """
        item = itemize(tracklet_raw[frame])

        tracklet.write(item)
        frame +=1

    tracklet.write(footer)