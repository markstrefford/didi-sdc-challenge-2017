## TODO: Remove this altogether??? Don't think it's needed!!

import os
os.environ['HOME'] = '/root'

SEED = 202


# std libs
import glob


# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import cv2
#import mayavi.mlab as mlab


from utils.draw import *


## save mpg:
##    os.system('ffmpeg -y -loglevel 0 -f image2 -r 15 -i %s/test/predictions/%%06d.png -b:v 2500k %s'%(out_dir,out_avi_file))
##
##----------------------------------------------------------------------------

## preset view points
#  azimuth=180,elevation=0,distance=100,focalpoint=[0,0,0]
## mlab.view(azimuth=azimuth,elevation=elevation,distance=distance,focalpoint=focalpoint)
MM_TOP_VIEW  = 180, 0, 120, [0,0,0]
MM_PER_VIEW1 = 120, 30, 70, [0,0,0]
MM_PER_VIEW2 = 30, 45, 100, [0,0,0]
MM_PER_VIEW3 = 120, 30,100, [0,0,0]


# main #################################################################
# for demo data:  /root/share/project/didi/data/didi/didi-2/Out/1/15

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    lidar_dir      ='/root/share/project/didi/data/didi/didi-2/Out/1/15/lidar'
    gt_boxes3d_dir ='/root/share/project/didi/data/didi/didi-2/Out/1/15/processed/gt_boxes3d'
    mark_dir       ='/root/share/project/didi/data/didi/didi-2/Out/1/15/processed/mark-gt-box3d'
    avi_file       ='/root/share/project/didi/data/didi/didi-2/Out/1/15/processed/mark-gt-box3d.avi'

    #mark_gt_box3d(lidar_dir,gt_boxes3d_dir,mark_dir)
    dir_to_avi(avi_file, mark_dir)

