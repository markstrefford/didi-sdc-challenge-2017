## Notes on Udacity bags

| Bag  | Comments  | Duration (seconds) | bag_to_kitti.py settings |   
|---|---|---|---|
| 1/2 | Stationary obstacle in front | 66 | |
| 1/3 | Stationary obstacle in front | 66 | |
| 1/4_f | Stationary obstacle in front, then drives away.  Camera car stationary.  | 24 | Appears to work ok with default settings |
| 1/6_f | Obstacle in front, drives away after a 1-2 seconds.  Camera car stationary.  | 11 |  Default settings show increasing error as the obstacle drives away |
| 1/8_f | Camera car driving round in circles. Stationary obstacle and also tree/pole in line of site. Obstacle drives away | 5 | Doesn't work with default settings.  *Suggest ignoring this for training.* |
| 1/9_f | Camera car driving round in straight line. Pulls up to the left of stationary obstacle   | 5 | Works ok on default settings |
|  1/10 | Same as 9_f, but starts further away | 9 | Appears to work ok with default settings.  After frame 170ish, the car is split across left and right in surround view so that bounding box doesn't render correctly.   |  
| 1/11  | Same as 10, but obstacle on left of capture vehicle  | 11  |  Works well with default settings |
| 1/13  | Similar to 11, but capture vehicle drives past obstacle. Obj car facing capture vehicle | 11  | Works well with default settings  |
| 1/14_f  | Obstacle on right of capture vehicle, starts close together and capture vehicle drives away  |  5 | Mid-frames split car across surround view (see 1/10)  |
| 1/15  |  Capture vehicle driving towards obstacle.  Passes it on the right. Obs car facing capture chicle (similar to 1/13??) |  7 |  RTK data not aligned with point cloud (maybe 5 frames ahead?) t=-13  *time offset varies, may be difficult to train with* |
|  1/17 | Simialr to 1/15 but obstacle car is moving  |  7 |  RTK  ahead of PCL.  t=-12 |
| 1/18  | Similar to 1/17 but cars moving quicker  |  7 |  RTK  ahead of PCL.  t=-12 |
|  1/19 | Similar to 1/18. Cars moving quicker again?  |  11 | RTK  ahead of PCL.  t=-12  |
| 1/20  | Capture car following obstacle (both moving), then both stop (capture car behind obstacle car)  | 18  |  Timestamp offset makes no difference here as both cars are moving!! *May not use for training. Look at modifying x by a percentage in LIDAR frame, only would solve issues when cars are both moving* |
|  1/21_f | Capture car behind obstacle. Both start to move off together  | 4  |  Timestamp offset makes no difference here as both cars are moving!! *May not use for training. Look at modifying x by a percentage in LIDAR frame, only would solve issues when cars are both moving* |
| 1/23  | Capture vehicle reverses a few meters then stops.  Obstacle moves across in front like at a junction.  There's an obstacle to the left of the capture vehicle.  | 9  |  t=-30 |
| 1/26 / 1/26_2  |  Capture vehicle stopped.  Obst appears from the right and stops like at a junction.  Capture vehicle moves off, turns left and keeps going |  23 | t=-30 will work up to circa camera frame 190 and then it goes off track as the capture vehicle moves while the obstacle stays still.  With t=0, works ok mostly in top view from frame 190, but issues in surround view (it's not tracking the car correctly)  |
|   |   |   |   |
| 2/1  |  Capture vehicle parked behind obstacle, not quite in line.  Obstacles to the right (trees?) | 66  |   |
| 2/2  | Short version of 2/1 ?  |  17 | Works with defaults  |
| 2/3_f  | Same position as above.  Obst vehicle drives off (then stops in the distance?)  | 21  |   |
| 2/6_f  | Same starting position as above.  Capture vehicle turns left until obst vehicle comes into view  | 6  |   |
| 2/8_f  | Capture vehicle on right of obst.  Drives past it, other trees, etc around.  Banking on the right  | 4  |   |
|  2/11_f | Capture vehicle behind obst.  They both drive off, turning left then straight then turning right, then stopping  | 43  |   |
|  2/12_f | Capture vehicle fulls up behind obst. Grass bank to the left.  They sit for a while, then both drive off straight, turn 180deg right, then drive straight.  Turn right at the end and stop behind the obst vehicle | 78  |   |
| 2/13 | Capture vehicle facing away from grass bank.  Obst vehicle passes in front | 10 | |
| 2/14_f | Similar to 2/13 but obst starts on the right of capture vehicle and drives away (don't really see it in the camera) | 4 | |
| 2/17 | Starts like 2/13.  Obst pulls up to left of capture vehicle, which then drives off, turns to the left and then drives straight | 22 | |
| | | | |
| 3/1 | Capture vehicle parked behind obstacle. Neither move.  Grass bank, etc. to the right | 97 | |
| 3/2_f | Starts as 3/1, then obstacle drives away | 4 | |
| 3/4 | As 3/2_f, but obstacle remains stationary and capture vehicle reverses away | 11 | |
| 3/6 | Start alongside obstacle / to the rear, then reverse and pull in behind it as continues to reverse away (like an overtake in reverse?)|12 | |
| 3/7| Capture vehicle comes up behind obstacle and then overtakes it on the right | 10 | |
| 3/8 | Start to right of obstacle, then drive towards the grassy bank | 8 | |
| 3/9 | Passing stationary obstacle on the right.  Obstacle facing us this time | 11 | |
| 3/11_f| Driving straight, no obstacle.  Grassy bank to the right | 5 | |
| 3/12_f | Same as 3/9 but obstacle moving  | 7 | |
| 3/13_f | Same as 3/12_f but obstacle moving faster, grassy bank is closer to the right of the capture vehicle | | |
| 3/14 | Obstacle passes from left to right, then capture vehicle drives towards grassy bank | 13 | |
| 3/15_f| Capture vehicle facing away from grassy bank, obstacle drives left to right in front, then capture vehicle moves off, turns right, and follows obstacle | 18 | |



Obstacles:

1/* - BMW X3
2/* - Toyota Prius
3/* - ?? 

