
![](https://github.com/hpennington/mono-visual-odometry/raw/numpy/point_cloud.gif)

# Monocular visual odometry
A Work in Progress implementation of visual odometry in numpy. Currently aided by some opencv functions. The plan is to remove these and implement with purely numpy.


#### Functions to implement in native numpy
  - [ ] goodFeaturesToTrack
  - [ ] ransac
  - [ ] BFMatcher
  - [ ] ORB feature descriptor
  - [x] triangulate
  - [x] extract_pose (Needs refactoring)
  - [x] fundamentalToEssential
  - [x] make_homogeneous


## Setup

```bash
conda env create -f environment.yml
conda activate pyvo
```

## Demo
Press 'q' to end the demo sooner than the full duration.

### Feature matches
The red and green points represent the detected feature in the current and previous frame. A blue line is draw between these points.
![](https://github.com/hpennington/mono-visual-odometry/raw/main/features.jpeg)


### Integrated pose plot
Red is the ground truth and green is the predicted pose
![](https://github.com/hpennington/mono-visual-odometry/raw/main/vo.png)

### Point cloud
Open the point cloud in the open3d app. http://www.open3d.org/download/
![](https://github.com/hpennington/mono-visual-odometry/raw/triangulation/point_cloud.png)



