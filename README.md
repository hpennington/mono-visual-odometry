
![](https://github.com/hpennington/mono-visual-odometry/raw/main/point_cloud.gif)

# Monocular visual odometry
A Work in Progress implementation of visual odometry in numpy. Currently aided by some opencv functions. The plan is to remove these and implement with purely numpy.


#### Functions to implement in native numpy
  - [ ] goodFeaturesToTrack
  - [x] ransac
  - [x] FundamentalMatrixTransform
  - [ ] Brute-Force Matcher (BFMatcher)
  - [ ] Compute ORB feature descriptors
  - [x] triangulate
  - [x] extract_pose (Needs refactoring for deciding ambiguities)
  - [x] fundamentalToEssential
  - [x] make_homogeneous


## Setup

```bash
conda env create -f environment.yml
conda activate pyvo
```

My suggestion is to open src/visual_odometry.ipynb in VS Code.
Any web browser and jupyter notebook will work, but I find that VS Code make the nicest notebook environment.

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



