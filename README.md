
![](https://github.com/hpennington/mono-visual-odometry/raw/numpy/point_cloud.gif)

# Mono visual odometry
A Work in Progress implementation of visual odometry in numpy. Currently aided by some opencv functions. The plan is to remove these and implement with purely numpy.


#### Functions to implement in native numpy
  - [ ] goodFeaturesToTrack
  - [ ] ransac
  - [ ] BFMatcher
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
![](https://github.com/hpennington/mono-visual-odometry/raw/main/features.jpeg)


### Red is the ground truth and green is the predicted pose
![](https://github.com/hpennington/mono-visual-odometry/raw/main/vo.png)

![](https://github.com/hpennington/mono-visual-odometry/raw/triangulation/point_cloud.png)



