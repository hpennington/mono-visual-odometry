#!/usr/bin/env bash

if [ ! -d "./videos" ]
then
    curl -O https://hpennington-cv.s3.amazonaws.com/visual_odometry_videos.zip
    unzip ./visual_odometry_videos
fi
