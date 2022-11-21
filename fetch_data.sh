#!/usr/bin/env bash

if [ ! -d "../videos" ]
then
    cd ..
    curl -O https://hpennington-cv.s3.amazonaws.com/visual_odometry_videos.zip
    unzip ./visual_odometry_videos
    rm visual_odometry_videos.zip
    cd src
fi
