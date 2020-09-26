#!/usr/bin/env python3
# -*- coding:utf-8 -*-
u'''
Created on 2019年4月22日

@author: wuluo
'''
__author__ = 'wuluo'
__version__ = '1.0.0'
__company__ = u'重庆交大'
__updated__ = '2019-04-26'

# 原始jpg已经畸变矫正
import numpy as np
import cv2
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
capL = cv2.VideoCapture(2)
capR = cv2.VideoCapture(0)
imgL = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), np.uint8)
imgR = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), np.uint8)

stereo = None
opencv_measure_version = int(cv2.__version__.split('.')[0])
windowSize = 5
minDisp = 10
numDisp = 250 - minDisp

stereo = cv2.StereoSGBM_create(
    minDisparity=minDisp,
    numDisparities=numDisp,
    blockSize=16,
    P1=8 * 3 * windowSize**2,
    P2=32 * 3 * windowSize**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
capL.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
imgL = cv2.imread('lef2t.jpg')
imgR = cv2.imread('right.png')

# create gray images
imgGrayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgGrayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# calculate histogram
imtGrayL = cv2.equalizeHist(imgGrayL)
imtGrayR = cv2.equalizeHist(imgGrayR)

# through gausiann filter
imgGrayL = cv2.GaussianBlur(imgGrayL, (5, 5), 0)
imgGrayR = cv2.GaussianBlur(imgGrayR, (5, 5), 0)
cv2.imshow("image left",imgGrayL)
cv2.imshow("image right", imgGrayR)

# calculate disparity
disparity = stereo.compute(imgGrayL, imgGrayR).astype(np.float32) / 16
disparity = (disparity - minDisp) / numDisp
cv2.imshow("disparity", disparity)
cv2.waitKey(0)

if __name__ == "__main__":
    pass