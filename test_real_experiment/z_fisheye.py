# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:52:34 2021

@author: Win10
"""
import cv2,time
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
CHECKERBOARD = (7,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('0408/*.png')
for fname in images:
    #fname = images[3]
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)


N_OK = len(objpoints)
K = np.zeros((3, 3))
xi = np.zeros(1)
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

DIM=_img_shape[::-1]
K=np.array(K.tolist())
D=np.array(D.tolist())
X=np.array(xi.tolist())
'''
DIM=(1280, 960)

K = [[313.3001610909224, 0.0, 593.4510680010466],
 [0.0, 312.64362827119885, 459.89320082421756],
 [0.0, 0.0, 1.0]]

D = [[-0.01953323506502111],
 [0.0086013889007471],
 [-0.005590935727437955],
 [0.0008895912850066853]]
'''
def undistort(img_path):
    # img_path = '0409/0074.png'
    img = cv2.imread(img_path)
    time1 = time.time()
    undist_image = cv2.omnidir.undistortImage(img, K, D, X, cv2.omnidir.RECTIFY_CYLINDRICAL, np.eye(3), new_size=(img.shape[0], img.shape[1]))
    time2 = time.time()
    print("running time:",format((time2-time1),'.3f'))
    cv2.imshow("undistorted", undist_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
undistort('0409/0075.png')