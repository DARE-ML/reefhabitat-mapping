# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:43:42 2022

@author: Saharsh
"""

# =============================================================================
# Accessing Images TIF to PNG

# import rasterio
# from rasterio.plot import show
# fp = r'./Images/09DEC22000944-P2AS_R1C2-052348982010_01_P003.tif'
# img = rasterio.open(fp)
#print(img.profile)
# show(img)
# =============================================================================

import rasterio
from rasterio.plot import show
import cv2
import numpy as np
import os
fp=r'./Images/Worldview_2_images/geo_image_1.png'
img=cv2.imread(fp)
# cv2.imshow('View_Image',img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# =============================================================================
# K - Means clustering
# =============================================================================

img2 = np.float32(img.reshape((-1,3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #( type of termination, max_iter, epsilon/req. acc )

#defining k clusters
k=17
attempts=10
ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

#reshaping into uint8 -> original image
center=np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#saving image

path = './Images/knn/'
filename='geo_image_1_knn_'+str(k)+'.jpg'
cv2.imwrite(os.path.join(path ,filename), res2)

#visualise
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# 
# =============================================================================
