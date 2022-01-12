# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:13:37 2022

@author: Saharsh
"""

import rasterio
from rasterio.plot import show
import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import os
# =============================================================================
# fp=r'./Images/Worldview_2_images/052348982010_01/052348982010_01_P002_MUL/09DEC05235332-M2AS_R1C1-052348982010_01_P002.tif'
# img = rasterio.open(fp)
# print(img.profile)
# # show(img) #PAN
# show(img.read(3)) #MUL
# =============================================================================

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Read in raster image 
fp = r'./Images/Worldview_2_images/052348982010_01/052348982010_01_P001_PAN/09DEC14000113-P2AS_R1C2-052348982010_01_P001.tif'
img_ds = gdal.Open(fp, gdal.GA_ReadOnly) #reads as gdal.dataset object


#utilising all the bands of the image:
    
# loading a multi-band image into a numpy
img=np.zeros((img_ds.RasterYSize,img_ds.RasterXSize,img_ds.RasterCount),gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
#print("Band sise",img.shape[2])

for b in range(img.shape[2]):
    img[:,:,b]=img_ds.GetRasterBand(b + 1).ReadAsArray()
#reshaping array:
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
X= img[:,:,:img.shape[2]].reshape(new_shape)

#k-nn Clusterinng
k_means = cluster.KMeans(n_clusters=7)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(img[:, :, 0].shape)

print("Clustering Done ...")
#Visualising
plt.figure(figsize=(20,20))
plt.imshow(X_cluster, cmap="hsv")

plt.show()