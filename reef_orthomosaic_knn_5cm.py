# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:00:35 2022

@author: Saharsh
"""

import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Read in raster image 
fp = r'./Images/Orthomosaic/2021_Heron_Drone_Orthomosaic_5cm.tif'
img_ds = gdal.Open(fp, gdal.GA_ReadOnly)

'''
Raster profile:
{'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 6286, 'height': 8234, 'count': 4, 'crs': CRS.from_epsg(7856), 'transform': Affine(0.04999999999999815, 0.0, 388792.0548731448,
       0.0, -0.050000000000022624, 7407021.525006049), 'blockxsize': 128, 'blockysize': 128, 'tiled': True, 'compress': 'lzw', 'interleave': 'pixel'}
RasterBandCount = 4
Size = (8234, 6286)
'''
#utilising all the bands of the image:
    
# loading a multi-band image into a numpy
img=np.zeros((img_ds.RasterYSize,img_ds.RasterXSize,img_ds.RasterCount),gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:,:,b]=img_ds.GetRasterBand(b + 1).ReadAsArray()
#reshaping array:
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
X= img[:,:,:4].reshape(new_shape)

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
