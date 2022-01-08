# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:43:42 2022

@author: Saharsh
"""

# =============================================================================
# Accessing Images 
# =============================================================================
import rasterio
from rasterio.plot import show
fp = r'./Images/09DEC22000944-P2AS_R1C2-052348982010_01_P003.tif'
img = rasterio.open(fp)
show(img)