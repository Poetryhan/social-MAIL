# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:57:27 2022

@author: uqjsun9
"""
# first use 'pip install pyproj' to install the pyproj package
# change your working directory to 'C:\......\interaction-dataset-master\python'

from utils import map_vis_without_lanelet
import matplotlib.pyplot as plt



fig, axes = plt.subplots(1, 1)
# plt.figure(figsize=(10,5))

map_path  = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/maps/DR_USA_Intersection_MA.osm'
map_vis_without_lanelet.draw_map_without_lanelet(map_path, axes, 0, 0)
# plt.xlim([980,1060])
# plt.ylim([970,1030])