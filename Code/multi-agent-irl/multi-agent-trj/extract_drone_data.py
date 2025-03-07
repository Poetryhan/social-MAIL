# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:43:44 2020

@author: uqjsun9
"""
import csv
import numpy as np
gps_data_all=[]

dates=['20181024_dX_0830_0900', '20181024_dX_0900_0930', '20181024_dX_0930_1000', '20181024_dX_1000_1030', '20181024_dX_1030_1100',
       '20181029_dX_0800_0830', '20181029_dX_0830_0900', '20181029_dX_0900_0930', '20181029_dX_0930_1000', '20181029_dX_1000_1030', 
       '20181030_dX_0800_0830', '20181030_dX_0830_0900', '20181030_dX_0900_0930', '20181030_dX_0930_1000', '20181030_dX_1000_1030', 
       '20181101_dX_0800_0830', '20181101_dX_0830_0900', '20181101_dX_0900_0930', '20181101_dX_0930_1000', '20181101_dX_1000_1030'] 

for date in dates:
    
    # date = '20181101_dX_0800_0830'
    path = 'E:/EFPL data/' + date + '.csv'

    gps_data = []
    gps_data1 = []
    gps_data2 = []
    gps_data3 = []
    csv.field_size_limit(100000000)
    with open(path) as csvfile:
        reader = csv.reader(csvfile)    
        for row in reader:   
            if reader.line_num > 1:
                veh = list(row[0].split('; '))
                if veh[1] in ['Car','Taxi', 'Medium Vehicle']:
                    gps_data.append(np.zeros([241,4]))   
                    gps_data1.append(np.zeros([31,4]))   
                    gps_data2.append(np.zeros([31,4]))   
                    gps_data3.append(np.zeros([31,4]))
                    
                    for step in range((len(veh)-5)//6):
                        time = float(veh[9+step*6])
                        if time < 1200 and (time%5 < 0.02 or time%5 > 9.98):
                            gps_data[-1][round(time/5)]=np.array([float(i) for i in [veh[4+step*6],veh[5+step*6],veh[6+step*6],veh[9+step*6]]])
                        # if 300<=time < 600 and (time%10 < 0.02 or time%10 > 9.98):
                        #     gps_data1[-1][round(time/10)-30]=np.array([float(i) for i in [veh[4+step*6],veh[5+step*6],veh[6+step*6],veh[9+step*6]]])
                        # if 600<=time < 900 and (time%10 < 0.02 or time%10 > 9.98):
                        #     gps_data2[-1][round(time/10)-60]=np.array([float(i) for i in [veh[4+step*6],veh[5+step*6],veh[6+step*6],veh[9+step*6]]])
                        # if 900<=time < 1200 and (time%10 < 0.02 or time%10 > 9.98):
                        #     gps_data3[-1][round(time/10)-90]=np.array([float(i) for i in [veh[4+step*6],veh[5+step*6],veh[6+step*6],veh[9+step*6]]])
                            
       
    
    gps_data_all.append(gps_data)
    # gps_data_all.append(gps_data1)
    # gps_data_all.append(gps_data2)
    # gps_data_all.append(gps_data3)
    
gps_data_all= list(gps_data_all)
np.save('gps_data_all-5s.npy',gps_data_all)
#%%
max_log = 23.739
min_log = 23.728
max_lat = 37.986
min_lat = 37.975
gps_data_all2=[]
for gps_data in gps_data_all:
    for j in reversed(range(len(gps_data))):
        index = np.where(np.any([gps_data[j][:,0] > max_lat, gps_data[j][:,0] < min_lat, gps_data[j][:,1] > max_log, gps_data[j][:,1] < min_log], axis=0))
        gps_data[j][index,:] = 0
        
        arr = np.where(gps_data[j][:,2]>0)
        if arr[0].size>0:
            gps_data[j][0:arr[0][0],:] = 0
            if arr[0][-1]<241:
                gps_data[j][arr[0][-1]+1:241,:] = 0
        
        if  np.count_nonzero(gps_data[j][:,2])<10:
            gps_data = np.delete(gps_data,j,0)
    gps_data_all2.append(gps_data)


for gps_data in gps_data_all2:
    if gps_data.size==0:
        gps_data_all2.remove(gps_data)


np.save('gps_data_all-5.npy',gps_data_all2)


#%%

path = 'E:/EFPL data/20181024_d1_0830_0900.csv'


gps_data_1 = []
csv.field_size_limit(100000000)

with open(path) as csvfile:
    reader = csv.reader(csvfile)    
    for row in reader:   
        if reader.line_num > 1:
            veh = list(row[0].split('; '))    
            
                    
            for step in range((len(veh)-5)//6):
                # time = float(veh[9+step*6])
                # if time < 1200 and time%30 < 0.0001:
                lat = float(veh[4+step*6])
                log = float(veh[5+step*6])
                max_lat = max(max_lat,lat) 
                min_lat = min(min_lat,lat) 
                max_log = max(max_log,log) 
                min_log = min(min_log,log)
      
            
            
