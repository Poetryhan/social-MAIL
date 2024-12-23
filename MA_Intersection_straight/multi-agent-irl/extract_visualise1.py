
import difflib 
import csv  
import numpy as np 
import matplotlib.pyplot as plt
from utils import map_vis_without_lanelet
import imageio
from shapely import geometry
import pandas as pd

poly = geometry.Polygon([(997,985),
(1045,987),
(1040,1017),
(997,1015),
(997,985)])


int_shape = np.load('int_map.npy',allow_pickle=True)
int_shape = [tuple(ac) for ac in int_shape]
poly_int = geometry.Polygon(int_shape)

# x,y = poly_int.exterior.xy
# plt.plot(x,y)

def differ(s1, s2):
    s = 0
    ss = []
    for k in s2:
        if k in s1:
            s += 1
        else:
            ss.append(k)
    return ss     

def in_insection(x,y):
   point = geometry.Point(x,y)
   if poly.contains(point):
       return True
   else :
       return False
   
def collision_intersection(x,y, degree = 0, scale = 1):
    point = np.zeros([4,2])
    x_diff = 2.693 *np.cos(np.radians(degree + 21.8)) /scale
    y_diff = 2.693 *np.sin(np.radians(degree + 21.8)) /scale
    x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
    y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
    point[0] = [x + x_diff, y + y_diff]
    point[1] = [x + x_diff1, y + y_diff1]
    point[2] = [x - x_diff, y - y_diff]
    point[3] = [x - x_diff1, y - y_diff1]
    # print(point)
    
    for i in range(4):
        point1 = geometry.Point(point[i])
    
        if poly_int.contains(point1):
            continue
        else : 
            return True
            
    return False

# x = 1000
# y=1000
# degree = -380
# collision_intersection(x,y, 0 , 1)
#%%
ind_data = pd.read_csv('C:/Users/uqjsun9/Desktop/inD-dataset-v1.0/data/18_tracks.csv')
ind_data_Meta = pd.read_csv('C:/Users/uqjsun9/Desktop/inD-dataset-v1.0/data/18_tracksMeta.csv')

for i in range(0,max(ind_data['trackId'])):
    veh = ind_data[ind_data['trackId'] == i]
    if ind_data_Meta[ind_data_Meta['trackId'] == max(veh['trackId'])]['class'].iloc[0] =='car':
    
        plt.plot(veh['xCenter'],veh['yCenter'], color = 'blue')
    else:
        plt.plot(veh['xCenter'],veh['yCenter'], color = 'red')
        
plt.xlim([40,70])
plt.ylim([-50,-20])


#%%
date =  '001'

ind_data = pd.read_csv('C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_EP1/vehicle_tracks_' + date + '.csv')
ind_data_Meta = pd.read_csv('C:/Users/uqjsun9/Desktop/inD-dataset-v1.0/data/18_tracksMeta.csv')

for i in range(0,max(ind_data['track_id'])):
    veh = ind_data[ind_data['track_id'] == i]
    # if ind_data_Meta[ind_data_Meta['trackId'] == max(veh['trackId'])]['class'].iloc[0] =='car':
    
    plt.plot(veh['x'],veh['y'], color = 'blue')
    # else:
    #     plt.plot(veh['xCenter'],veh['yCenter'], color = 'red')
        
# plt.xlim([40,70])
# plt.ylim([-50,-20])

#%%
period = 50
images = []
jj=13
 
for t in range(period):
    fig, axes = plt.subplots(1, 1)
    plt.figure(figsize=(10,5))
    
    map_path  = 'C:/Users/uqjsun9/Desktop/inD-dataset-v1.0/lanelets/location2.osm'
    aa = map_vis_without_lanelet.draw_map_without_lanelet(map_path, axes, 0, 0)
    plt.xlim([980,1060])
    plt.ylim([970,1030])

    # plt.plot(x,y)
    # for u in range(87):
    #     plt.scatter(a[u][0],a[u][1], c=5, vmin=0, vmax=15)
    # plt.scatter(1045,987, c=5, vmin=0, vmax=15)
    # plt.scatter(997,1015, c=5, vmin=0, vmax=15)
    # plt.scatter(1040,1017, c=5, vmin=0, vmax=15)

    # plt.scatter(aa[13][0],aa[13][1])
# for i in range(len(veh_data)):
#     veh = veh_data[i]
#     plt.scatter(veh[:,1],veh[:,2],c=veh[:,5])

    for veh in gps_datass[jj]:
        if veh[t,1] > 0 :
            plt.scatter([veh[t,1]],[veh[t,2]],c=[veh[t,5]], vmin=0, vmax=15)
            
    for ped in ped_datass[jj]:
        if ped[t,1] > 0 :
            plt.scatter([ped[t,1]],[ped[t,2]],c=[ped[t,5]], vmin=0, vmax=15, marker='*')
    plt.colorbar()
    plt.savefig('images/image.png')
    plt.close()
    
    image = imageio.imread('images/image.png')  
    images.append(image)       
images = np.array(images)
imageio.mimsave('images/gif-int.mp4', images,  fps=10)
#%%
a = []

i = 2
x = aa[i]
for ii in reversed(range(len(x[0]))):
    if (x[0][ii] > 980 and x[0][ii] < 1060) and (x[1][ii] > 970 and x[1][ii] < 1030):
        a.append([x[0][ii],x[1][ii]])


a.append([1006.5,1030])

ab = [tuple(ac) for ac in a]

# np.save('int_map',a)
#%%
images = []
jj=13
s = expertss[jj][0][:,0]
for t in range(len(expertss[jj])):
    # fig, axes = plt.subplots(1, 1)
    plt.figure(figsize=(10,5))
    
    map_path  = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/maps/DR_USA_Intersection_MA.osm'
    map_vis_without_lanelet.draw_map_without_lanelet(map_path, axes, 0, 0)

# for i in range(len(veh_data)):
#     veh = veh_data[i]
#     plt.scatter(veh[:,1],veh[:,2],c=veh[:,5])
    for veh in expertss[jj][t]:
        if veh[0] <1000:
            if veh[0] in s:
                plt.scatter([veh[2]],[veh[3]],c=[1], vmin=0, vmax=15)
            else:
                plt.scatter([veh[2]],[veh[3]],c=[10], vmin=0, vmax=15)
        else:
            if veh[0] in s:
                plt.scatter([veh[2]],[veh[3]],c=[1], vmin=0, vmax=15, marker='*')
            else:
                plt.scatter([veh[2]],[veh[3]],c=[10], vmin=0, vmax=15, marker='*')


    # plt.colorbar()
    plt.savefig('images/image.png')
    plt.close()
    
    image = imageio.imread('images/image.png')  
    images.append(image)       
images = np.array(images)
imageio.mimsave('images/gif-int.mp4', images,  fps=10)


#%%
expertss = []
gps_datass = []
period = 60*10    

dates = [str(a).rjust(3,'0') for a in range(22)]

date = '000'

for date in dates:
    
    veh_data = []
    data_path = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/vehicle_tracks_' + date + '.csv'
    data_path2 = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/pedestrian_tracks_' + date + '.csv'
    print(date)

    
    screen_shots = []  
    for t in range(3100):
        screen_shot = []
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)       
                
            for row in reader: 
                
                if reader.line_num > 1 and int(row[1])==t:
                                            # print(row[4])
                    if float(row[4]) > 980 and float(row[4]) < 1060 and float(row[5]) < 1030 and float(row[5]) > 970:                       
                        screen_shot.append(list([int(date),int(row[0]),int(row[1]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),0,0,0]))
                        
        screen_shot = np.array(screen_shot) 
        if len(screen_shot)>0:
            screen_shots.append(screen_shot)
            
    experts = []
    expert = []
    index = 0
    s1 = [0]
    for i in range(index,len(screen_shots)): 
        s = screen_shots[i][:,1]
        if  len(expert) == 0 and len(differ(s1,s))/len(s) > 0.8:
            if len(screen_shots[i])>= 10:
                expert.append(screen_shots[i])
                s1 = expert[0][:,1]
        elif len(expert) > 0: 
            s2 = screen_shots[i][:,1]
            if len(differ(s1,s2)) > 0:
                i_i = 0
                for k in differ(s1,s2):
                    kk = np.where(screen_shots[i][:,1] == k)[0][0]
                    if in_insection(screen_shots[i][kk,3], screen_shots[i][kk,4]):
                        i_i += 1
                if i_i == 0: 
                    expert.append(screen_shots[i])
                else:
                    experts.append(expert)
                    expert = []              
            else:
                expert.append(screen_shots[i])
                
                

    for j in range(len(experts)):
        for jj in range(len(experts[j])):
            t = int(experts[j][jj][0,1])         
            with open(data_path2) as csvfile:
                reader = csv.reader(csvfile)                   
                for row in reader: 
                        if reader.line_num > 1 and int(row[1])==t:
                                                    # print(row[4])
                            # print(j,jj)
                            ped = list([int(date),1000+int(row[0][1:]),int(row[1]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),0,0,0])
                            ped = np.array(ped).reshape(1,-1)
                            experts[j][jj]=np.concatenate((experts[j][jj],ped),axis=0)
    expertss += experts   
            
np.save('intsection_experts', expertss)
       # s1=[1,8,3,9,4,9,3,8,1,2,3] 
       # s2=[1,8,1,3,9,4,9,3,8,1,2,3]
       # overlap(s1, s2)

# for i in reversed(range(len(expertss))):
#     if len(expertss[i]) <100:
#         expertss.remove(expertss[i])

# expertss = np.load('intsection_experts.npy',allow_pickle=True)
# experts = expertss[0]
#%%
summary = np.load('summary1.npy',allow_pickle=True)
# gps_datass =np.load('intsection_MA.npz',allow_pickle=True)['arr_0']
# ped_datass =np.load('intsection_MA.npz',allow_pickle=True)['arr_1']
# i=1
period = 180

gps_datass = []
# gps_data = np.zeros([period,10])
a=[]
for i in range(162):
    experts = expertss[i]
    if len(experts) >= 50:
        gps_datas = []
        vehs = experts[0][:,1]
        # print(vehs)
        # start_time = experts[0][0,1]
        for j in vehs:
            gps_data = np.zeros([period,11])
            degree = 0
            for k in range(period):                
                if k < len(experts) and j in experts[k][:,1]:
                    index = np.where(experts[k][:,1] ==j)[0][0]
                    gps_data[k][:10] = experts[k][index,:]
                    
                    angle = np.arctan2(experts[k][index,:][5],experts[k][index,:][4])
                    degree = np.degrees(angle) % 360
                    if degree != 0:
                        gps_data[k,7] = degree
                    else:
                        gps_data[k,7] = gps_data[k-1,7] 
                    
                    index2 = np.where(np.logical_and(summary[:,0] == gps_data[k][0], summary[:,1] == gps_data[k][1]))[0][0]
                    gps_data[k,8:] = summary[index2,2:]
                    
            if gps_data[0,-1] ==  5 :
                gps_data[:,-1] = 2
                gps_data[:,8:10][gps_data[:,0]>0] = gps_data[:,3:5][gps_data[:,0]>0][-1]
                
                
                a.append(gps_data)
                # print(i)
              
            gps_datas.append(gps_data) 
        gps_datas = np.array(gps_datas)        
        gps_datass.append(gps_datas)            
            
            
np.save('gps_datass', gps_datass)     

#%%
init_pointss = []
init_num = np.zeros([131,4])
for i in range(len(gps_datass)):
    gps_datas = gps_datass[i]
    init_points = [[],[],[],[]]
    for j in range(len(gps_datas)): 
        veh = gps_datas[j]
        if max(veh[:,-1]) == 1:
            init_points[0].append(veh[0, [3,4,5,6,8,9]])
            init_num[i,0] += 1
        elif max(veh[:,-1]) == 2:
            init_points[1].append(veh[0, [3,4,5,6,8,9]])
            init_num[i,1] += 1
        elif max(veh[:,-1]) == 3:
            init_points[2].append(veh[0, [3,4,5,6,8,9]])
            init_num[i,2] += 1
        elif max(veh[:,-1]) == 4:
            init_points[3].append(veh[0, [3,4,5,6,8,9]])
            init_num[i,3] += 1
    init_points = [np.array(init_points[i]) for i in range(4)]   
    init_pointss.append(init_points)

np.save('init_pointss',init_pointss)
#%%

start_points = [[],[],[],[]]
init_pointss = []
init_num = np.zeros([131,4])

n = np.zeros([131,3])
for i in range(len(gps_datass)):
    gps_datas = gps_datass[i]
    init_points = [[],[],[],[]]
    for j in range(len(gps_datas)): 
        veh = gps_datas[j]
        
        angle = np.arctan2(veh[:,5],veh[:,4])
        degree = np.degrees(angle) % 360
        
        # gps_datass[i][j]
        if veh[0,0]<1000:
        
        
            direction = np.diff(degree[degree != 0])
            turn_degree = np.sum(direction[direction>min(direction)]) if len(direction)>0 else 0
            # print ('Start point:', veh[[degree != 0][0],1][0], veh[[degree != 0][0],2][0])
            if turn_degree > 30:
                # print('turn left:')
                # turn_flag = np.ones(600).T
                n[i,0] += 1
                if veh[0,0] == 0:
                    start_points[0].append(veh[degree != 0][0, [2,3,4,5]])
                else:
                    init_points[0].append(veh[0, [2,3,4,5]])
                    init_num[i,0] += 1
            elif turn_degree < -30:
                # print('turn right:')
                # turn_flag = (np.ones(600)*2).T
                n[i,2] += 1
                if veh[0,0] == 0:
                    start_points[2].append(veh[degree != 0][0, [2,3,4,5]])
                else:
                    init_points[2].append(veh[0, [2,3,4,5]])
                    init_num[i,2] += 1
            else:
                if len(veh[[degree != 0][0],2])!=0:
                
                
                    if veh[[degree != 0][0],2][0]> 1014 and veh[[degree != 0][0],3][0] >1029:
                        # print('turn left:')
                        # turn_flag = np.ones(600).T
                        n[i,0] += 1
                        if veh[0,0] == 0:
                            start_points[0].append(veh[degree != 0][0, [2,3,4,5]])
                        else:
                            init_points[0].append(veh[0, [2,3,4,5]])
                            init_num[i,0] += 1
                    else:
                        # print('go straight:')
                        # turn_flag = np.zeros(600).T
                        n[i,1] += 1
                        if veh[0,0] == 0:
                            start_points[1].append(veh[degree != 0][0, [2,3,4,5]])
                        else:
                            init_points[1].append(veh[0, [2,3,4,5]])
                            init_num[i,1] += 1
                            
                else: 
                    init_points[1].append(veh[0, [2,3,4,5]])
                    init_num[i,1] += 1
                        
        else:
            init_points[3].append(veh[0, [2,3,4,5]])
            init_num[i,3] += 1
            start_points[3].append(veh[veh[:,0] != 0][0, [2,3,4,5]])
    # for ped in ped_datass[i]:
        
    #     if ped[0,0] == 0:
    #         start_points[3].append(ped[ped[:,0] != 0][0, [1,2,3,4]])
    #     else:
    #         init_points[3].append(ped[0, 1:3])
    #         init_num[i,3] += 1
    init_pointss.append(init_points)        

start_points = [np.array(start_points[i]) for i in range(4)]
init_points = [np.array(init_points[i]) for i in range(4)]
# veh = gps_datas[-8]
fig = plt.figure(figsize=(10,5))
ax = fig.gca()
k = 3      
plt.hist (init_num[:,2]) 
plt.scatter(start_points[k][:,0],start_points[k][:,1])
plt.xlim([980,1060])
plt.ylim([970,1030])

# np.save('init_pointss',init_pointss)


#%%
ped_datass = []
period = 60*10    

dates = [str(a).rjust(3,'0') for a in range(22)]
for date in dates:
    
    veh_data = []
    data_path = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/pedestrian_tracks_' + date + '.csv'
    for i in range(20):
        veh = []
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)       
                
            for row in reader: 
                
                if reader.line_num > 1 and row[0]=='P'+ str(i):
                                            # print(row[4])
                    veh.append(list([int(row[1]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),(float(row[6])**2+float(row[7])**2)**0.5]))
                        
        veh = np.array(veh) 
        if len(veh)>0:
            veh_data.append(veh)
                            # 

    for ti in range(5):
        gps_datas = []
        for veh in veh_data:
            gps_data = np.zeros([period,7])
            for j in range(len(veh)):
                if veh[j][0]>ti*period and veh[j][0] < (ti+1)*period+1:
                    if veh[j][1] > 980 and veh[j][1] < 1060 and veh[j][2] < 1030 and veh[j][2] > 970:
                        gps_data[int(veh[j][0]%period)-1][:6] = veh[j][:]
                        angle = 0
                        if veh[j][3] !=0 :
                            angle = np.arctan2(veh[j][4],veh[j][3])
                        gps_data[int(veh[j][0]%period)-1][6] = angle
            if sum(gps_data[:,0])>0:
                gps_datas.append(gps_data)
        gps_datas = np.array(gps_datas)
        ped_datass.append(gps_datas)
        

# np.savez('intsection_MA.npz', gps_datass ,ped_datass)
# gps_datass =np.load('intsection_MA.npz',allow_pickle=True)['arr_0']
# ped_datass =np.load('intsection_MA.npz',allow_pickle=True)['arr_1']

#%%
# expertss = []
summary = []
# period = 60*10    

dates = [str(a).rjust(3,'0') for a in range(22)]

date = '000'
veh_data = []
for date in dates:
    
 
    data_path = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/vehicle_tracks_' + date + '.csv'
    data_path2 = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/pedestrian_tracks_' + date + '.csv'
    print(date)
    
    for t in range(300):
        veh = []
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)       
                
            for row in reader: 
                if reader.line_num > 1 and int(row[0])==t:
                                            # print(row[4])
                    if float(row[4]) > 980 and float(row[4]) < 1060 and float(row[5]) < 1030 and float(row[5]) > 970:

                        veh.append(list([int(date),int(row[0]), int(row[1]), float(row[4]),float(row[5]),float(row[6]),float(row[7])]))
                        
        veh = np.array(veh) 
        if len(veh)>0:
            veh_data.append(veh)
            
for date in dates:        
        data_path2 = 'C:/Users/uqjsun9/Documents/MA Intersection/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/DR_USA_Intersection_MA/pedestrian_tracks_' + date + '.csv'
        print(date)
        
        for t in range(20):
            veh = []
            with open(data_path2) as csvfile:
                reader = csv.reader(csvfile)       
                    
                for row in reader: 
                    if reader.line_num > 1 and t==int(row[0][1:]):
                                                # print(row[4])
                        if float(row[4]) > 980 and float(row[4]) < 1060 and float(row[5]) < 1030 and float(row[5]) > 970:

                            veh.append(list([int(date),int(row[0][1:])+1000, int(row[1]), float(row[4]),float(row[5]),float(row[6]),float(row[7])]))
                            
            veh = np.array(veh) 
            
            if len(veh)>0:
                veh_data.append(veh)
                
np.save('summary0', veh_data)
#%%
veh_data =np.load('summary0.npy',allow_pickle=True)
summary = []
for i in range(len(veh_data)):
    veh = veh_data[i]
    veh = veh[(veh[:,2]).argsort()]
    summary.append(list([veh[-1,0],veh[-1,1],veh[-1,3],veh[-1,4]]))
    
    if len(veh)<5:
        print(i)
        summary[-1].append(0)
        continue
    
    
    angle = np.arctan2(veh[:,6],veh[:,5])
    degree = np.degrees(angle) % 360
    direction = np.diff(degree[degree != 0])
    # if len(direction)>2 :
    
    direction = direction[direction>-20] 
    direction = direction[direction<20] 
    
    # turn_degree = np.sum(direction[direction>min(direction)]) if len(direction)>0 else 0
    # turn_degree = np.sum(direction[direction>min(direction)]) if len(direction)>0 else 0
    turn_degree = (np.sum(direction) if len(direction)>0 else 0)
    
    if turn_degree<-200 :
        turn_degree = turn_degree%360
    if turn_degree>200:
        turn_degree = turn_degree-360
    
    turn_degree2 = np.degrees(np.arctan2(veh[-1,6]-veh[0,6],veh[-1,5]-veh[0,5])) 
    distance1 = abs(veh[-1,3]-veh[0,3])
    distance2 = abs(veh[-1,4]-veh[0,4])
    
    if max(np.diff(veh[:,2]))>1:
        print(i)
        summary[-1].append(0)
    
    
    
    elif veh[0,1]<1000:
        
        
        
        # gps_datass[i][j]
        # if veh[0,0]<1000:
        

        # print ('Start point:', veh[[degree != 0][0],1][0], veh[[degree != 0][0],2][0])
        if distance1 >20 and distance2 >20 :
            if turn_degree > 40:
                # print('turn left:')                
                summary[-1].append(1)
                
            elif  turn_degree < -40:
                # print('turn right:')
                summary[-1].append(3)
                
            else: summary[-1].append(5)
            
        else:
            if (distance1 > 50 or distance2>50) and abs(turn_degree) < 40:
                # print('go straight:')
                summary[-1].append(2)
                
            else: 
                
                if abs(turn_degree) < 10 or (abs(turn_degree) < 20 and (distance1 < 3 or distance2 < 3)):                
                    # print('go straight:')
                    summary[-1].append(2)
                    
                else:
                    if turn_degree > 10 :
                        if (distance1 < 5 or distance2 < 5) and len(veh)>150 :
                            summary[-1].append(5)
    
                        # print('turn left:')                
                        else:summary[-1].append(1)
                        
                    elif  turn_degree < -10 :
                        # print('turn right:')
                        summary[-1].append(3)
                        
                    else: 
                        summary[-1].append(5)
        

                        
    else:
        if abs(turn_degree) < 10 and (distance1 > 50 or distance2>50):
            summary[-1].append(2)
        elif  (max(veh[:,4])>1025 or max(veh[:,3])>1058):
            summary[-1].append(1)
        else:        
            summary[-1].append(4)
        
        
    if i == 346:
        summary[-1][-1]=2
        
    if i ==1640:
        summary[-1][-1]=3
        # init_points[3].append(veh[0, [2,3,4,5]])
        # init_num[i,3] += 1
        # start_points[3].append(veh[veh[:,0] != 0][0, [2,3,4,5]])
summary = np.array (summary)        
# np.save('summary1', summary)
#%%
    aa = np.where(summary[:,4]==5)[0]

# for k in aa:
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    # k = 2900     
    # plt.hist (init_num[:,2]) 
    for k in aa:
        
        # if min(veh_data[k][:,3])<1000 and  max(veh_data[k][:,4])>1020: 
            # print(k)
            plt.scatter(veh_data[k][:,3],veh_data[k][:,4])
    
    plt.xlim([980,1060])
    plt.ylim([970,1030])