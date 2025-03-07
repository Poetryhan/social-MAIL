# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:07:36 2020

@author: uqjsun9
"""


from __future__ import print_function
import numpy as np
import LSTM.build_model as build_model
import scipy.io as scio
import pickle as pkl
from sklearn.metrics import mean_absolute_error
from scipy.spatial import distance
from nltk.translate.bleu_score import sentence_bleu
from shapely import geometry

def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a

poly = geometry.Polygon([(997,985),
(1045,987),
(1040,1017),
(997,1015),
(997,985)])

int_shape = np.load('int_map.npy',allow_pickle=True)
int_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape]
poly_int = geometry.Polygon(int_shape)

def in_insection(x,y):
   point = geometry.Point(x,y)
   if poly_int.contains(point):
       return True
   else :
       return False
   
num_agents = 13
experts = pkl.load(  open ('multi-agent-trj/expert_trjs/intersection_131_str.pkl','rb'))

n_obs = 20

x_data = []
y_data = []
x_test = []
y_test = []

    
for i in range(100):
     for j in range(num_agents):
         if experts[i]['ob'][j][0,0] !=0:
             x_data += [experts[i]['ob'][j][:,:]]
             y_data += [experts[i]['ac'][j][n_obs:]]
    
    # for j in range(np.size(cars,1)):
    #     x_data.append([onehot(int(k),88) for k in cars[:,j]])        
    
    
    
for i in range(100,129):
    for j in range(num_agents):
    
        x_test += [experts[i]['ob'][j]]
        y_test += [experts[i]['ac'][j]]
    
x_data = np.array(x_data)
y_data = np.array(y_data)
x_test = np.array(x_test)
y_test = np.array(y_test)
    





#%% model
#model.summary()

model=build_model.lstm_model(features=18,Lr=0.001)

# model=build_model.TCN_model()

history=model.fit(x_data,y_data,
          batch_size=64,
          epochs=10, validation_split=0.1)

#%%
sample_trajs = []
n_veh = 13

# dependency = np.load('dependency_dd.npy', allow_pickle = True)
test_index =[9, 15, 64, 28 ,89 ,93, 29, 8, 73 ,0 ,40, 36, 16, 11, 54 ,88, 62, 33 ,72 ,78 ]
test_index =[109, 104,115,100,117,127,128,125,116,117,126,108,109,100,110,108,122,104,119,116]
# test_index =[0]

for k in test_index:#range(100,120):
    # traj = np.zeros([180,n_veh])
    obss = np.squeeze (experts[k]['ob'] )
    obs = np.zeros([13,179,18])
    
    obs[:,:n_obs,:]=obss[:,:n_obs,:]
        
    # obs[-1,:n_obs,:2]=0.415 ###############    
        
    for i in range(0,179-n_obs):
        for j in range (n_veh):
            x_p = obs[j]
            x_p = x_p.reshape(-1,179,18)
            
            preds = model.predict(x_p, verbose=0)[0][i]
            
            n_state = move(obs[j][n_obs+i-1,0],obs[j][n_obs+i-1,1],obs[j][n_obs+i-1,2],obs[j][n_obs+i-1,3],preds[0],preds[1],obs[j][n_obs+i-1,4],obs[j][n_obs+i-1,5])
        
            obs[j][n_obs+i,:6] = n_state
            
        for j in range (n_veh):
            p_deg = np.degrees(np.arctan2(obs[j][n_obs+i-1,3],obs[j][n_obs+i-1,2])) % 360
            vehs = [ obs[ii][n_obs+i,:4]  for ii in range (n_veh) if ii != j]
        
            degreedif = [np.degrees(np.arctan2(ve[1]-obs[j][n_obs+i-1,1],ve[0]-obs[j][n_obs+i-1,0])) % 360 - p_deg for ve in vehs]                            
            distances = [np.linalg.norm(np.array([obs[j][n_obs+i-1,0],obs[j][n_obs+i-1,1]])-vehs[ii][:2]) if (abs(degreedif[ii])%360 < 80 or abs(degreedif[ii])%360>280) else 100 for ii in range(len(degreedif))]
            
            # distances = [np.linalg.norm(agent.state.p_pos-other.state.p_pos) for other in world.agents]
            index = sorted(range(len(distances)), key=lambda k: distances[k])
            
                        
            obs[j][n_obs+i,6:] = np.array([obs[j][n_obs+i-1,:4]- vehs[index[ii]] if vehs[index[ii]][0]>0 else vehs[index[ii]] for ii in range(3)] ).reshape([1,-1])
        
        # sample_trajs.append(preds)
    sample_trajs.append(obs)
    
    
        # obs = np.array(obss[0,j])
        # if j<=10:
        #     obs = np.array([np.random.randint(1,143+1)],dtype='float16')
        # traj[0,j] = obs
        # env.state = obs
#         for i in range(1,200):
            
#                 # obs = obss[i,j]
#             preds = model.predict(x_p, verbose=0)[0][i-1]
#             action = np.argmax(preds)
#             # print(action)
#             if action > 0 and  action < 5:
#                 obs = int(dependency[int(traj[i-1,j])][int(action)-1])
                    
                
#                 # x_p[0][i] = 0
#                 # x_p[0][i][int(obs)] = 1
                
#             # print(obs)
#             traj[i,j] = obs
#     sample_trajs.append(traj)
#   # env.render()
# np.save('sample_lstm_200.npy',sample_trajs)


#%%
def move(x,y,vx,vy,acc,yaw,dx,dy):
    
    p_deg = np.degrees(np.arctan2(vy,vx)) % 360

    if  x != 0:
        potential_x = x + vx *15/800
        potential_y = y + vy *15/800
                      
        speed = (vx **2 + vy **2) **0.5
        
        # dist = np.sqrt(np.sum(np.square(entity.state.p_des - entity.state.p_pos)))
        # potential_pos =  entity.action.u
        
        if in_insection(potential_x,potential_y):
            n_x = potential_x
            n_y = potential_y
            
            n_dx = dx - vx *15/800
            n_dy = dy - vy *15/800
            
            speed += acc/15
            if speed < 0:
                speed = 0
        
        # else: potential_pos = entity.state.p_pos
        # if entity.action.u[0] < 10 and entity.action.u[0] >0:
            # move_x = (entity.action.u[0]) * np.cos(np.deg2rad(entity.state.p_deg)) /15
            # move_y = (entity.action.u[0]) * np.sin(np.deg2rad(entity.state.p_deg)) /15
            p_deg += yaw
            # if np.sign(np.cos(np.deg2rad(entity.state.p_deg))) == np.sign(entity.state.p_vel[0] + move_x)
            n_vx = speed * np.cos(np.deg2rad(p_deg)) 
            n_vy = speed * np.sin(np.deg2rad(p_deg)) 
            
            
            # if x==0.415 and y == 0.415:
            #     n_x=n_y=0.4
            #     n_vx=n_vy=0
 
            
        else: 
            # entity.state.p_deg = 10
            
            n_x=n_y=n_vx=n_vy=n_dx=n_dy =0
            
    else: n_x=n_y=n_vx=n_vy=n_dx=n_dy =0
    return(np.array([n_x,n_y,n_vx,n_vy,n_dx,n_dy]))        

            
