# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:29:31 2020

@author: uqjsun9
"""
import numpy as np
from irl.render import makeModel,render
import pickle as pkl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from shapely import geometry
from scipy.spatial import distance

poly = geometry.Polygon([(997,985),
(1045,987),
(1040,1017),
(997,1015),
(997,985)])

def in_insection(x,y):
    
    x = x*80+980
    y= y*80+970
    point = geometry.Point(x,y)
    if poly.contains(point):
        return True
    else :
        return False

experts = pkl.load(  open ('multi-agent-trj/expert_trjs/intersection_131_str_2.pkl','rb'))

int_shape = np.load('int_map.npy',allow_pickle=True)
int_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape]
poly_int = geometry.Polygon(int_shape)
x,y = poly_int.exterior.xy
# 

def mape(y_true, y_pred): 
    mask = y_true != 0
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true))

# num_links = 147
# num_vehs = 820
# num_agents = num_links + num_vehs
# n_steps = 31
# nn=147

    # sample_trajs = np.load('C:/Users/uqjsun9\Desktop\m0001_sample_trajs.npy', allow_pickle=True)


#%%plot 

for i in range(1):
    
    # print((init_pointss[i,0][0]-980)/80)
    
    traj_data = sample_trajs[i]

    vehicles=[obs[j][:,:4] for j in range(num_agents)]
    
    # print(i,vehicles[0][0])


#%
    # k = 17
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()     
    for k in range(num_agents):
        plt.scatter(vehicles[k][:,0],vehicles[k][:,1])
        
        plt.scatter(vehicles[k][0,0],vehicles[k][0,1], c=5)
    # plt.xlim([980,1060])
    # plt.ylim([970,1030])
    plt.plot(x,y)
    plt.xlim([0,1])
    plt.ylim([0,0.75])


#%%
actionss = []
for ii in range (100,120): #len(experts)
    # ii= 104
    expert = [experts[ii]['ob'][j][:,:4] for j in range(num_agents)]
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()     
    for k in range(13):
        plt.scatter(expert[k][:,0],expert[k][:,1])
        
        plt.scatter(expert[k][0,0],expert[k][0,1], c=5)
    plt.xlim([980,1060])
    plt.ylim([970,1030])
    plt.plot(x,y)
    plt.xlim([0,1])
    plt.ylim([0,0.75])
    
    actions = np.squeeze(experts[ii]['ac'])
    actionss.append(actions)
#%%    
yaw_rate = [[],[],[]]
acc_rate = [[],[],[]]

for ii in range(129):
    for j in range(13):
        for step in range(179):
            if abs(actionss[ii][j][step,1])>0.0 and actionss[ii][j][step,1]!=0 :
                if j<13:
                    yaw_rate[0].append(actionss[ii][j][step,1])
                elif j<20:
                    yaw_rate[1].append(actionss[ii][j][step,1])
                elif j<22:
                    yaw_rate[2].append(actionss[ii][j][step,1])
                # print (ii,j)
                
# min (yaw_rate[0]) = np.array(yaw_rate[0])
                
plt.hist(yaw_rate[0])
#%%    
    results = []
# for sample_trajs in sample_trajss:
    # sample_trajs = sample_trajss[8]
    speed_gen_all = []
    speed_gen_all_in = []
    speed_gen_all_out = []
    test_index = []
    rmse_all = []
    for sample_traj in sample_trajs:
        index = sorted(list(range(0, 18*num_agents, 18))+list(range(1, 18*num_agents+1, 18)))
        trajectories = np.array([sample_traj[j][:,:2] for j in range(num_agents)])
        speeds_gen = np.array([sample_traj[j][:,:4] for j in range(num_agents)])
        for speed_gen in speeds_gen:
            speed_gen_all += [(speed_gen[i,2] **2 + speed_gen[i,1] **3) **0.5 *15 for i in range(n_obs,179) if speed_gen[i,0]>0]
            speed_gen_all_in += [(speed_gen[i,2] **2 + speed_gen[i,1] **3) **0.5 *15 for i in range(n_obs,179) if in_insection(speed_gen[i,0], speed_gen[i,1])]
            speed_gen_all_out += [(speed_gen[i,2] **2 + speed_gen[i,1] **3) **0.5 *15 for i in range(n_obs,179) if (not in_insection(speed_gen[i,0], speed_gen[i,1]) and speed_gen[i,0]>0)]
        iii = 1000
        for ii in range(100,129):   
            trajs_expert = np.squeeze([experts[ii]['ob'][j][:,:2] for j in range(num_agents)])   
            if trajectories[0][0,0] != 0 and trajectories[0][0,0] in trajs_expert[:,0,0]:
                iii=ii
                test_index.append(iii)
        # iii = 9#107 
        expert = np.squeeze([experts[iii]['ob'][j][:,:2] for j in range(num_agents)])
        rmse = []
        
        for i in range(num_agents):
            traj_gen = trajectories[i]
            
            traj_index = np.where(expert[:,0,0] == traj_gen[0,0])
            if (traj_index[0].size)>0:
                traj_obs = expert[traj_index[0][0]]
            else:
                traj_obs = np.zeros([179,2])
            
            
            mask = np.logical_and(traj_obs[:,0] != 0 , traj_gen[:,0] != 0)     
            mask[:n_obs] = False
            mask[n_obs] = True
            
            rmse.append (mean_squared_error(traj_obs[mask], traj_gen[mask], squared=False))
            
        rmse = np.array(rmse)
        rmse_all.append(np.mean(rmse[rmse>0]))
        
    speed_gen_distr_in = np.histogram(speed_gen_all_in, bins=np.arange(20))
    speed_gen_distr_in_test =  speed_gen_distr_in[0]/np.sum(speed_gen_distr_in[0])   
    
    speed_gen_distr = np.histogram(speed_gen_all, bins=np.arange(20))
    speed_gen_distr_test =  speed_gen_distr[0]/np.sum(speed_gen_distr[0])   
    
    speed_gen_distr_out = np.histogram(speed_gen_all_out, bins=np.arange(20))
    speed_gen_distr_out_test =  speed_gen_distr_out[0]/np.sum(speed_gen_distr_out[0]) 
    
    dis = distance.jensenshannon(speed_distr_test, speed_gen_distr_test)
    dis_in = distance.jensenshannon(speed_distr_in_test, speed_gen_distr_in_test)
    dis_out = distance.jensenshannon(speed_distr_out_test, speed_gen_distr_out_test)
    
    results.append([np.mean(rmse_all), dis, dis_in, dis_out])
    
    print('rmse:', np.mean(rmse_all))
    print('JS dis:', dis)
    print('JS dis_in:', dis_in)
    print('JS dis_out:', dis_out)
    
    # plt.plot(speed_gen_distr_test)
    # plt.plot(speed_gen_distr_in_test)
    # plt.plot(speed_gen_distr_out_test)
    # plt.legend(['overall','in','out'])
# aa = np.array(results)        
#%%        
speed_all = []
speed_all_in = []
speed_all_out = []

for ii in test_index:    
    trajs_expert = np.squeeze([experts[ii]['ob'][j][:,:2] for j in range(num_agents)])
    actions = np.squeeze(experts[ii]['ac'])
    speeds = np.array([ (experts[ii]['ob'][j][:,:4]) for j in range(num_agents)])
    for speed in speeds:
        speed_all += [(speed[i,2] **2 + speed[i,1] **3) **0.5 *15 for i in range(179) if speed[i,0]>0]
        speed_all_in += [(speed[i,2] **2 + speed[i,1] **3) **0.5 *15 for i in range(179) if in_insection(speed[i,0], speed[i,1])]
        speed_all_out += [(speed[i,2] **2 + speed[i,1] **3) **0.5 *15 for i in range(179) if (not in_insection(speed[i,0], speed[i,1]) and speed[i,0]>0)]

speed_distr = np.histogram(speed_all, bins=np.arange(20))
speed_distr_test =  speed_distr[0]/np.sum(speed_distr[0])

speed_distr_in = np.histogram(speed_all_in, bins=np.arange(20))
speed_distr_in_test =  speed_distr_in[0]/np.sum(speed_distr_in[0])

speed_distr_out = np.histogram(speed_all_out, bins=np.arange(20))
speed_distr_out_test =  speed_distr_out[0]/np.sum(speed_distr_out[0])

# plt.plot(speed_distr2)
plt.plot(speed_distr_test)
plt.plot(speed_distr_in_test)
plt.plot(speed_distr_out_test)
plt.legend(['overall','in','out'])
# plt.hist(speed_distr[0])
# plt.hist(speed_all)
# plt.hist(speed_all_in)
# plt.hist(speed_all_out)



#%% match_accuracy
    k = 0 
    n_trajs = 0  
    mae_state = []   
    mape_travel_time = []  
    mape_departure_time = []
    traj_state = []
    traj_tt = []
    depart_time =[]
    
    for i in range(10):
    
        traj_data = sample_trajs[i]
        link_state = []
        link_state2 = []
        travel_time = []
        mae_state1 = []
        mae_state2 = []
        mae_traj_prob = []
        mape_travel_time_temp = []
        mape_departure_time_temp = []
        cars=traj_data["all_ob"][:,list(range(3*num_links, 3*num_agents, 3))]
        trajs =[]
        depart_time.append([0]*n_steps)
        
        for kk in range(num_vehs):
            arr = np.where(cars[:,kk]>1)
            # if arr[0].size>0:
            #     if arr[0][0]>0:
            #         depart_time[-1][arr[0][0]-1] +=1
                    
            if arr[0].size>0:
                if arr[0][0]>0:
                    depart_time[-1][arr[0][0]] +=1
                else: depart_time[-1][0] +=1
        depart_time[-1]= np.array(depart_time[-1])/ num_vehs
        
        
        # mape_departure_time.append()
        
    # depart_time =   np.array(depart_time) 
    
    
    
    
    
    
    
        for j in range(nn):
            
            state = list(traj_data["ob"][j][:,1])        
            link_state.append(state)
            link_state2.append([])
            
            # for step in range(n_steps):
                # state_tem = sum(cars[step]==j+1)
                
                # if state_tem>=3:
                #     link_state2[j].append(3)
                # elif state_tem<=1:
                #     link_state2[j].append(1)
                # else: link_state2[j].append(2)
                    
            time_tem=[]
            for car in range(num_vehs):
                time_tem.append(sum(cars[:,car]==j+1)) 
                
            if sum(np.array(time_tem)!=0)>0:
                travel_time.append(sum(time_tem)/sum(np.array(time_tem)!=0))
            else:
                travel_time.append(0)
            
        traj_state.append(link_state)    
        traj_tt.append(travel_time)    
        
        for ij in range(50):
            mae_state1.append(mean_absolute_error(base_states[ij], link_state))
            # mae_state2.append(mean_absolute_error(link_state, link_state2))
            mape_travel_time_temp.append(mape(base_travel_time[ij], travel_time))
            
            mape_departure_time_temp.append(mape(base_depart_time[ij], depart_time[-1]))
            
        # print(min(mae2))
        mae_state.append(min(mae_state1))   
        mape_travel_time.append(min(mape_travel_time_temp))
        mape_departure_time.append(min(mape_departure_time_temp))
        trajs_num = [0] * len(base_trajs_num)
        
        for j in range(num_links, num_agents):
            
            traj = list(traj_data["ob"][j][:,0])
            traj = sorted(set(traj), key=traj.index)
            n_trajs += 1
            trajs.append(traj)
            if len(traj)>1:
                if traj in base_trajs:
                    k += 1
                else:
                    for ii in range(len(base_trajs)):
                        if set(traj) <= set(base_trajs[ii]):
                            k += 1
                            break
                
                        
                if traj in base_trajs_set:
                    for ij in range(len(base_trajs_set)):                
                        if set(traj) == set(base_trajs_set[ij]):
                            trajs_num[ij] += 1
                else:
                    for ij in range(len(base_trajs_set)):
                        if set(traj) <= set(base_trajs_set[ij]):
                            trajs_num[ij] += 1
                        # break
        mae_traj_prob.append(mape(np.array(base_trajs_num)/sum(base_trajs_num),np.array(trajs_num)/sum(trajs_num)))
    aa_state1 = np.mean(traj_state, axis=0)  

    
    results.append([k, n_trajs, k/n_trajs,np.mean(mae_state), mean_absolute_error(aa_state, aa_state1),np.mean(mape_travel_time), np.mean(mae_traj_prob), np.mean(mape_departure_time)])                 
    print('match accuracy:', k, n_trajs, k/n_trajs)
    print('state mae:', np.mean(mae_state), mean_absolute_error(aa_state, aa_state1))
    print('travel time mape:', np.mean(mape_travel_time))
    print('traj prob mae:', np.mean(mae_traj_prob))
    print('depart prob mae:', np.mean(mape_departure_time))
    
    aa=np.array(results)
    reference=([[str(int(i)) for i in base_trajs2[j]] for j in range(len(base_trajs2))])

    bleu=np.zeros((len(trajs),4),dtype=np.float)
    
    for m in range(len(trajs)):
        candidate=[str(int(i)) for i in trajs[m]]
        bleu[m,0]=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu[m,1]=sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu[m,2]=sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu[m,3]=sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_results.append(np.mean(bleu,axis=0))
    bleu_results = np.array(bleu_results)

aa_depart1 = np.mean(depart_time, axis=0)
aa_tt1 = np.mean(traj_tt, axis=0)



mape(aa_tt, aa_tt1)
