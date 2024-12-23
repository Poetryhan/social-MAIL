# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:29:31 2020

@author: uqjsun9
"""
import numpy as np
from irl.render import mimic, render  , makeModel
import pickle as pkl
from sklearn.metrics import mean_absolute_error
from scipy.spatial import distance
from nltk.translate.bleu_score import sentence_bleu

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true))

def Bleu(base_trajs_set,trajs):
    reference=([[str(int(i)) for i in base_trajs_set[j]] for j in range(len(base_trajs_set))])

    bleu=np.zeros((len(trajs),4),dtype=np.float)
    
    for m in range(len(trajs)):
        candidate=[str(int(i)) for i in trajs[m]]
        bleu[m,0]=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu[m,1]=sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu[m,2]=sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu[m,3]=sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
       
    return np.mean(bleu,axis=0)

num_links = 0
num_vehs = 200
num_agents = num_links + num_vehs
n_steps = 80
nn=143

# num_links = 12
# num_vehs = 50 
# num_agents = num_links + num_vehs
# n_steps = 30 
# nn=12

#% create_base_trajs
expert_path='multi-agent-trj/expert_trjs/dronedata_veh143_8_200s.pkl'
with open(expert_path, "rb") as f:
    base_traj = pkl.load(f)
base_trajs = []
base_trajs_set = []
base_trajs_num = []
base_states = []
base_travel_time = []
base_depart_time = []

for i in range(15):
    traj_data = base_traj[i]
    base_states.append([])
    cars=traj_data["all_ob"][:,::1]
    base_travel_time.append([])
    
    base_depart_time.append([0]*n_steps)
    
    
    for k in range(num_vehs):
        arr = np.where(cars[:,k]>1)
        if arr[0].size>0:
            if arr[0][0]>0:
                base_depart_time[-1][arr[0][0]] +=1
            else: base_depart_time[-1][0] +=1
    base_depart_time[-1]= np.array(base_depart_time[-1])/(num_vehs)
        
# 
    # base_depart_time=np.array(base_depart_time)
    # np.mean(base_depart_time[:,1:], axis=1)
    
    
    
    for j in range(nn):
        time_tem=[]
        for car in range(num_vehs):
            time_tem.append(sum(cars[:,car]==j+1))
        if sum(np.array(time_tem)!=0)>0:
            base_travel_time[-1].append(sum(time_tem)/sum(np.array(time_tem)!=0))
        else:
            base_travel_time[-1].append(0)
        # base_travel_time[i].append(sum(time_tem)/sum(np.array(time_tem)!=0))    
    
    for j in range(num_links, num_agents):
        
        traj = list(traj_data["ob"][j][:,0])
        traj = sorted(set(traj), key=traj.index)
        if len(traj)>0:
                # if traj[0]==0:
                #     traj.remove(0.0)
                # if traj[-1]==144:
                #     traj.remove(144)
            base_trajs.append(traj)
        
        
        
base_trajs_set.append([])
base_trajs_num.append(0)   
for i in range(len(base_trajs)):
    traj_tem = base_trajs[i]
    a = 1
    for ii in range(len(base_trajs_set)):
        
        if set(traj_tem) == set(base_trajs_set[ii]):
            # base_trajs_num[ii] += 1
            a = 0
            break

        elif set(traj_tem) > set(base_trajs_set[ii]): 
            base_trajs_set[ii] = traj_tem
            # base_trajs_num[ii] += 1
            a = 0
            break
        elif set(traj_tem) < set(base_trajs_set[ii]):
            # base_trajs_num[ii] += 1
            a = 0
                  
    if a == 1:   
        base_trajs_set.append(traj_tem)
        # base_trajs_num.append(1)
base_trajs_num = [0] * len(base_trajs_set)
for traj in base_trajs:
    for ij in range(len(base_trajs_set)):
            if set(traj) <= set(base_trajs_set[ij]):            
                base_trajs_num[ij] += 1    
            
aa_state = np.mean(base_states, axis=0)            
aa_depart = np.mean(base_depart_time, axis=0)

base_travel_time2 = np.where(base_travel_time,base_travel_time,np.nan)
aa_tt = np.nanmean(base_travel_time2, axis=0) 
    
aa_tt[np.isnan(aa_tt)] = 0
# aaaa = set([i for item in base_trajs_set for i in (item)])
#%% rendering


sample_trajss= []
mids=['0001'] + [str(a).rjust(4,'0') for a in range(100,501,100)]
env_id = 'trj_network_drone'
model,env = makeModel(env_id)
# 

mid = '0100'
# sample_trajs = mimic(path, model,env)
for mid in mids:
    path = 'multi-agent-trj/logger/airl/trj_network_drone/decentralized/s-15/l-0.1-b-50-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-2/m_0'+mid
    # path = 'multi-agent-trj/logger/gail/trj_network/decentralized/s-200/l-0.01-b-1000-d-0.1-c-500/seed-1/m_0' + mid

    #sample_trajs = render(path, model,env)
    sample_trajs = mimic(path, model,env)
    sample_trajss.append(sample_trajs)
    
    np.save('sample_trajs_veh15_200s_mimic', sample_trajss)


#%%
# for iteration in range(len(mids)): 
    results = []
    bleu_results =[]
    
    num_links = 0
    # num_vehs = 200
    num_agents = num_links + num_vehs
    tt_mask = np.load('tt_mask_400.npy', allow_pickle = True)    
#     mid = mids[iteration]
#     # path = 'multi-agent-trj/logger/airl/trj_network2/decentralized/s-200/l-0.1-b-500-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-2/m_0'+mid
#     path = '/scratch/eait/uqjsun9/MA-AIRL-master/multi-agent-irl/multi-agent-trj/logger/airl/trj_network_drone/decentralized/s-50/l-0.1-b-50-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-1/m_0'+mid
#     # path = 'multi-agent-trj/logger/gail/trj_network/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500/seed-3/m_0' + mid
#     # path = 'multi-agent-trj/logger/gail/trj_network_drone/decentralized/s-400/l-0.1-b-50-d-0.1-c-200/seed-4/m_0'+mid

#     sample_traj = render(path, model,env)
#%%
#     sample_trajs.append(sample_traj)
sample_trajss = np.load('sample_trajs_veh15_200s_mimic.npy', allow_pickle = True)    
for sample_trajs in sample_trajss:
    #%% match_accuracy
    sample_trajs = sample_trajss[0]
    
    k = 0 
    n_trajs = 0  
    mae_state = []   
    mape_travel_time = []  
    mape_departure_time = []
    traj_state = []
    traj_tt = []
    depart_time =[]
    mae_traj_prob = []
    trajs_num = [0] * len(base_trajs_num)
    trajs= []
    
    for i in range(15,20):
        
        traj_data = sample_trajs[i]
        link_state = []
        link_state2 = []
        travel_time = []
        mae_state1 = []
        mae_state2 = []
        
        
        mape_travel_time_temp = []
        mape_departure_time_temp = []
        cars=traj_data["all_ob"][:,::1]#[i*180:i*180+80,::1]
    
        depart_time.append([0]*n_steps)
        
        for kk in range(num_vehs):
            arr = np.where((cars[:,kk]>1) & (cars[:,kk]<144))
            # if arr[0].size>0:
            #     if arr[0][0]>0:
            #         depart_time[-1][arr[0][0]-1] +=1
                    
            if arr[0].size>0:
                if arr[0][0]>0:
                    depart_time[-1][arr[0][0]] +=1
                else: depart_time[-1][0] +=1
        depart_time[-1]= np.array(depart_time[-1])/ num_vehs
        

        for j in range(nn):
            
          
            time_tem=[]
            for car in range(num_vehs):
                time_tem.append(sum(cars[:,car]==j+1)) 
                
            if sum(np.array(time_tem)!=0)>0:
                travel_time.append(sum(time_tem)/sum(np.array(time_tem)!=0))
            else:
                travel_time.append(np.nan)
            
        # traj_state.append(link_state)    
        traj_tt.append(travel_time)    
            
        # for ij in range(20):
        #     # mae_state1.append(mean_absolute_error(base_states[ij], link_state))
        #     # mae_state2.append(mean_absolute_error(link_state, link_state2))
        #     mape_travel_time_temp.append(mape(base_travel_time[ij], travel_time))
            
        #     mape_departure_time_temp.append(mape(base_depart_time[ij], depart_time[-1]))
            
        # print(min(mae2))
        # mae_state.append(min(mae_state1))   
        # mape_travel_time.append(min(mape_travel_time_temp))
        # mape_departure_time.append(min(mape_departure_time_temp))
        
        
        for j in range(num_links, num_agents):
            
            traj = list(cars[:,j])
            traj = sorted(set(traj), key=traj.index)
            if traj[0] == 144:
                traj = traj[1:]+[traj[0]]
            
            n_trajs += 1
            if len(traj)>0:
                # if traj[0]==0:
                #     traj.remove(0.0)
                # if traj[-1]==144:
                #     traj.remove(144)
                trajs.append(traj)    
                
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
                            a
                            trajs_num[ij] += 1
                        # break
        mae_traj_prob.append(mape(np.array(base_trajs_num)/sum(base_trajs_num),np.array(trajs_num)/sum(trajs_num)))
    aa_state1 = np.mean(traj_state, axis=0)      
    dis = distance.jensenshannon (np.array(trajs_num)/np.sum(trajs_num), np.array(base_trajs_num)/np.sum(base_trajs_num))                
    aa_tt1 = np.nanmean(traj_tt, axis=0)
    aa_tt1[np.isnan(aa_tt1)] = 1
    aa_depart1 = np.mean(depart_time, axis=0)
    aa_tt2 = np.stack([aa_tt[tt_mask],aa_tt1[tt_mask]],axis=-1)
    
    route_times = np.zeros([len(base_trajs_set),2])
    for k in range(len(base_trajs_set)):
    # k=9
        route = np.array(base_trajs_set[k])
        route_agg_time = np.zeros([len(route),2])
        route_time = np.zeros([len(route),2])
        for i in range(len(route)):
            if not (route[i] ==0 or route[i] ==144):
                route_agg_time[i, 0] = route_agg_time[i-1, 0]+aa_tt[int(route[i])-1]
                route_agg_time[i, 1] = route_agg_time[i-1, 1]+aa_tt1[int(route[i])-1]
                route_time[i, 0] = aa_tt[int(route[i])-1]
                route_time[i, 1] = aa_tt1[int(route[i])-1]
                mape(route_time[:, 0],route_time[:, 1])
        route_times[k,0] = max(route_agg_time[:, 0])
        route_times[k,1] = max(route_agg_time[:, 1])
    rt = mape(route_times[:,0],route_times[:,1])
    
    
    
    results.append([k, n_trajs, k/n_trajs,rt,0,mape(aa_tt[tt_mask],aa_tt1[tt_mask]), np.mean(mae_traj_prob), mape(aa_depart,aa_depart1),dis])                 
    print('match accuracy:', k, n_trajs, k/n_trajs)
    # print('state mae:', np.mean(mae_state), mean_absolute_error(aa_state, aa_state1))
    print('travel time mape:', mape(aa_tt[tt_mask],aa_tt1[tt_mask]),rt) #np.mean(mape_travel_time)
    print('traj prob mae:', np.mean(mae_traj_prob))
    print('depart prob mae:', mape(aa_depart,aa_depart1))
    print('JS-distance:',dis)
    
    bleu_results.append(Bleu(base_trajs_set,trajs)) 
    
    aa=np.array(results)

bleu_results = np.array(bleu_results)

traj_tt = np.where(traj_tt,traj_tt,np.nan)
# aa_tt1 = np.nanmean(traj_tt, axis=0)
# base_travel_time[np.isnan(base_travel_time)] = 0
# aa_tt[np.isnan(aa_tt)] = 0

mape(aa_tt,aa_tt1)
distance.jensenshannon(aa_depart,aa_depart1)
distance.jensenshannon(aa_tt,aa_tt1)
mape(aa_depart,aa_depart1)


mape(aa_tt[tt_mask],aa_tt1[tt_mask])



travel_time = np.where(travel_time,travel_time,np.nan)
travel_time[np.isnan(travel_time)] = 1
# reference=([[str(int(i)) for i in base_trajs_set[j]] for j in range(len(base_trajs_set))])
# # candidate=[[str(i) for i in trajs[j]] for j in range(len(trajs))]

# bleu=np.zeros((len(trajs),4),dtype=np.float)

# for m in range(len(trajs)):
#     candidate=[str(int(i)) for i in trajs[m]]
#     bleu[m,0]=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#     bleu[m,1]=sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#     bleu[m,2]=sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#     bleu[m,3]=sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

# np.mean(bleu,axis=0)

# def MAE_PER(y_true, y_pred): 
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     mask = y_true != 0
#     y_true, y_pred = y_true[mask], y_pred[mask]
#     return np.mean(np.abs((y_true - y_pred))) / np.mean(y_true)

# def mape_loss_func2(label,pred):
#     pred = np.clip(pred,1,10)
#     return np.fabs((label-pred)/np.clip(label,1,10)).mean()

# mape_loss_func2(aa_tt,aa_tt1)
