# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:06:54 2020

@author: uqjsun9
"""
from shapely import geometry
import math
import numpy as np
import scipy.io as scio
# from sandbox.mack.acktr_disc import onehot
import pickle as pkl
import random
# import matplotlib.pyplot as plt

# int_shape = np.load('int_map.npy',allow_pickle=True)
# int_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape]
#
#
# intera_shape = [(997,985),
# (1045,987),
# (1040,1017),
# (997,1015),
# (997,985)]
# intera_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in intera_shape]
#
# int_shape2 = [tuple([0,0]), tuple([0,0.75]), tuple([1,0.75]) ,tuple([1,0]),tuple([0,0])]
#
# poly_int = geometry.Polygon(int_shape)
# poly_int2 = geometry.Polygon(int_shape2)
# poly_intera = geometry.Polygon(intera_shape)
# # plt.plot(a)
#
# def collision_intersection2(x,y, degree = 0, scale = 80):
#     # point = np.zeros([4,2])
#     x_diff = 40 *np.cos(np.radians(degree)) /scale
#     y_diff = 40 *np.sin(np.radians(degree)) /scale
#     # x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
#     # y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
#     point = geometry.Point([x + x_diff, y + y_diff])
#     point2 = geometry.Point([x, y])
#     # point[1] = [x + x_diff1, y + y_diff1]
#     # point[2] = [x - x_diff, y - y_diff]
#     # point[3] = [x - x_diff1, y - y_diff1]
#     # print(point)
#
#
#     if poly_int.contains(point):
#         return 1
#     else :
#
#         path = geometry.LineString([point, point2])
#         a = path.intersection(poly_int)
#         # path.distance(poly_int)
#
#         if not a.intersects(poly_int2.boundary):
#             return 50/scale - a.distance(point)
#         else:
#             return 1
# #
# def dis_stop_line (x,y, degree = 0, scale = 80):
#     # point = np.zeros([4,2])
#
#
#     x_diff = 50 *np.cos(np.radians(degree)) /scale
#     y_diff = 50 *np.sin(np.radians(degree)) /scale
#     # x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
#     # y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
#     point = geometry.Point([x + x_diff, y + y_diff])
#     point2 = geometry.Point([x, y])
#     # point[1] = [x + x_diff1, y + y_diff1]
#     # point[2] = [x - x_diff, y - y_diff]
#     # point[3] = [x - x_diff1, y - y_diff1]
#     # print(point)
#
#
#     if poly_intera.contains(point2):
#         return 1
#     else :
#         path = geometry.LineString([point, point2])
#         a = path.intersection(poly_intera)
#
#         if a.intersects(poly_intera):
#             return  a.distance(point2)
#         else:
#             return 1
#
#         path.distance(poly_int)
#
# dis_stop_line (0.42,0.58, degree = -90, scale = 80)
#
#  # collision_intersection2(0.8,0.4)
#
#
# def onehot(value, depth):
#     a = np.zeros([depth])
#     a[value] = 1
#     return a
# #%%
# gps_datass = np.load('gps_datass.npy',allow_pickle=True)
#
# for gps_datas in gps_datass:
#     for gps_data in gps_datas:
#         for i in range(180):
#             gps_data[i,7] = np.degrees(np.arctan2(gps_data[i,6],gps_data[i,5])) % 360
#             if gps_data[i,7] == 0 and gps_data[i,3]>0 and i>0:
#                 gps_data[i,7] = gps_data[i-1,7]
#
#
#         if max(gps_data[:,7])>0 :
#             if gps_data[0,7]==0:
#                 gps_data[0:np.where(gps_data[:,7] >0 )[0][0],7] = gps_data[np.where(gps_data[:,7] >0 )[0][0],7]
#         else:
#             gps_data[0,-1] = 0
#
#
#
#

#%%
# gps_datass = np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\gps_datass.npy',allow_pickle=True)
# init_num = np.zeros([131,12])
# ii = 0
#
# for gps_datas in gps_datass:
#     for gps_data in gps_datas:
#         for i in range(180):
#             gps_data[i,7] = np.degrees(np.arctan2(gps_data[i,6],gps_data[i,5])) % 360
#             if gps_data[i,7] == 0 and gps_data[i,3]>0 and i>0:
#                 gps_data[i,7] = gps_data[i-1,7]
#
#
#         if max(gps_data[:,7])>0 :
#             if gps_data[0,7]==0:
#                 gps_data[0:np.where(gps_data[:,7] >0 )[0][0],7] = gps_data[np.where(gps_data[:,7] >0 )[0][0],7]
#         else:
#             gps_data[0,-1] = 0
#
#
#         if gps_data[0,-1] == 1 :
#             if np.mean(gps_data[:,5]) > 0:
#                 if np.mean(gps_data[:,6]) >0:
#
#                     gps_data[:,-1] = 11
#                     init_num[ii,0] += 1
#                 else:
#                     gps_data[:,-1] = 13
#                     init_num[ii,2] += 1
#
#             else:
#                 if np.mean(gps_data[:,6]) <0:
#                     gps_data[:,-1] = 12
#                     init_num[ii,1] += 1
#
#                 else:
#                     gps_data[:,-1] = 14
#                     init_num[ii,3] += 1
#
#         if gps_data[0,-1] == 3 :
#             if np.mean((gps_data[:,5])) > 0:
#                 if np.mean(gps_data[:,6]) <0:
#
#                     gps_data[:,-1] = 31
#                     init_num[ii,8] += 1
#                 else:
#                     gps_data[:,-1] = 34
#                     init_num[ii,11] += 1
#
#             else:
#                 if np.mean(gps_data[:,6]) <0:
#                     gps_data[:,-1] = 33
#                     init_num[ii,10] += 1
#
#                 else:
#                     gps_data[:,-1] = 32
#                     init_num[ii,9] += 1
#
#         if gps_data[0,-1] == 2 :
#             if np.mean(abs(gps_data[:,5])) > np.mean(abs(gps_data[:,6])):
#                 if np.mean(gps_data[:,5]) >0:
#
#                     gps_data[:,-1] = 21
#                     init_num[ii,4] += 1
#                 else:
#                     gps_data[:,-1] = 22
#                     init_num[ii,5] += 1
#
#             else:
#                 if np.mean(gps_data[:,6]) <0:
#                     gps_data[:,-1] = 23
#                     init_num[ii,6] += 1
#
#                 else:
#                     gps_data[:,-1] = 24
#                     init_num[ii,7] += 1
#     ii += 1
#
# sum (init_num)
# gps_datass=np.delete(gps_datass, [29,97])
# init_num = np.delete(init_num, [29,97],axis=0)
#
# gps_datass=gps_datass[np.sum(init_num,axis=1)>=5]
#
# arr = np.arange(117)
# np.random.shuffle(arr)
#
# gps_datass = gps_datass[arr]

#%np.max(init_num,axis=0)

gps_datass = (list(np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\gps_datass_4.npy', allow_pickle= True)))
all_vehss = []
n_lr = 7
n_rl = 3
n_ud = 4
n_du = 4
n_agents = n_lr + n_rl + n_ud + n_du

n1 = list(range(0,7))
n2 = list(range(7,10))
n3 = list(range(10,14))
n4 = list(range(14,18))

n_steps = 3 # 180
init_pointss = []

sample_trajs = []
for i in range(len(gps_datass)):   #len(gps_datass)  # 对于每一个场景来说

    # init_points = [[],[],[],[]]
    gps_datas = gps_datass[i]
    all_vehs = []
    if len(gps_datas)>0 :
        print(i)
        all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [],[0 for k in range(n_agents)]
        init_points = []

        for k in range(n_agents):
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])


        for step in range(n_steps): # 对于每一步长来说
            action = [np.zeros([2]) for _ in range(n_agents)]
            obs = [np.zeros([8+4*4]) for _ in range(n_agents)] ##############
            # obs = [np.zeros([7+4*4]) for _ in range(n_agents)] ##############
            all_veh = []
            obs2 = []
            if step < n_steps-1:
                k1 =  0
                k2 = n_lr
                k3 = n_lr + n_rl
                k4 = n_lr + n_rl + n_ud
                for veh in gps_datas:  # 对于每辆车来说

                    print('veh的shape:',veh.shape,'veh是:',veh)
                    if veh[step][-1] != 4: all_veh.append(veh[step])
                    print('all_veh是：',all_veh)
                        # degree = np.degrees(np.arctan2(veh[:,5],veh[:,4])) % 360
                        # for ii in range(len(degree)):
                        #     if veh[ii,2] !=0 and degree[ii]=0:
                        #         if ii == 0:
                        #             state[i] = 1
                        #         else:
                        #             arr = np.where(state[i:] > 0)[0]
                        #             if len(arr)>0:
                        #                 state[i-1:arr[0]+i+1] = np.around(np.linspace(state[i-1], state[arr[0]+i], arr[0]+2))
                        #             else:
                        #                 state[i-1:] =  np.around(np.linspace(state[i-1], 1, steps2-i+1))

                        # direction = np.diff(degree[degree != 0])
                        # turn_degree = np.sum(direction[direction>min(direction)]) if len(direction)>0 else 0

                    acs = (veh[step+1,5]**2 + veh[step+1, 6]**2)**0.5 - (veh[step,5]**2 + veh[step, 6]**2)**0.5 # 计算加速度

                    if veh[step+1, 3]==0:
                        acs = (veh[step,5]**2 + veh[step, 6]**2)**0.5 - (veh[step-1,5]**2 + veh[step-1, 6]**2)**0.5

                    ac = np.zeros([2])
                    if veh[step+1, 3]>0:
                        ac[0] = acs
                        ac[1] = veh[step+1, 7] - veh[step, 7]
                        if ac[1] > 50:
                            ac[1] -= 360
                            # print (ac[1])
                        elif ac[1] < -50:
                            ac[1] += 360
                            # print (ac[1])

                        if abs(ac[1]) > 50:
                            ac[1] = 0
                        # ac[1] *=

                    print('where是什么：',veh[np.where(veh[:,3]>0)[0][-1],3],veh[np.where(veh[:,3]>0)[0][-1],3].shape)
                    print('where是什么：', veh[np.where(veh[:, 3] > 0)],veh[np.where(veh[:, 3] > 0)].shape)
                    print('where是什么：', veh[np.where(veh[:, 3] > 0)[0][-1]],veh[np.where(veh[:, 3] > 0)[0][-1]].shape)

                    state = veh[step, [3,4,5,6,8,9,7,10]]  # x,y,vx,vy,终点x,终点y,角度,标签(哪个进口道)

                    if state[0]>0:
                        print('有数据的veh编号：',veh[step, [1]])

                        state[2] = state[2]/15 # vx
                        state[3] = state[3]/15 # vy
                        state[4] = (veh[np.where(veh[:,3]>0)[0][-1],3]-state[0])/80 # 当前点到终点的Δx
                        state[5] = (veh[np.where(veh[:,3]>0)[0][-1],4]-state[1])/80 # 当前点到终点的Δy
                        state[0] = (state[0]-980)/80 # x
                        state[1] = (state[1]-970)/80 # y

                    #     if abs(state[5]) > 0.00001 or  abs(state[4]) > 0.00001:
                    #         state[6] = (np.degrees(np.arctan2(state[5],state[4])) % 360 - state[7]) %360
                    #         if state[6] >180:
                    #             state[6] -= 360
                    #         state[6] /= 180

                    #     else:
                    #         state[6] = 0
                    #     # if veh[step+1, 3]>0:
                    #     #     ac[0] = (veh[step+1, 3]-980)/80
                    #     #     ac[1] = (veh[step+1, 4]-970)/80
                    # else:
                    #     state[6] = 0

                    if veh[0,-1] ==21:  # 其中某个进口道的车
                        action[k1][:2] = ac
                        obs[k1][:8] = state
                        obs[k1][7] = 0
                        print('21进口道：',k1)
                        k1 += 1

                    elif veh[0,-1] ==22:
                        action[k2][:2] = ac
                        obs[k2][:8] = state
                        obs[k2][7] = 0
                        print('22进口道：', k2)
                        k2 += 1

                    # obs[k2][7] = 0


                    elif veh[0,-1] ==23:
                        action[k3][:2] = ac
                        obs[k3][:8] = state
                        obs[k3][7] = 0
                        print('23进口道：', k3)
                        k3 += 1

                    elif veh[0,-1] ==24:
                        action[k4][:2] = ac
                        obs[k4][:8] = state
                        obs[k4][7] = 0
                        print('24进口道：', k4)
                        k4 += 1
                if step == 0 :
                    init_points = obs  # 当前时刻的这个场景内所有车的state(8个,对于左转和直行交互的场景来说，是6个)


                # obs2 = obs
                all_veh = np.array(all_veh)[:,[3,4,5,6,7,10]]  # 当前场景内这一个步长的所有车辆的一些参数（x,y,vx,vy,角度,标签(哪个进口道)）
                print('新的all_veh是：', all_veh, all_veh.shape)
                all_veh[:,0] = (all_veh[:,0] - 980)/80
                all_veh[:,1] = (all_veh[:,1] - 970)/80
                all_veh[:,2:4] = all_veh[:,2:4]/15
                all_vehs.append(all_veh)
                print('更新的all_veh是：', all_veh, all_veh.shape)
                print('all_vehs是：', all_vehs)

                for k in range(n_agents):
                    if obs[k][1] > 0: # y大于0  这里的y已经归一化过了 y都在0,1内
                        # 找到交互影响对象
                        print('k是：',k,obs[k])
                        vehs = [all_veh[ii]  for ii in range(len(all_veh)) if (all_veh[ii][1] != obs[k][1]) and all_veh[ii][1] >0]  # 找到交叉口范围内不是k车的，其他周围车辆集合
                        print('周围的vehs：',vehs)
                        # vehs = [obs[ii][[0,1,2,3,6]]   for ii in range(n_agents) if (obs[ii][1]>0 and ii != k)]
                        if len(vehs)<5:
                            for _ in range(5): vehs.append(np.array([-1,-1,-1,-1,-1]))

                        degreedif = np.array( [np.degrees(np.arctan2(ve[1]-obs[k][1],ve[0]-obs[k][0])) % 360 - obs[k][6] for ve in vehs]   )%360  # 每一辆周围车和k车的角度与k车角度的差
                        degreespeeddif = np.array([(ve[4]- obs[k][6])%360 for ve in vehs] ) # 每一辆周围车和k车的角度差
                        distances = np.array([np.linalg.norm(obs[k][:2]-ve[:2]) for ve in vehs]) # 每一辆周围车和k车的距离

                        # index = np.logical_or(degreedif%360 < 90 , degreedif%360>270)
                        veh1 = []
                        veh2 = []
                        veh3 = []
                        veh4 = []
                        for ii in range(len(vehs)):
                            if (vehs[ii][0]>0 and (degreedif[ii]%360 < 90 or degreedif[ii]%360>270)):

                                if (degreespeeddif[ii]%360 < 20 or  degreespeeddif[ii]%360 >340):
                                    if (degreedif[ii]%360 < 10 or  degreedif[ii]%360 >350 and distances[ii]<20/80):
                                        veh1.append([vehs[ii], distances[ii]/10])
                                    elif (degreedif[ii]%360 < 20 or  degreedif[ii]%360 >340):
                                        veh1.append([vehs[ii],distances[ii]])
                                elif (abs(degreespeeddif[ii]%360 - 270) < 20 ) and (degreedif[ii]%360 <90):#
                                    veh2.append([vehs[ii], distances[ii]])

                                elif (abs(degreespeeddif[ii]%360 - 90) < 20 ) and (degreedif[ii]%360 >270):
                                    veh3.append([vehs[ii], distances[ii]])


                                elif (abs(degreespeeddif[ii]%360 - 180) < 20 ) :
                                    veh4.append([vehs[ii], distances[ii]])

                        veh1 = np.array(veh1,dtype=object)
                        veh2 = np.array(veh2,dtype=object)
                        veh3 = np.array(veh3,dtype=object)
                        veh4 = np.array(veh4,dtype=object)
                        veh_neig = []
                        # for ii in range(18):
                        #     plt.scatter(obs[ii][0],obs[ii][1], s = (ii+1)*5)
                        for veh_ in [veh1, veh2,veh3,veh4 ] :
                            if len(veh_) > 0:
                                veh_neig.append(veh_[ np.argmin(veh_[:,1])][0][:4])  # 找到四辆距离k车最近的
                                print('veh_neig:',veh_neig)
                            else: veh_neig.append(np.zeros(4))

                        # veh_neig = np.array(veh_neig)



                        # index = sorted(range(len(distances)), key=lambda k: distances[k])
                        # a= obs[k][6:] = np.array([obs[k][:4]- veh_neig[ii] if (veh_neig[ii][0]>0) \
                        #                         else np.ones(4)*(-2) for ii in range(4)] ).reshape([1,-1])

                        if k < 10:  # lr和rl两个方向的车
                            a = np.array([obs[k][[1,0,3,2]] - veh_neig[ii][[1,0,3,2],] if (veh_neig[ii][0]>0) \
                                                    else np.array([1,1,0,0]) for ii in range(4)] ).reshape([1,-1])
                            print('a:',a,a.shape)
                        else:
                            a = np.array([obs[k][:4]- veh_neig[ii] if (veh_neig[ii][0]>0) \
                                                    else np.array([1,1,0,0]) for ii in range(4)] ).reshape([1,-1])
                        # a[:,[0,1,4,5,8,9,12,13]] = abs(a[:,[0,1,4,5,8,9,12,13]])
                        obs[k][8:] = abs(a)
                        # obs[k] = abs(obs[k])


                        # obs[k][7] = dis_stop_line (obs[k][0],obs[k][1], obs[k][6])
                        # obs[k][6] = collision_intersection2(obs[k][0],obs[k][1], obs[k][6])

                    if action[k][0] < -100:
                        action[k] = action[k-1]
                    elif action[k][0] > 100:
                        action[k] = np.array([5,5])

                    # obs[k] = obs[k] #[[0,1,2,3,4,5]]
                    # obs2 = [abs(obs[ii][list(range(6))+list(range(7,23))]) for ii in range(len(obs))]
                    all_ob[k].append(obs[k])
                    all_ac[k].append(action[k])
                print('obs:',obs)
                all_agent_ob.append(np.concatenate(obs, axis=0))
                print('all_agent_ob:',all_agent_ob)

                for k in range(n_agents):

                    if action[k][0]!=0:
                        all_rew[k].append(1)
                        ep_ret[k] += 1
                    else:
                        all_rew[k].append(0)

        for k in range(n_agents):
            print('k为：', k)
            print('原来的all_ob[k]:', all_ob[k])
            all_ob[k] = np.squeeze(all_ob[k])
            print('现在的all_ob[k]:', all_ob[k].shape,all_ob[k])
        print('原来的all_agent_ob:', all_agent_ob)
        all_agent_ob = np.squeeze(all_agent_ob)
        print('现在的all_agent_ob:', all_agent_ob.shape,all_agent_ob)
        traj_data_all = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob
        }
        print('traj_data_all:',traj_data_all)
        sample_trajs.append(traj_data_all)
        print('sample_trajs:',sample_trajs)
    all_vehss.append(all_vehs)
    # init_pointss.append(init_points)
    # init_pointss = np.array(init_pointss)
pkl.dump(sample_trajs, open('multi-agent-trj/expert_trjs/intersection_131_str_5_4.pkl', 'wb'))
# np.save('init_pointss_4',init_pointss)
# np.save('gps_datass_4',gps_datass)
# np.save('all_vehss',all_vehss)
#%
