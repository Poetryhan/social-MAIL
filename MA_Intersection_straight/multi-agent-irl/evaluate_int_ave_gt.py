# -*- coding: utf-8 -*-
"""
Created on Wang Shihan Dec 13:29:31 2023

@author: uqjsun9
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from irl.render import makeModel, render, get_dis
from irl.mack.kfac_discriminator_airl import Discriminator
import pickle as pkl
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from shapely import geometry
import math
from scipy.spatial import distance

from irl.render import makeModel, render, get_dis
from irl.mack.kfac_discriminator_airl import Discriminator
import pickle as pkl
import pickle
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from shapely import geometry
from scipy.spatial import distance
from scipy import interpolate

from matplotlib.widgets import Button, Slider
from utils.DataReader import read_tracks_all, read_tracks_meta, read_light
import tensorflow as tf

# 画交叉口的代码
try:
    import lanelet2

    use_lanelet2_lib = True
    from utils import map_vis_lanelet2
except ImportError:
    import warnings

    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

# from loguru import logger
import glob
import os
import argparse

def args_parser():
    configs = argparse.ArgumentParser(description='')

    configs.add_argument('--path', default="D:/Study/同济大学/博三/面向自动驾驶测试的仿真/数据/SinD"
                                           "/github/SinD-main/SinD-main/Data/",
                         help="Dir with track files", type=str)
    configs.add_argument('--record_name', default="8_3_1",
                         help="Dir with track files", type=str)
    configs.add_argument('--plot_traffic_light', default=True,
                         help="Optional: decide whether to plot the traffic light state or not.",
                         type=bool)

    configs.add_argument('--behaviour_type', default=True,
                         help="Optional: decide whether to show the vehicle's violation behavior by color of text or not.",
                         type=bool)

    configs.add_argument('--skip_n_frames', default=3,
                         help="Skip n frames when using the second skip button.",
                         type=int)
    configs.add_argument('--plotTrackingLines', default=True,
                         help="Optional: decide whether to plot the direction lane intersection points or not.",
                         type=bool)
    configs.add_argument('--plotFutureTrackingLines', default=True,
                         help="Optional: decide whether to plot the tracking lines or not.",
                         type=bool)

    configs = vars(configs.parse_args())

    return configs

poly = geometry.Polygon([(5.21, 34.36),
                                     (2.10, 27.47), (-4.50, 25.39),
                                     (-4.13, 6.41),
                                     (3.87, 4.70),
                                     (7.29, -2.64),
                                     (22.26, -2.57),
                                     (25.23, 3.74),
                                     (33.61, 6.41),
                                     (33.83, 25.76),
                                     (24.56, 28.06),
                                     (20.34, 34.73)])

def in_insection(x, y):
    x = x * 38 - 4
    y = y * 23 + 14
    point = geometry.Point(x, y)
    if poly.contains(point):
        return True
    else:
        return False


experts = pkl.load(open(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs\MA_Intersection_straight' \
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1.pkl','rb'))
init_pointss = np.load(
            r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs\MA_Intersection_straight' \
                      r'\AV_test\DATA\init_sinD_nvnxuguan_9jiaohu_social_dayu1.npy', allow_pickle=True)


def mape(y_true, y_pred):
    mask = y_true != 0
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true))


num_agents = 8
num_left = 3
num_straight = 5

# % rendering
results = []

# path = 'multi-agent-trj/logger/airl/trj_network/decentralized/s-200/l-0.1-b-489-d-0.1-c-489-l2-0.1-iter-1-r-0.0/seed-2/m_0'+mid
# path = 'multi-agent-trj/logger/gail/trj_network/decentralized/s-200/l-0.01-b-1000-d-0.1-c-489/seed-1/m_0' + mid
bleu_results = []
env_id = 'trj_intersection_4'
model = makeModel(env_id)

# mids=['0100']
# %%
sample_trajss = []
mids = ['0001'] + [str(a).rjust(4, '0') for a in range(10, 571, 30)] + ['0570'] # 1001  (50, 1001, 50)
# mids = ['0310']
print('mids:', len(mids),mids)
# 上一行的print结果如下 mids: ['0001', '0100', '0200', '0300', '0400', '0489']

scenario_test = 79  #  6

# # # mids=['0001']
# for iteration in range(len(mids)):
#     mid = mids[iteration]
#     path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight' \
#            r'\multi-agent-irl\irl\mack\multi-agent-trj\logger\airl\trj_intersection_4\decentralized' \
#            r'\v13\l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0\seed-13\m_0' + mid
#     # for i in range (450,451):
#     # 在每次循环开始时清除默认图
#     # tf.reset_default_graph()
#     sample_trajs = render(path, model, env_id, mid)
#     sample_trajss.append(sample_trajs)
#     # sample_trajs = np.load('C:/Users/uqjsun9\Desktop\m0001_sample_trajs.npy', allow_pickle=True)
# # pkl.dump(sample_trajss, open(r'E:\wsh-科研\nvn_xuguan_sind\sinD_nvn_xuguan_9jiaohu_history_LSTM_ATT-GPU'
# #                              r'\MA_Intersection_straight\results_evaluate\v5\有注意力的视频\专门生成生成轨迹的数据\sample_trajss_v5.pkl', 'wb'))
# # 指定文件夹路径
# folder_path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
#               r'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频\专门生成生成轨迹的数据'
# # 将 scenario 加入文件夹路径中
# folder_path_with_scenario = os.path.join(folder_path, str(scenario_test))
#
# # 创建文件夹（如果不存在）
# os.makedirs(folder_path_with_scenario, exist_ok=True)
# # scenario 名称
# scenario_name = "sample_trajss_" + str(scenario_test)
#
# # 构建保存 pkl 文件的完整路径
# pkl_file_path = os.path.join(folder_path_with_scenario, f"{scenario_name}.pkl")
#
# # 将数据 sample_trajss 存入 pkl 文件
# with open(pkl_file_path, 'wb') as f:
#     pkl.dump(sample_trajss, f)

path_read = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
           fr'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频\专门生成生成轨迹的数据' \
           fr'\{scenario_test}\sample_trajss_{scenario_test}.pkl'
f_read = open(path_read,'rb')
sample_trajss = pkl.load(f_read)

# %%评估plot 位置 位置rmse actionrmse a_rush a_yield
root = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
       r'\MA_Intersection_straight\results_evaluate\v13'
# 下面这一段代码是画出专家轨迹
actionss = []
expert = []
ini_point = []
expert_ac = []
sv = []  # range(14,18)

# scenario_test_add1 = 152
for ii in range(scenario_test, scenario_test + 1):  # 原来是这个(107,108)，对应着第107个场景 len(experts) 112  测试场景
    # ii= 104
    expert = [experts[ii]['ob'][j][:, :] for j in range(num_agents)]
    ini_point = init_pointss[ii]
    expert_trj_all = pd.DataFrame()  # 存放输出的专家轨迹
    # print(np.array(experts[ii]['ob'][0][:, :2]))

    fig = plt.figure(figsize=(6, 3 / 4 * 6))
    # fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    for k in range(num_agents):
        if k not in sv:
            trj = pd.concat([pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                            axis=1)  # 横向拼接

            if ini_point[k][0] != 0:

                x = expert[k][:, 0] * 38 - 4
                y = expert[k][:, 1] * 23 + 14

                non_zero_points = (x != -4) & (y != 14)
                print('non_zero_points',non_zero_points)
                print('x[non_zero_points], y[non_zero_points', x[non_zero_points], y[non_zero_points])

                plt.scatter(x[non_zero_points], y[non_zero_points], s=10, zorder=100, c='none', marker='o', edgecolors='g')  # 整条轨迹
                plt.scatter(expert[k][0, 0] * 38 - 4, expert[k][0, 1] * 23 + 14, c='k', s=20, zorder=100,
                            marker='o')  # 轨迹的初始点 黑色
                # plt.scatter(expert[k][:, 0] * 39 - 5, expert[k][:, 1] * 38 - 3, s=10)  # 整条轨迹
                # plt.scatter(expert[k][0, 0] * 39 - 5, expert[k][0, 1] * 38 - 3, c='k', s=20)  # 轨迹的初始点 黑色
                expert_trj_all = pd.concat([expert_trj_all, trj], axis=0)  # 纵向拼接


    # plt.xlim(-5, 34)
    # plt.ylim(-3, 35)

    # 把交叉口的范围画出来
    config = args_parser()

    map_path = glob.glob(config['path'] + '/*.osm')[0]
    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
        laneletmap = lanelet2.io.load(map_path, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
    else:
        # 这里你需要将之前的画交叉口边界的代码整合到这里
        map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

    plt.savefig(root + '\轨迹对比图片' + '\%s.png' % ('real_scenario_' + str(ii) + ''), dpi=300)
    # expert_trj_all.to_csv(root + '\scenario_0_trj.csv')
    plt.close()

    actions = np.squeeze(experts[ii]['ac'])
    actionss.append(actions)


# 专家的速度和加速度分布
yaw_rate = [[], []]
acc_rate = [[], []]

# print('专家的vehicles:', np.shape(vehicles))
# print('专家的actions:', np.shape(actionss)) # 专家的actions: (1, 18, 222, 2)

for ii in range(1):  # 原来是129，好像是所有场景的个数
    for j in range(8):  # 一个场景内的每一个agent
        for step in range(185):
            if j < 3:
                acc_rate[0].append(((9.2 * (actionss[ii][j][step][0] + 1)) / 2) - 4.8)
                yaw_rate[0].append(((4.5 * (actionss[ii][j][step][1] + 1)) / 2) - 0.8)
                #
                # acc_rate[0].append(actionss[ii][j][step][0] * 4.9)
                # yaw_rate[0].append(actionss[ii][j][step][1] * 3.7)

            elif 3 <= j:
                acc_rate[1].append(((8.5 * (actionss[ii][j][step][0] + 1)) / 2) - 3.6)
                yaw_rate[1].append(((2.4 * (actionss[ii][j][step][1] + 1)) / 2) - 1.2)

                # acc_rate[1].append(actionss[ii][j][step][0] * 4.9)
                # yaw_rate[1].append(actionss[ii][j][step][1] * 3.7)

# 画出生成轨迹 以及 加速度 角度变化和专家轨迹真实值的对比
sv = []

generate_expert_trj_all = pd.DataFrame()  # 存放所有模型的轨迹结果
generate_expert_acc_yaw_all = pd.DataFrame()  # 存放所有模型的action结果

# for k in range(66):
for ii in range(21):  # 模型迭代个数 21
    sample_trajs = sample_trajss[ii]
    generate_expert_trj_one_model = pd.DataFrame()  # 存放一个模型这个场景的专家轨迹和生成轨迹
    generate_expert_acc_yaw_one_model = pd.DataFrame()  # 存放一个模型这个场景的专家轨迹和生成轨迹
    # print('sample_trajs：',np.shape(sample_trajs))
    for i in range(1):  # 这个好像是和render，py文件中的num_trajs对应的，一个并行的模型跑两次，得到了两次结果,这里取1
        traj_data = sample_trajs[i]  # 一个场景中的所有轨迹
        # print('traj_data：', np.shape(traj_data['ob']))
        vehicles = [traj_data["ob"][j] for j in range(num_agents)]
        actions = [traj_data["ac"][j] for j in range(num_agents)]
        # print('vehicles:', np.shape(vehicles))
        print('actions:', np.shape(actions))
        # print('评估的actions为!!!!!!!!!!!!!!!!!!!!!!!:',np.shape(actions),actions,actions[1])

        # print(i,vehicles[0][0])
        yaw_rate_gerente = [[], []]
        acc_rate_gerente = [[], []]
        yaw_rate_expert = [[], []]
        acc_rate_expert = [[], []]

        # 画出左转车的轨迹，并存储轨迹数据
        for k in range(num_agents):

            if k < 3:
                for step in range(230):
                    acc_rate_gerente[0].append(((9.2 * (actions[k][step][0] + 1)) / 2) - 4.8)
                    yaw_rate_gerente[0].append(((2.8 * (actions[k][step][1] + 1)) / 2) - 0.3)


                    if actions[k][step][1]>1.2 or actions[k][step][1]<-1.3:
                        print('生成轨迹 不对的数据，转角超出阈值：',actions[k][step][1])

                    # acc_rate_gerente[0].append(actions[k][step][0] * 4.9)
                    # yaw_rate_gerente[0].append(actions[k][step][1] * 3.7)
                for step in range(185):
                    acc_rate_expert[0].append(((9.2 * (actionss[i][k][step][0] + 1)) / 2) - 4.8)
                    yaw_rate_expert[0].append(((2.8 * (actionss[i][k][step][1] + 1)) / 2) - 0.3)

                    # acc_rate_expert[0].append(actionss[i][k][step][0] * 4.9)
                    # yaw_rate_expert[0].append(actionss[i][k][step][1] * 3.7)

            # print('acc_rate_gerente[0]:',np.shape(acc_rate_gerente[0]), acc_rate_gerente[0])


            if k not in sv and k < 3:
                fig = plt.figure(figsize=(6, 3 / 4 * 6))

                x_g = vehicles[k][:, 0] * 38 - 4
                y_g = vehicles[k][:, 1] * 23 + 14

                non_zero_points_g = (x_g != -4) & (y_g != 14)

                x_e = expert[k][:, 0] * 38 - 4
                y_e = expert[k][:, 1] * 23 + 14

                non_zero_points_e = (x_e != -4) & (y_e != 14)

                plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g], s=10, zorder=100, c='none', marker='o',
                            edgecolors='b')  # 车辆的x和y # 生成轨迹 蓝色
                plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                            marker='o', edgecolors='k')  # 车辆初始时刻的位置 黑色
                plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, zorder=99, c='none', marker='o',
                            edgecolors='g')  # 专家整条轨迹 # 绿色

                # plt.scatter(vehicles[k][:, 0] * 39 - 5, vehicles[k][:, 1] * 38 - 3, c='b', s=10)  # 车辆的x和y # 生成轨迹 蓝色
                # plt.scatter(vehicles[k][0][0] * 39 - 5, vehicles[k][0][1] * 38 - 3, c='k', s=20)  # 车辆初始时刻的位置 黑色
                # plt.scatter(expert[k][:, 0] * 39 - 5, expert[k][:, 1] * 38 - 3, c='g', s=30)  # 专家整条轨迹 # 绿色
                # plt.xlim(-5, 34)
                # plt.ylim(-3, 35)

                # 把交叉口的范围画出来
                config = args_parser()

                map_path = glob.glob(config['path'] + '/*.osm')[0]
                if use_lanelet2_lib:
                    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                    laneletmap = lanelet2.io.load(map_path, projector)
                    map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
                else:
                    # 这里你需要将之前的画交叉口边界的代码整合到这里
                    map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

                plt.savefig(root + '\轨迹对比图片' + '\%s.png' % (
                            'scenario_' + str(scenario_test) + 'model_' + str(ii) + 'left_vehicle_' + str(k)), dpi=300)
                plt.close()


                generate_trj_left = pd.concat(
                    [pd.DataFrame(vehicles[k][:, 0] * 38 - 4), pd.DataFrame(vehicles[k][:, 1] * 23 + 14)],
                    axis=1)  # 横向拼接

                generate_action_left = pd.concat(
                    [pd.DataFrame(acc_rate_gerente[0][0+k*230:230+k*230]), pd.DataFrame(yaw_rate_gerente[0][0+k*230:230+k*230])], axis=1)

                generate_angle_left = pd.DataFrame(vehicles[k][:, 6] * 191 - 1)  # 横向拼接
                generate_v_left = pd.DataFrame(np.sqrt((vehicles[k][:, 2].astype(float) * 21 - 14)**2 + (vehicles[k][:, 3].astype(float) * 12 - 2)**2))  # 横向拼接
                # generate_vy_left = pd.DataFrame(vehicles[k][:, 3] * 20 - 1)  # 横向拼接

                expert_trj_left = pd.concat(
                    [pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                    axis=1)  # 横向拼接

                expert_action_left = pd.concat(
                    [pd.DataFrame(acc_rate_expert[0][0+k*185:185+k*185]), pd.DataFrame(yaw_rate_expert[0][0+k*185:185+k*185])], axis=1)

                expert_angle_left = pd.DataFrame(expert[k][:, 6] * 191 - 1)  # 横向拼接
                expert_v_left = pd.DataFrame(np.sqrt((expert[k][:, 2] * 21 - 14) ** 2 + (expert[k][:, 3] * 12 - 2) ** 2))

                # expert_action_left = pd.concat([pd.DataFrame(acc_rate[0][-41:]), pd.DataFrame(yaw_rate[0][-41:])], axis=1)

                generate_expert_trj_left = pd.concat([generate_trj_left, expert_trj_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, generate_action_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, generate_angle_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, expert_action_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, expert_angle_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, generate_v_left], axis=1)
                generate_expert_trj_left = pd.concat([generate_expert_trj_left, expert_v_left], axis=1)

                generate_expert_trj_left['agent_id'] = k
                generate_expert_trj_left['model_id'] = ii
                generate_expert_trj_left['direction'] = 'left'
                generate_expert_trj_one_model = pd.concat([generate_expert_trj_one_model, generate_expert_trj_left],axis=0)  # 纵向拼接 存放一个模型内的左转车的生成和专家轨迹
                # print('左转车的generate_expert_trj_one_model：',np.shape(generate_expert_trj_one_model), generate_expert_trj_one_model)
        # 画出直行车的轨迹，并存储
        for k in range(num_agents):

            if 3 <= k:
                for step in range(230):
                    acc_rate_gerente[1].append(((8.5 * (actions[k][step][0] + 1)) / 2) - 3.6)
                    yaw_rate_gerente[1].append(((2.4 * (actions[k][step][1] + 1)) / 2) - 1.2)

                    if actions[k][step][1]>1 or actions[k][step][1]<-1:
                        print('生成轨迹 不对的数据，转角超出阈值：', actions[k][step][1])
                    # acc_rate_gerente[1].append(actions[k][step][0] * 4.9)
                    # yaw_rate_gerente[1].append(actions[k][step][1] * 3.7)
                    # print('agent11的action到底是不是0？',step, actions[11][step][0][0],actions[11][step][0][1])
                for step in range(185):
                    acc_rate_expert[1].append(((8.5 * (actionss[i][k][step][0] + 1)) / 2) - 3.6)
                    yaw_rate_expert[1].append(((2.4 * (actionss[i][k][step][1] + 1)) / 2) - 1.2)

                    # acc_rate_expert[1].append(actionss[i][k][step][0] * 4.9)
                    # yaw_rate_expert[1].append(actionss[i][k][step][1] * 3.7)

            if k not in sv and k >= 3:
                fig = plt.figure(figsize=(6, 3 / 4 * 6))

                x_g = vehicles[k][:, 0] * 38 - 4
                y_g = vehicles[k][:, 1] * 23 + 14

                non_zero_points_g = (x_g != -4) & (y_g != 14)

                x_e = expert[k][:, 0] * 38 - 4
                y_e = expert[k][:, 1] * 23 + 14

                non_zero_points_e = (x_e != -4) & (y_e != 14)

                plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g],
                            s=10, zorder=100, c='none', marker='o', edgecolors='b')  # 车辆的x和y # 生成轨迹,蓝色
                plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14,
                            s=20, zorder=100, c='none', marker='o', edgecolors='k')  # 车辆初始时刻的位置 黑色
                plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, zorder=99, c='none', marker='o',
                            edgecolors='g')  # 专家整条轨迹 绿色

                # plt.scatter(vehicles[k][:, 0] * 39 - 5, vehicles[k][:, 1] * 38 - 3, c='b',
                #             s=10)  # 车辆的x和y # 生成轨迹,蓝色
                # plt.scatter(vehicles[k][0][0] * 39 - 5, vehicles[k][0][1] * 38 - 3, c='k',
                #             s=20)  # 车辆初始时刻的位置 黑色
                # plt.scatter(expert[k][:, 0] * 39 - 5, expert[k][:, 1] * 38 - 3, c='g', s=30)  # 专家整条轨迹 绿色
                # plt.xlim(-5, 34)
                # plt.ylim(-3, 35)

                # 把交叉口的范围画出来
                config = args_parser()

                map_path = glob.glob(config['path'] + '/*.osm')[0]
                if use_lanelet2_lib:
                    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                    laneletmap = lanelet2.io.load(map_path, projector)
                    map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
                else:
                    # 这里你需要将之前的画交叉口边界的代码整合到这里
                    map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

                plt.savefig(root + '\轨迹对比图片' + '\%s.png' % (
                            'scenario_' + str(scenario_test) + 'model_' + str(ii) + 'straight_vehicle_' + str(k)),
                            dpi=300)
                plt.close()


                generate_trj_straight = pd.concat(
                    [pd.DataFrame(vehicles[k][:, 0] * 38 - 4), pd.DataFrame(vehicles[k][:, 1] * 23 + 14)],
                    axis=1)  # 横向拼接

                generate_action_straight = pd.concat(
                    [pd.DataFrame(acc_rate_gerente[1][0+(k-3)*230:230+(k-3)*230]), pd.DataFrame(yaw_rate_gerente[1][0+(k-3)*230:230+(k-3)*230])], axis=1)

                generate_angle_straight = pd.DataFrame(vehicles[k][:, 6] * 191 - 1)  # 横向拼接
                generate_v_straight = pd.DataFrame(np.sqrt((vehicles[k][:, 2].astype(float) * 21 - 14) ** 2 + (vehicles[k][:, 3].astype(float) * 12 - 2) ** 2))


                expert_trj_straight = pd.concat(
                    [pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                    axis=1)  # 横向拼接

                expert_action_straight = pd.concat(
                    [pd.DataFrame(acc_rate_expert[1][0+(k-3)*185:185+(k-3)*185]), pd.DataFrame(yaw_rate_expert[1][0+(k-3)*185:185+(k-3)*185])], axis=1)

                expert_angle_straight = pd.DataFrame(expert[k][:, 6] * 191 - 1)  # 横向拼接
                expert_v_straight = pd.DataFrame(np.sqrt((expert[k][:, 2] * 21 - 14) ** 2 + (expert[k][:, 3] * 12 - 2) ** 2))

                generate_expert_trj_straight = pd.concat([generate_trj_straight, expert_trj_straight],
                                                         axis=1)

                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, generate_action_straight],
                                                         axis=1)
                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, generate_angle_straight],
                                                         axis=1)
                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, expert_action_straight],
                                                         axis=1)
                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, expert_angle_straight],
                                                         axis=1)
                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, generate_v_straight],
                                                         axis=1)
                generate_expert_trj_straight = pd.concat([generate_expert_trj_straight, expert_v_straight],
                                                         axis=1)

                generate_expert_trj_straight['agent_id'] = k
                generate_expert_trj_straight['model_id'] = ii
                generate_expert_trj_straight['direction'] = 'straight'

                # print('generate_expert_trj_one_model', generate_expert_trj_one_model.shape,
                #       generate_expert_trj_one_model)
                # print('generate_expert_trj_straight', generate_expert_trj_straight.shape, generate_expert_trj_straight)
                generate_expert_trj_one_model = pd.concat([generate_expert_trj_one_model, generate_expert_trj_straight], axis=0)  # 纵向拼接 存放一个模型内的左转车的生成和专家轨迹

        # 把左转的专家轨迹和生成车辆放到一起输出图片
        fig_all = plt.figure(figsize=(6, 3 / 4 * 6))
        for k in range(num_agents):
            if k < 3:
                # fig_all = plt.figure(figsize=(6, 3 / 4 * 6))
                if ini_point[k][0] != 0:
                    x_g = vehicles[k][:, 0] * 38 - 4
                    y_g = vehicles[k][:, 1] * 23 + 14

                    non_zero_points_g = (x_g != -4) & (y_g != 14)

                    x_e = expert[k][:, 0] * 38 - 4
                    y_e = expert[k][:, 1] * 23 + 14

                    non_zero_points_e = (x_e != -4) & (y_e != 14)

                    plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g], s=10, zorder=100, c='none', marker='o',
                                edgecolors='b')  # 车辆的x和y 蓝色

                    plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                                marker='o', edgecolors='k')  # 车辆初始时刻的位置 黑色

                    plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, alpha=0.5, zorder=99, c='none',
                                marker='o', edgecolors='g')  # 绿色

                    # plt.scatter(vehicles[k][:, 0] * 39 - 5, vehicles[k][:, 1] * 38 - 3, c='b', s=10)  # 车辆的x和y 蓝色
                    #
                    # plt.scatter(vehicles[k][0][0] * 39 - 5, vehicles[k][0][1] * 38 - 3, c='k', s=20)  # 车辆初始时刻的位置 黑色
                    #
                    # plt.scatter(expert[k][:, 0] * 39 - 5, expert[k][:, 1] * 38 - 3, c='g', s=30, alpha=0.5)  # 绿色

        # plt.xlim(-5, 34)
        # plt.ylim(-3, 35)

        # 把交叉口的范围画出来
        config = args_parser()

        map_path = glob.glob(config['path'] + '/*.osm')[0]
        if use_lanelet2_lib:
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            laneletmap = lanelet2.io.load(map_path, projector)
            map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
        else:
            # 这里你需要将之前的画交叉口边界的代码整合到这里
            map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)
        plt.savefig(root + '\轨迹对比图片' + '\%s.png' % (
                    'scenario_' + str(scenario_test) + 'model_' + str(ii) + '_left_vehicle'), dpi=300)
        plt.close()
        # plt.savefig(root + '\轨迹对比图片' +  '\%s.png' % ('scenario_' + str(scenario_test) + 'model_' + str(ii) + '_tess_left_vehicle'))

        # 将直行车的轨迹画到一张图上
        fig_all = plt.figure(figsize=(6, 3 / 4 * 6))
        for k in range(num_agents):
            if k >= 3:
                if expert[k][0, 0] * 38 - 4 != -4:
                    x_g = vehicles[k][:, 0] * 38 - 4
                    y_g = vehicles[k][:, 1] * 23 + 14

                    non_zero_points_g = (x_g != -4) & (y_g != 14)

                    x_e = expert[k][:, 0] * 38 - 4
                    y_e = expert[k][:, 1] * 23 + 14

                    non_zero_points_e = (x_e != -4) & (y_e != 14)

                    plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g], s=10, zorder=100, c='none', marker='o',
                                edgecolors='b')  # 车辆的x和y 蓝色

                    plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                                marker='o', edgecolors='k')  # 车辆初始时刻的位置 黑色

                    plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, alpha=0.5, zorder=99, c='none',
                                marker='o', edgecolors='g')  # 专家整条轨迹 绿色

                    # plt.scatter(vehicles[k][:, 0] * 39 - 5, vehicles[k][:, 1] * 38 - 3, c='b', s=10)  # 车辆的x和y 蓝色
                    #
                    # plt.scatter(vehicles[k][0][0] * 39 - 5, vehicles[k][0][1] * 38 - 3, c='k', s=20)  # 车辆初始时刻的位置 黑色
                    #
                    # plt.scatter(expert[k][:, 0] * 39 - 5, expert[k][:, 1] * 38 - 3, c='g', s=30, alpha=0.5)  # 专家整条轨迹 绿色

        # plt.xlim(-5, 34)
        # plt.ylim(-3, 35)

        # 把交叉口的范围画出来
        config = args_parser()

        map_path = glob.glob(config['path'] + '/*.osm')[0]
        if use_lanelet2_lib:
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            laneletmap = lanelet2.io.load(map_path, projector)
            map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
        else:
            # 这里你需要将之前的画交叉口边界的代码整合到这里
            map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

        plt.savefig(root + '\轨迹对比图片' + '\%s.png' % (
                    'scenario_' + str(scenario_test) + 'model_' + str(ii) + '_straight_vehicle'), dpi=300)


        # 把这个模型的左转车和直行车的acc和yaw存储下来
        generate_acc_left_i = pd.DataFrame(acc_rate_gerente[0])
        generate_yaw_left_i = pd.DataFrame(yaw_rate_gerente[0])
        generate_acc_delta_angle = pd.concat([generate_acc_left_i, generate_yaw_left_i], axis=1)  # 一列一列的往后接 竖着排列
        # print('generate_acc_delta_angle左转：',generate_acc_delta_angle)

        generate_acc_straight_i = pd.DataFrame(acc_rate_gerente[1])
        generate_yaw_straight_i = pd.DataFrame(yaw_rate_gerente[1])
        generate_acc_delta_angle = pd.concat([generate_acc_delta_angle, generate_acc_straight_i],axis=1)  # 一列一列的往后接 竖着排列
        generate_acc_delta_angle = pd.concat([generate_acc_delta_angle, generate_yaw_straight_i],axis=1)  # 一列一列的往后接 竖着排列
        # print('generate_acc_delta_angle左转+直行：', generate_acc_delta_angle)
        # generate_acc_delta_angle['agent_id'] = i
        # print('generate_acc_delta_angle',np.shape(generate_acc_delta_angle))

        expert_acc_left_i = pd.DataFrame(acc_rate[0])
        expert_yaw_left_i = pd.DataFrame(yaw_rate[0])
        expert_acc_delta_angle = pd.concat([expert_acc_left_i, expert_yaw_left_i], axis=1)  # 一列一列的往后接 竖着排列
        # print('expert_acc_delta_angle左转：', expert_acc_delta_angle)
        expert_acc_straight_i = pd.DataFrame(acc_rate[1])
        expert_yaw_straight_i = pd.DataFrame(yaw_rate[1])
        expert_acc_delta_angle = pd.concat([expert_acc_delta_angle, expert_acc_straight_i], axis=1)  # 一列一列的往后接 竖着排列
        expert_acc_delta_angle = pd.concat([expert_acc_delta_angle, expert_yaw_straight_i], axis=1)  # 一列一列的往后接 竖着排列
        # print('expert_acc_delta_angle左转+直行：', expert_acc_delta_angle)


        generate_expert_acc_yaw_one_model = pd.concat([generate_acc_delta_angle, expert_acc_delta_angle],axis=1)  # 一行一行的往后接 横着排列
        generate_expert_acc_yaw_one_model['model_id'] = ii

        # 画出参数的分布
        fig2 = plt.figure(figsize=(6, 3 / 4 * 6))
        plt.hist(yaw_rate[0], alpha=0.5)
        plt.hist(yaw_rate_gerente[0], alpha=0.5)
        plt.savefig(root + '\单个场景action分布图' + '\%s.png' % (str(scenario_test) + '_左转车_yaw_rate_' + 'tess_model_' + str(ii)))
        # plt.xlim(-28, 27)
        # plt.ylim(-4, 22)

        fig3 = plt.figure(figsize=(6, 3 / 4 * 6))
        plt.hist(yaw_rate[1], alpha=0.5)
        plt.hist(yaw_rate_gerente[1], alpha=0.5)
        plt.savefig(root + '\单个场景action分布图' + '\%s.png' % (str(scenario_test) + '_直行车_yaw_rate_' + 'tess_model_' + str(ii)))

        fig4 = plt.figure(figsize=(6, 3 / 4 * 6))
        plt.hist(acc_rate[0], alpha=0.5)
        plt.hist(acc_rate_gerente[0], alpha=0.5)
        plt.savefig(root + '\单个场景action分布图' + '\%s.png' % (str(scenario_test) + '_左转车_acc_' + 'tess_model_' + str(ii)))

        fig5 = plt.figure(figsize=(6, 3 / 4 * 6))
        plt.hist(acc_rate[1], alpha=0.5)
        plt.hist(acc_rate_gerente[1], alpha=0.5)
        plt.savefig(root + '\单个场景action分布图' + '\%s.png' % (str(scenario_test) + '直行车_acc_' + 'tess_model_' + str(ii)))

    # print('generate_expert_trj_one_model:',generate_expert_trj_one_model)
    generate_expert_trj_all = pd.concat([generate_expert_trj_all, generate_expert_trj_one_model], axis=0)
    generate_expert_acc_yaw_all = pd.concat([generate_expert_acc_yaw_all, generate_expert_acc_yaw_one_model], axis=0)

# print('generate_expert_acc_yaw_all',generate_expert_acc_yaw_all)
generate_expert_trj_all.columns = ['generate_x', 'generate_y', 'expert_x', 'expert_y', 'generate_acc', 'generate_yaw','generate_angle', 'expert_acc', 'expert_yaw', 'expert_angle','generate_v','expert_v','agent_id', 'model_id', 'direction']
generate_expert_acc_yaw_all.columns = ['generate_acc_left', 'generate_yaw_left', 'generate_acc_straight','generate_yaw_straight', 'expert_acc_left', 'expert_yaw_left','expert_acc_straight', 'expert_yaw_straight', 'model_id']

generate_expert_trj_all['length'] = 4.6
generate_expert_trj_all['width'] = 1.8
generate_expert_acc_yaw_all['length'] = 4.6
generate_expert_acc_yaw_all['width'] = 1.8

generate_expert_trj_all['generate_angle_now'] = generate_expert_trj_all['generate_yaw'] + generate_expert_trj_all['generate_angle']
generate_expert_trj_all['expert_angle_now'] = generate_expert_trj_all['expert_yaw'] + generate_expert_trj_all['expert_angle']

generate_expert_trj_all.to_csv(f"{root + '/用于做分布图和计算KL散度的数据'}/{scenario_test}_生成轨迹和专家轨迹.csv")
generate_expert_acc_yaw_all.to_csv(f"{root + '/加速度角度变化'}/{scenario_test}_生成轨迹和专家轨迹的加速度和角度变化.csv")

# 按照model来计算位置的RMSE，还有加速度、角度变化率的RMSE，以及角度变化率和加速度的分布KL散度，是一个场景十个模型的计算方式

# 计算位置的RMSE
average_pos_rmse = [[], []]  # 用于存放所有model的位置rmse的均值，第一个元素存放左转，第二个元素存放直行

model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]  # 划分场景
for one_model in model_datass:
    left_trj_model = one_model[one_model['direction'] == 'left']  # 找到场景中的左转轨迹
    straight_trj_model = one_model[one_model['direction'] == 'straight']  # 找到场景中的直行轨迹
    a = 0  # 统计这个场景内左转车相比67缺少的agent的数量
    # 分别计算每一辆左转车的轨迹RMSE
    trj_pos_rmse_left = float(0)  # 这个model内所有左转车的位置rmse
    trj_pos_rmse_left_2 = float(0)  # 这个model内所有左转车的位置rmse 用mean_squared_error来计算的
    left_trj_all = [name[1] for name in left_trj_model.groupby(['agent_id'])]
    rmse_left_1 = []
    rmse_left_2 = []
    rmse_straight_1 = []
    rmse_straight_2 = []
    for left_trj in left_trj_all:
        left_trj.index = range(len(left_trj))
        if len(left_trj[(left_trj['generate_x'] != -4)&(left_trj['generate_y'] != 14)
                        &(left_trj['expert_x'] != -4)&(left_trj['expert_x'].notna())&(left_trj['expert_y'] != 14)]) == 0:
            a = a + 1
            continue
        else:
            evaluate_left_trj = left_trj[(left_trj['generate_x'] != -4) & (left_trj['expert_x'] != -4) & (left_trj['expert_x'].notna())]
            expert_left_df = evaluate_left_trj[['expert_x','expert_y']]
            generate_left_df = evaluate_left_trj[['generate_x', 'generate_y']]
            trj_pos_rmse_left = np.sum(np.sqrt((evaluate_left_trj['generate_x'] - evaluate_left_trj['expert_x']) ** 2 + (
                        evaluate_left_trj['generate_y'] - evaluate_left_trj['expert_y']) ** 2)) / (len(evaluate_left_trj))
            # trj_pos_rmse_left_2 = mean_squared_error(expert_left_df, generate_left_df, squared=False)
            # trj_pos_rmse_left_2 = trj_pos_rmse_left_2 + np.sqrt(np.sum(
            #     (evaluate_left_trj['generate_x'] - evaluate_left_trj['expert_x']) ** 2 + (
            #                 evaluate_left_trj['generate_y'] - evaluate_left_trj['expert_y']) ** 2) / (len(evaluate_left_trj)))

            # trj_pos_rmse_left_2 = trj_pos_rmse_left_2 + mean_squared_error(expert_left_df, generate_left_df, squared=True)
            rmse_left_1.append(trj_pos_rmse_left)
            # rmse_left_2.append(trj_pos_rmse_left_2)
            # trj_pos_rmse_left = trj_pos_rmse_left + np.sqrt(np.sum(
            #     (left_trj['generate_x'] - left_trj['expert_x']) ** 2 + (
            #             left_trj['generate_y'] - left_trj['expert_y']) ** 2) / 139)

    rmse_left_1_array = np.array(rmse_left_1)
    # rmse_left_2_array = np.array(rmse_left_2)
    average_pos_rmse[0].append(np.mean(rmse_left_1_array[rmse_left_1_array >= 0]))
    # average_pos_rmse[0].append([np.mean(rmse_left_1_array[rmse_left_1_array > 0]),trj_pos_rmse_left_2 / (num_left - a)])
    # average_pos_rmse[0].append(trj_pos_rmse_left / (num_left - a))

    # 分别计算每一辆直行车的轨迹RMSE
    b = 0  # 统计这个场景内直行车相比67缺少的agent的数量
    trj_pos_rmse_straight = float(0)  # 这个model内所有直行车的位置rmse
    trj_pos_rmse_straight_2 = float(0)  # 这个model内所有直行车的位置rmse 用mean_squared_error来计算的
    straight_trj_all = [name[1] for name in straight_trj_model.groupby(['agent_id'])]
    for straight_trj in straight_trj_all:
        straight_trj.index = range(len(straight_trj))
        if len(straight_trj[(straight_trj['generate_x'] != -4)&(straight_trj['generate_y'] != 14)
                        &(straight_trj['expert_x'] != -4)&(straight_trj['expert_x'].notna())&(straight_trj['expert_y'] != 14)]) == 0:
            b = b + 1
            continue
        else:
            evaluate_straight_trj = straight_trj[(straight_trj['generate_x'] != -4) & (straight_trj['expert_x'] != -4)&(straight_trj['expert_x'].notna())]
            expert_straight_df = evaluate_straight_trj[['expert_x', 'expert_y']]
            generate_straight_df = evaluate_straight_trj[['generate_x', 'generate_y']]


            trj_pos_rmse_straight = np.sum(np.sqrt((evaluate_straight_trj['generate_x'] - evaluate_straight_trj['expert_x']) ** 2 + (
                        evaluate_straight_trj['generate_y'] - evaluate_straight_trj['expert_y']) ** 2)) / (
                                                                        len(evaluate_straight_trj))
            # trj_pos_rmse_straight = mean_squared_error(expert_straight_df,generate_straight_df, squared=False)

            # trj_pos_rmse_straight_2 = trj_pos_rmse_straight_2 + np.sqrt(np.sum(
            #     (evaluate_straight_trj['generate_x'] - evaluate_straight_trj['expert_x']) ** 2 + (
            #             evaluate_straight_trj['generate_y'] - evaluate_straight_trj['expert_y']) ** 2) / (
            #                                                             len(evaluate_straight_trj)))
            # trj_pos_rmse_straight_2 = trj_pos_rmse_straight_2 + mean_squared_error(expert_straight_df, generate_straight_df, squared=True)

            rmse_straight_1.append(trj_pos_rmse_straight)
            # rmse_straight_2.append(trj_pos_rmse_straight_2)

    rmse_straight_1_array = np.array(rmse_straight_1)
    # rmse_straight_2_array = np.array(rmse_straight_2)
    average_pos_rmse[1].append(np.mean(rmse_straight_1_array[rmse_straight_1_array >= 0]))

    # average_pos_rmse[1].append([np.mean(rmse_straight_1_array[rmse_straight_1_array > 0]),trj_pos_rmse_straight_2 / (num_straight - b)])

    # average_pos_rmse[1].append(trj_pos_rmse_straight / (num_straight - b))

average_pos_rmse_df = pd.DataFrame(average_pos_rmse)
# average_pos_rmse_df.columns = ['model_11']
average_pos_rmse_df.columns = ['model_0', 'model_1' , 'model_2', 'model_3', 'model_4',
                              'model_5', 'model_6','model_7','model_8', 'model_9','model_10','model_11',
                               'model_12','model_13','model_14','model_15',
                              'model_16','model_17','model_18','model_19','model_20'] # ,'model_21','model_22','model_23','model_24','model_25','model_26']
    #                           'model_5','model_6','model_7',
    #                           'model_8', 'model_9','model_10','model_11','model_12','model_13','model_14','model_15',
    #                            'model_16','model_17','model_18','model_19','model_20']
    # ['model_0', 'model_1', 'model_2', 'model_3', 'model_4',
    #                           'model_5']  #, 'model_6','model_7',
                              #'model_8', 'model_9','model_10','model_11','model_12','model_13','model_14','model_15',
                               #'model_16','model_17','model_18','model_19','model_20'
# average_pos_rmse_df.columns = ['model_0']
    # ,'model_21','model_22','model_23'
    #                            ,'model_24','model_25','model_26','model_27','model_28','model_29','model_30','model_31'
    #                            ,'model_32','model_33','model_34','model_35','model_36','model_37','model_38','model_39'
    #                            ,'model_40','model_41','model_42','model_43','model_44','model_45','model_46','model_47','model_48','model_49','model_50']
average_pos_rmse_df.to_csv(f"{root + '/轨迹位置的rmse'}/{scenario_test}_生成轨迹和专家轨迹位置的rmse.csv")

# 计算加速度和角度变化的RMSE
generate_expert_acc_yaw_all = pd.read_csv(f"{root + '/加速度角度变化'}/{scenario_test}_生成轨迹和专家轨迹的加速度和角度变化.csv")
average_acc_yaw = [[], [], [], []]  # 用于存放所有model的位置rmse的均值，第一个元素存放左转acc，第二个元素存放左转yaw,第三个元素存放直行acc，第四个元素存放直行yaw,

model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]  # 划分场景
for one_model in model_datass:
    # 分别计算一个模型中所有左转车和所有直行车的动作RMSE
    one_model_eva_use_left = one_model[(one_model['expert_x']!=-4)&(one_model['expert_x'].notna())&(one_model['generate_x']!=-4)&(one_model['direction']=='left')]
    one_model_eva_use_straight = one_model[(one_model['expert_x']!=-4)&(one_model['expert_x'].notna())&(one_model['generate_x']!=-4)&(one_model['direction']=='straight')]
    trj_acc_rmse_left = np.sqrt(
        pd.DataFrame((one_model_eva_use_left['generate_acc'] - one_model_eva_use_left['expert_acc']) ** 2).mean()[0])
    trj_yaw_rmse_left = np.sqrt(
        pd.DataFrame((one_model_eva_use_left['generate_yaw'] - one_model_eva_use_left['expert_yaw']) ** 2).mean()[0])
    trj_acc_rmse_straight = np.sqrt(
        pd.DataFrame((one_model_eva_use_straight['generate_acc'] - one_model_eva_use_straight['expert_acc']) ** 2).mean()[0])
    trj_yaw_rmse_straight = np.sqrt(
        pd.DataFrame((one_model_eva_use_straight['generate_yaw'] - one_model_eva_use_straight['expert_yaw']) ** 2).mean()[0])

    average_acc_yaw[0].append(trj_acc_rmse_left)
    average_acc_yaw[1].append(trj_yaw_rmse_left)
    average_acc_yaw[2].append(trj_acc_rmse_straight)
    average_acc_yaw[3].append(trj_yaw_rmse_straight)

average_acc_yaw_df = pd.DataFrame(average_acc_yaw)
# 创建一个新的索引
new_index = ['acc_left_rmse', 'yaw_left_rmse', 'acc_straight_rmse', 'yaw_straight_rmse']

# 设置新的索引并为其命名
average_acc_yaw_df = average_acc_yaw_df.set_index(pd.Index(new_index))
# average_acc_yaw_df.columns = ['model_11']
average_acc_yaw_df.columns = ['model_0', 'model_1' , 'model_2', 'model_3', 'model_4',
                              'model_5', 'model_6','model_7','model_8', 'model_9','model_10',
                              'model_11','model_12','model_13','model_14','model_15',
                              'model_16','model_17','model_18','model_19','model_20'] #,'model_21','model_22','model_23','model_24','model_25','model_26']
                    # ]
    # ['model_0', 'model_1', 'model_2', 'model_3', 'model_4',
    #                           'model_5', 'model_6','model_7',
    #                           'model_8', 'model_9','model_10','model_11','model_12','model_13','model_14','model_15',
    #                            'model_16','model_17','model_18','model_19','model_20']
    # , 'model_1', 'model_2', 'model_3', 'model_4',
    #                           'model_5','model_6','model_7',
    #                           'model_8', 'model_9','model_10','model_11','model_12','model_13','model_14','model_15',
    #                            'model_16','model_17','model_18','model_19','model_20']
    # ['model_0', 'model_1', 'model_2', 'model_3', 'model_4',
    #                           'model_5', 'model_6','model_7',
    #                           'model_8', 'model_9','model_10','model_11','model_12','model_13','model_14','model_15',
    #                            'model_16','model_17','model_18','model_19','model_20']
# average_acc_yaw_df.columns = ['model_0']
    # ,'model_21','model_22','model_23'
    #                            ,'model_24','model_25','model_26','model_27','model_28','model_29','model_30','model_31'
    #                            ,'model_32','model_33','model_34','model_35','model_36','model_37','model_38','model_39'
    #                            ,'model_40','model_41','model_42','model_43','model_44','model_45','model_46','model_47','model_48','model_49','model_50']

average_acc_yaw_df.to_csv(f"{root + '/轨迹action的rmse'}/{scenario_test}_生成轨迹和专家轨迹action的rmse.csv")

# # 评估动态决策过程
# # 计算改编过的协作加速度
# # 𝑎𝑎𝑑𝑑 = 2(𝑑𝑑𝑙𝑙 − 𝑣𝑣𝑙𝑙𝑇𝑇𝑠𝑠 + 𝜃𝜃𝑑𝑑𝑚𝑚𝑚𝑚𝑚𝑚)/𝑇𝑇𝑐𝑐2
#
def calculate_rectangle_vertices(center_x, center_y, length, width, angle):
    # 计算矩形的四个顶点相对于中心点的坐标（逆时针方向）
    angle_rad = np.radians(angle)  # 将角度转换为弧度
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # 计算矩形的四个顶点相对于中心点的坐标
    x_offset = 0.5 * length
    y_offset = 0.5 * width
    vertices = [
        (center_x - x_offset * cos_angle + y_offset * sin_angle,
         center_y - x_offset * sin_angle - y_offset * cos_angle),
        (center_x + x_offset * cos_angle + y_offset * sin_angle,
         center_y + x_offset * sin_angle - y_offset * cos_angle),
        (center_x + x_offset * cos_angle - y_offset * sin_angle,
         center_y + x_offset * sin_angle + y_offset * cos_angle),
        (center_x - x_offset * cos_angle - y_offset * sin_angle,
         center_y - x_offset * sin_angle + y_offset * cos_angle)
    ]

    return vertices


def check_intersection(rect1_vertices, rect2_vertices):
    # 检查两个矩形是否相交
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # 线段 pqr 共线
        return 1 if val > 0 else 2  # 顺时针或逆时针方向

    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def do_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # 一般情况下
        if o1 != o2 and o3 != o4:
            return True

        # 特殊情况
        if (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, q2, q1)) or \
                (o3 == 0 and on_segment(p2, p1, q2)) or (o4 == 0 and on_segment(p2, q1, q2)):
            return True

        return False

    for i in range(4):
        for j in range(4):
            if do_intersect(rect1_vertices[i], rect1_vertices[(i + 1) % 4], rect2_vertices[j],
                            rect2_vertices[(j + 1) % 4]):
                return True

    return False


def Cal_GT_gt(Agent_x, Agent_y, Agent_v, Agent_angle_last, Agent_direction,
              Jiaohu_x, Jiaohu_y, Jiaohu_v, Jiaohu_angle_last):  # time_trj,neig_left均为1行的dataframe

    # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
    dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
    if dis_between_agent_jiaohu <= 15:
        # 计算和这个车辆的GT
        agent_v = Agent_v
        neig_v = Jiaohu_v

        veh_length = 5  # 4.6 调大一些，这样就可以更安全一些
        veh_width = 2  # 1.8

        # 两辆车的k，斜率
        a_agent = math.tan(np.radians(Agent_angle_last))  # 斜率a_agent
        a_neig = math.tan(np.radians(Jiaohu_angle_last))  # 斜率a_neig

        # 两辆车的b
        b_agent = (Agent_y) - a_agent * (Agent_x)
        b_neig = (Jiaohu_y) - a_neig * (Jiaohu_x)

        # 两车的交点
        # 计算两直线的交点
        GT_value = 1  # 1代表安全，-1代表不安全
        dis_min = 0  # 如果不会碰撞，就距统计最近距离（涉及到两车都把彼此作为冲突对象的话，距离就是交互路径上最短的距离；如果不是的话，就是无意义的数字0，只是为了形式上的统一），如果会碰撞，则为0，也只是为了形式上的统一
        # 先判断当下时刻是否碰撞
        # 绘制两个矩形
        # 计算矩形的四个顶点坐标
        vertices_agent_now = calculate_rectangle_vertices(Agent_x, Agent_y, veh_length, veh_width,
                                                          Agent_angle_last)
        vertices_neig_now = calculate_rectangle_vertices(Jiaohu_x, Jiaohu_y, veh_length, veh_width,
                                                         Jiaohu_angle_last)
        # 判断两个矩阵是否有交集
        intersect_now = check_intersection(vertices_agent_now, vertices_neig_now)
        if intersect_now == True:  # 1.1.1(数字对应笔记本上的写法)
            # 说明agent和交互对象在这一位置相撞了
            GT_value = 0  # 非常不安全
            # dis_min = 0
        else:  # 2.
            # 继续判断未来或者当下
            # 先判断是否为主车视野前方的车辆
            a = np.array([Jiaohu_x - Agent_x, Jiaohu_y - Agent_y])
            b = np.zeros(2)

            if 0 <= Agent_angle_last < 90:  # tan>0
                b = np.array([1, math.tan(math.radians(Agent_angle_last))])
            elif Agent_angle_last == 90:
                b = np.array([0, 2])
            elif 90 < Agent_angle_last <= 180:  # tan<0
                b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
            elif 180 < Agent_angle_last < 270:  # tan>0
                b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
            elif Agent_angle_last == 270:  # 负无穷
                b = np.array([0, -2])
            elif 270 < Agent_angle_last <= 360:  # tan<0
                b = np.array([1, math.tan(math.radians(Agent_angle_last))])
            elif -90 < Agent_angle_last < 0:  # tan<0
                b = np.array([1, math.tan(math.radians(Agent_angle_last))])
            elif Agent_angle_last == -90:
                b = np.array([0, -2])
            Lb = np.sqrt(b.dot(b))
            La = np.sqrt(a.dot(a))
            cos_angle = np.dot(a, b) / (La * Lb)
            angle_hudu = np.arccos(cos_angle)
            angle_jiaodu = angle_hudu * 360 / (2 * np.pi)
            if (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                # neig为前车
                # 无交点，GT无穷大
                if a_neig == a_agent:
                    # 无交点，如果当下时刻没有撞，就不会撞了
                    GT_value = None  # 安全
                    # dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
                # 有交点，需继续分情况
                else:
                    # 先计算交点
                    jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)  # 真实的坐标
                    jiaodiany = a_neig * jiaodianx + b_neig
                    # print('jiaodianx_expert:', jiaodianx_expert, 'agent_x_expert', agent_x_expert,
                    #       'jiaodiany_expert', jiaodiany_expert, 'neig_y_expert:', neig_y_expert)

                    # 用交点是否在双方车辆视野范围内来计算GT
                    agent_b = np.zeros(2)
                    if 0 <= Agent_angle_last < 90:  # tan>0
                        agent_b = np.array([1, math.tan(math.radians(Agent_angle_last))])
                    elif Agent_angle_last == 90:
                        agent_b = np.array([0, 2])
                    elif 90 < Agent_angle_last <= 180:  # tan<0
                        agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                    elif 180 < Agent_angle_last < 270:  # tan>0
                        agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                    elif Agent_angle_last == 270:  # 负无穷
                        agent_b = np.array([0, -2])
                    elif 270 < Agent_angle_last <= 360:  # tan<0
                        agent_b = np.array([1, math.tan(math.radians(Agent_angle_last))])
                    elif -90 < Agent_angle_last < 0:  # tan<0
                        agent_b = np.array([1, math.tan(math.radians(Agent_angle_last))])
                    elif Agent_angle_last == -90:
                        agent_b = np.array([0, -2])

                    agent_a = np.array([jiaodianx - Agent_x, jiaodiany - Agent_y])
                    dot_product_agent = np.dot(agent_a, agent_b)

                    neig_b = np.zeros(2)
                    if 0 <= Jiaohu_angle_last < 90:  # tan>0
                        neig_b = np.array([1, math.tan(math.radians(Jiaohu_angle_last))])
                    elif Jiaohu_angle_last == 90:
                        neig_b = np.array([0, 2])
                    elif 90 < Jiaohu_angle_last <= 180:  # tan<0
                        neig_b = np.array([-1, -1 * math.tan(math.radians(Jiaohu_angle_last))])
                    elif 180 < Jiaohu_angle_last < 270:  # tan>0
                        neig_b = np.array([-1, -1 * math.tan(math.radians(Jiaohu_angle_last))])
                    elif Jiaohu_angle_last == 270:  # 负无穷
                        neig_b = np.array([0, -2])
                    elif 270 < Jiaohu_angle_last <= 360:  # tan<0
                        neig_b = np.array([1, math.tan(math.radians(Jiaohu_angle_last))])
                    elif -90 < Jiaohu_angle_last < 0:  # tan<0
                        neig_b = np.array([1, math.tan(math.radians(Jiaohu_angle_last))])
                    elif Jiaohu_angle_last == -90:
                        neig_b = np.array([0, -2])

                    neig_a = np.array([jiaodianx - Jiaohu_x, jiaodiany - Jiaohu_y])
                    dot_product_neig = np.dot(neig_a, neig_b)

                    if dot_product_agent >= 0 and dot_product_neig >= 0:
                        agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                        neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)

                        agent_first_dis = agent_dis + 0.5 * veh_width + 0.5 * veh_length
                        neig_last_dis = neig_dis - 0.5 * veh_width - 0.5 * veh_length
                        agent_last_dis = agent_dis - 0.5 * veh_width - 0.5 * veh_length
                        neig_first_dis = neig_dis + 0.5 * veh_width + 0.5 * veh_length
                        dis_agent_neig = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)

                        if (agent_first_dis / agent_v) < (neig_last_dis / neig_v):  # agent先通过
                            GT_value = abs(neig_last_dis / neig_v - agent_first_dis / agent_v)
                        elif (neig_first_dis / neig_v) < (agent_last_dis / agent_v):  # neig先通过
                            GT_value = abs(agent_last_dis / agent_v - neig_first_dis / neig_v)
                        else:
                            GT_value = min(abs(agent_last_dis / agent_v - neig_first_dis / neig_v),
                                           abs(neig_last_dis / neig_v - agent_first_dis / agent_v))

                    elif dot_product_agent >= 0 and dot_product_neig < 0:
                        GT_value = None  # 安全

                    else:
                        # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                        # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                        GT_value = None  # 安全
            else:
                # neig不是前车 2.4.2
                GT_value = None  # 不看做交互，因为当前时刻已经没有碰撞了
    else:
        GT_value = None  # 不交互
    return GT_value
def Cal_GT(time_trj,neig_trj,type):  # time_trj,neig_left均为1行的dataframe
    # 计算和这个车辆的GT
    time_trj.index = range(len(time_trj))
    neig_trj.index = range(len(neig_trj))
    neig_x = neig_trj[type+'_x'][0]
    neig_y = neig_trj[type+'_y'][0]
    agent_x = time_trj[type + '_x'][0]
    agent_y = time_trj[type + '_y'][0]
    neig_heading_now = neig_trj[type + '_angle'][0]  # 角度
    agent_heading_now = time_trj[type + '_angle'][0]  # 角度
    neig_v = neig_trj[type + '_v'][0]
    agent_v = time_trj[type + '_v'][0]

    veh_length = time_trj['length'][0]
    veh_width = time_trj['width'][0]

    # 两辆车的k，斜率
    a_neig = math.tan(np.radians(neig_heading_now))  # 斜率a_neig
    a_agent = math.tan(np.radians(agent_heading_now))  # 斜率a_agent

    # 两辆车的b
    b_neig = (neig_y) - a_neig * (neig_x)
    b_agent = (agent_y) - a_agent * (agent_x)

    # 两车的交点
    # 计算两直线的交点
    GT_value = None
    if a_neig == a_agent: # 无交点，GT无穷大
        GT_value = None # 安全
    else:
        jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)  # 真实的坐标
        jiaodiany = a_neig * jiaodianx + b_neig
        # print('jiaodianx_expert:', jiaodianx_expert, 'agent_x_expert', agent_x_expert,
        #       'jiaodiany_expert', jiaodiany_expert, 'neig_y_expert:', neig_y_expert)

        # 用交点是否在双方车辆视野范围内来计算GT
        agent_b = np.zeros(2)
        if 0 <= agent_heading_now < 90:  # tan>0
            agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
        elif agent_heading_now == 90:
            agent_b = np.array([0, 2])
        elif 90 < agent_heading_now <= 180:  # tan<0
            agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
        elif 180 < agent_heading_now < 270:  # tan>0
            agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
        elif agent_heading_now == 270:  # 负无穷
            agent_b = np.array([0, -2])
        elif 270 < agent_heading_now <= 360:  # tan<0
            agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
        elif -90 < agent_heading_now < 0:  # tan<0
            agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
        elif agent_heading_now == -90:
            agent_b = np.array([0, -2])

        agent_a = np.array([jiaodianx - agent_x, jiaodiany - agent_y])
        dot_product_agent = np.dot(agent_a, agent_b)

        neig_b = np.zeros(2)
        if 0 <= neig_heading_now < 90:  # tan>0
            neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
        elif neig_heading_now == 90:
            neig_b = np.array([0, 2])
        elif 90 < neig_heading_now <= 180:  # tan<0
            neig_b = np.array([-1, -1 * math.tan(math.radians(neig_heading_now))])
        elif 180 < neig_heading_now < 270:  # tan>0
            neig_b = np.array([-1, -1 * math.tan(math.radians(neig_heading_now))])
        elif neig_heading_now == 270:  # 负无穷
            neig_b = np.array([0, -2])
        elif 270 < neig_heading_now <= 360:  # tan<0
            neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
        elif -90 < neig_heading_now < 0:  # tan<0
            neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
        elif neig_heading_now == -90:
            neig_b = np.array([0, -2])

        neig_a = np.array([jiaodianx - neig_x, jiaodiany - neig_y])
        dot_product_neig = np.dot(neig_a, neig_b)

        if dot_product_agent > 0 and dot_product_neig > 0:
            # 有冲突
            # 情况1 agent先通过冲突点
            agent_dis = np.sqrt((agent_x - jiaodianx) ** 2 + (agent_y - jiaodiany) ** 2)
            neig_dis = np.sqrt((neig_x - jiaodianx) ** 2 + (neig_y - jiaodiany) ** 2)

            agent_first_dis = agent_dis + 0.5*veh_width + 0.5*veh_length
            neig_last_dis = neig_dis - 0.5*veh_width - 0.5*veh_length
            agent_last_dis = agent_dis - 0.5 * veh_width - 0.5 * veh_length
            neig_first_dis = neig_dis + 0.5 * veh_width + 0.5 * veh_length

            if agent_first_dis/agent_v < neig_last_dis/neig_v:
                GT_value = abs(neig_last_dis/neig_v - agent_first_dis/agent_v)
            else:
                GT_value = abs(agent_last_dis/agent_v - neig_first_dis/neig_v)

    return GT_value
# 输入完整的两条轨迹，计算动态决策的相关参数
def Jiaohu_change(agent_trj, neig_trj, type, time, direction):
    # 找到这两辆车的轨迹交点
    agent_trj.index = range(len(agent_trj))
    neig_trj.index = range(len(neig_trj))

    # 将x不等于-5的点保留
    agent_trj_model = agent_trj[agent_trj[type + '_x'] != -4]
    neig_trj_model = neig_trj[neig_trj[type + '_x'] != -4]
    neig_index = neig_trj_model[neig_trj_model['time_ms'] == time].index[0]
    agent_index = agent_trj_model[agent_trj_model['time_ms'] == time].index[0]
    # print(time,'neig_index:',neig_index,'agent_index:',agent_index)
    # agent_trj_model.sort_values(by='time_ms', inplace=True)
    # agent_trj_model.index = range(len(agent_trj_model))

    # neig_trj_model.sort_values(by='time_ms', inplace=True)
    # neig_trj_model.index = range(len(neig_trj_model))
    # 计算两条轨迹上的所有点的欧氏距离

    distances = []  # 统计两条轨迹每个点之间的距离，i是agnet的轨迹点索引，j是neig的轨迹点索引
    distances.append([100, 100, 30])
    # min_distance = []
    min_distance = [[100, 100, 30]]
    for i in range(len(agent_trj_model)):
        agentx = agent_trj_model[type+'_x'][i]
        agenty = agent_trj_model[type+'_y'][i]
        for j in range(len(neig_trj_model)):
            neigx = neig_trj_model[type+'_x'][j]
            neigy = neig_trj_model[type+'_y'][j]
            dis = np.sqrt((agentx - neigx) ** 2 +
                          (agenty - neigy) ** 2)
            if dis < min_distance[0][2]:
                min_distance = [[i, j, dis]]
    # agent_trj_model_new_0 = agent_trj_model
    agent_trj_model_new = agent_trj_model
    neig_trj_model_new = neig_trj_model
    type_jiaodian = None
    # 找到最小距离对应的索引
    if min_distance[0][2] < 1.5:
        type_jiaodian = 'point'
        min_distance_index = [min_distance[0][0], min_distance[0][1]]

        # 获取最小距离对应的坐标
        intersection_x = agent_trj_model[type+'_x'].iloc[min_distance_index[0]]
        intersection_y = agent_trj_model[type+'_y'].iloc[min_distance_index[0]]

        # print("近似交点坐标：", intersection_x, intersection_y)
        # 找到agent到达交点的时刻
        agent_trj_time_at_intersection = agent_trj_model['time_ms'].iloc[min_distance_index[0]]

        # 找到neig到达交点的时刻
        neig_trj_time_at_intersection = neig_trj_model['time_ms'].iloc[min_distance_index[1]]
        # print("直行车到达交点的时刻：", neig_trj_time_at_intersection)

        # 统计两辆车到冲突点的时间
        agent_trj_model[direction+'_'+'jiaodian_time_ms'] = agent_trj_time_at_intersection
        agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = agent_trj_time_at_intersection - agent_trj_model['time_ms']
        neig_trj_model[direction+'_'+'jiaodian_time_ms'] = neig_trj_time_at_intersection
        neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']

        # 在agent这一行存储交互对象到达交点的时刻
        agent_trj_model[direction + '_' + 'jiaodian_time_ms_neig'] = neig_trj_time_at_intersection
        agent_trj_model[direction + '_' + 'delat_jiaodian_time_ms_neig'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']

        # 保留每辆车到达冲突点之前的时刻
        agent_trj_model_new = agent_trj_model[agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
        neig_trj_model_new = neig_trj_model[neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
        agent_trj_model_new.index = range(len(agent_trj_model_new))
        neig_trj_model_new.index = range(len(neig_trj_model_new))

        if time < agent_trj_time_at_intersection:
            # 如果当前时刻还没有到达冲突点，计算指标
            # 计算agent当前时刻到冲突点的距离
            agent_index_new = agent_trj_model_new[agent_trj_model_new['time_ms'] == time].index[0]
        else:
            agent_index_new = None
        if time < neig_trj_time_at_intersection:
            # 如果当前时刻还没有到达冲突点，计算指标
            # 计算neig当前时刻到冲突点的距离
            neig_index_new = neig_trj_model_new[neig_trj_model_new['time_ms'] == time].index[0]
        else:
            neig_index_new = None


        if time < agent_trj_time_at_intersection:
            # 如果当前时刻还没有到达冲突点，计算指标
            # 计算agent当前时刻到冲突点的距离
            # agent_index_new = agent_trj_model_new[agent_trj_model_new['time_ms'] == time].index[0]
            trj_length_agent = float(0)
            for i in range(len(agent_trj_model_new)-1):
                dis_agent = np.sqrt((agent_trj_model_new[type+'_x'][i] - agent_trj_model_new[type+'_x'][i + 1]) ** 2
                                    + (agent_trj_model_new[type+'_y'][i] - agent_trj_model_new[type+'_y'][i + 1]) ** 2)
                trj_length_agent = trj_length_agent + dis_agent
            # print(trj)

            if time == agent_trj_model_new['time_ms'].min():
                agent_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_agent
            else:
                go_dis_agent = 0
                for j in range(1, 1 + agent_index_new):
                    go_dis_agent = go_dis_agent + np.sqrt(
                        (agent_trj_model_new[type+'_x'][j] - agent_trj_model_new[type+'_x'][j - 1]) ** 2
                        + (agent_trj_model_new[type+'_y'][j] - agent_trj_model_new[type+'_y'][j - 1]) ** 2)
                agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] = trj_length_agent - go_dis_agent

        if time < neig_trj_time_at_intersection:
            # 如果当前时刻还没有到达冲突点，计算指标
            # 计算neig当前时刻到冲突点的距离
            # neig_index_new = neig_trj_model_new[neig_trj_model_new['time_ms'] == time].index[0]
            trj_length_neig = float(0)
            for i in range(len(neig_trj_model_new) - 1):
                dis_neig = np.sqrt(
                    (neig_trj_model_new[type + '_x'][i] - neig_trj_model_new[type + '_x'][i + 1]) ** 2
                    + (neig_trj_model_new[type + '_y'][i] - neig_trj_model_new[type + '_y'][i + 1]) ** 2)
                trj_length_neig = trj_length_neig + dis_neig
            # print(trj)
            if time == neig_trj_model_new['time_ms'].min():
                neig_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_neig
            else:
                go_dis_neig = 0
                # print('neig_index:',1 + neig_index_new, time,
                #       'neig_trj_time_at_intersection',neig_trj_time_at_intersection,
                #       'neig_trj_model_new:',len(neig_trj_model_new),neig_trj_model_new)
                for j in range(1, 1 + neig_index_new):
                    # print('neig_trj_model_newtype_xj:',neig_trj_model_new[type + '_x'][j],'neig_trj_model_newtype_xj_1:',neig_trj_model_new[type + '_x'][j - 1],
                    #       'neig_trj_model_newtype_yj:',neig_trj_model_new[type + '_y'][j],' neig_trj_model_newtype_yj1:', neig_trj_model_new[type + '_y'][j - 1])
                    go_dis_neig = go_dis_neig + np.sqrt(
                        (neig_trj_model_new[type + '_x'][j] - neig_trj_model_new[type + '_x'][j - 1]) ** 2
                        + (neig_trj_model_new[type + '_y'][j] - neig_trj_model_new[type + '_y'][j - 1]) ** 2)
                neig_trj_model_new[direction+'_'+'chongtu_dis'][neig_index_new] = trj_length_neig - go_dis_neig

        # 计算agent和neig分别的期望加速度
        if (time < agent_trj_time_at_intersection) and (time < neig_trj_time_at_intersection):

            if (agent_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][agent_index_new] > 0) \
                    and (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] > 0):
                ad_rush = 2 * (agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] - agent_trj_model_new[type + '_v'][
                    agent_index_new] *
                               neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001 + 2) / (
                                  (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001) ** 2)
                ad_yield = 2 * (agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] - agent_trj_model_new[type + '_v'][
                    agent_index_new] *
                                neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001 - 2) / (
                                   (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001) ** 2)
                agent_trj_model_new[direction+'_'+'ad_rush'][agent_index_new] = ad_rush
                agent_trj_model_new[direction+'_'+'ad_yield'][agent_index_new] = ad_yield
                agent_trj_model_new[direction+'_'+'ad_rush_delta'][agent_index_new] = agent_trj_model_new[type + '_acc'][
                                                                            agent_index_new] - ad_rush
                agent_trj_model_new[direction+'_'+'ad_yield_delta'][agent_index_new] = agent_trj_model_new[type + '_acc'][
                                                                             agent_index_new] - ad_yield
                agent_trj_model_new[direction+'_'+'dydt'][agent_index_new] = (agent_trj_model_new[type + '_acc'][
                                                                    agent_index_new] - ad_yield) / (ad_rush - ad_yield)
                agent_trj_model_new[direction + '_dongtai_id'] = neig_trj_model_new['agent_id'][0]
            else:
                agent_trj_model_new = agent_trj_model
                agent_trj_model_new[direction + '_dongtai_id'] = neig_trj_model_new['agent_id'][0]

    # 没有找到最小距离对应的索引，需要根据轨迹的最后一个点的方向确定两条轨迹的交点，进一步计算指标
    else:

        agent_heading_now = agent_trj_model[type+'_angle_now'][len(agent_trj_model) - 1]
        # print('neig_trj_model:',neig_trj_model)
        neig_heading_now = neig_trj_model[type + '_angle_now'][len(neig_trj_model) - 1]

        neig_x = neig_trj_model[type + '_x'][len(neig_trj_model) - 1]
        neig_y = neig_trj_model[type + '_y'][len(neig_trj_model) - 1]
        agent_x = agent_trj_model[type + '_x'][len(agent_trj_model) - 1]
        agent_y = agent_trj_model[type + '_y'][len(agent_trj_model) - 1]
        neig_v = neig_trj_model[type + '_v'][len(neig_trj_model) - 1]
        agent_v = agent_trj_model[type + '_v'][len(agent_trj_model) - 1]


        neig_x_time = neig_trj_model[type + '_x'][neig_index]
        neig_y_time = neig_trj_model[type + '_y'][neig_index]

        # print('agent_trj_model:',time, agent_trj_model)

        agent_x_time = agent_trj_model[type + '_x'][agent_index]
        agent_y_time = agent_trj_model[type + '_y'][agent_index]

        # 计算直线
        # 两辆车的k，斜率
        a_neig = math.tan(np.radians(neig_heading_now))  # 斜率a_neig
        a_agent = math.tan(np.radians(agent_heading_now))  # 斜率a_agent

        # 两辆车的b
        b_neig = (neig_y) - a_neig * (neig_x)
        b_agent = (agent_y) - a_agent * (agent_x)

        # 两车的交点
        # 计算两直线的交点
        if a_neig != a_agent:  # 有交点，继续。无交点，参数都为空
            jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)  # 真实的坐标
            jiaodiany = a_neig * jiaodianx + b_neig
            # 用交点是否在双方车辆视野范围内来计算动态决策参数
            agent_b = np.zeros(2)
            if 0 <= agent_heading_now < 90:  # tan>0
                agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
            elif agent_heading_now == 90:
                agent_b = np.array([0, 2])
            elif 90 < agent_heading_now <= 180:  # tan<0
                agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
            elif 180 < agent_heading_now < 270:  # tan>0
                agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
            elif agent_heading_now == 270:  # 负无穷
                agent_b = np.array([0, -2])
            elif 270 < agent_heading_now <= 360:  # tan<0
                agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
            elif -90 < agent_heading_now < 0:  # tan<0
                agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
            elif agent_heading_now == -90:
                agent_b = np.array([0, -2])

            agent_a = np.array([jiaodianx - agent_x_time, jiaodiany - agent_y_time])
            dot_product_agent = np.dot(agent_a, agent_b)

            neig_b = np.zeros(2)
            if 0 <= neig_heading_now < 90:  # tan>0
                neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
            elif neig_heading_now == 90:
                neig_b = np.array([0, 2])
            elif 90 < neig_heading_now <= 180:  # tan<0
                neig_b = np.array([-1, -1 * math.tan(math.radians(neig_heading_now))])
            elif 180 < neig_heading_now < 270:  # tan>0
                neig_b = np.array([-1, -1 * math.tan(math.radians(neig_heading_now))])
            elif neig_heading_now == 270:  # 负无穷
                neig_b = np.array([0, -2])
            elif 270 < neig_heading_now <= 360:  # tan<0
                neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
            elif -90 < neig_heading_now < 0:  # tan<0
                neig_b = np.array([1, math.tan(math.radians(neig_heading_now))])
            elif neig_heading_now == -90:
                neig_b = np.array([0, -2])

            neig_a = np.array([jiaodianx - neig_x_time, jiaodiany - neig_y_time])
            dot_product_neig = np.dot(neig_a, neig_b)

            if dot_product_agent > 0 and dot_product_neig > 0:  # 两辆车都没到轨迹交点
                type_jiaodian = 'yanshen'
                # 获取最小距离对应的坐标
                intersection_x = jiaodianx
                intersection_y = jiaodiany

                # print("近似交点坐标：", intersection_x, intersection_y)
                # 找到agent到达交点的时刻

                agent_out_time = np.sqrt((agent_x_time - jiaodianx)**2+(agent_y_time - jiaodiany)**2)/agent_v
                agent_trj_time_at_intersection = agent_trj_model['time_ms'].iloc[agent_index] + agent_out_time*1000

                # 找到neig到达交点的时刻
                neig_out_time = np.sqrt((neig_x_time - jiaodianx) ** 2 + (neig_y_time - jiaodiany) ** 2) / neig_v
                neig_trj_time_at_intersection = neig_trj_model['time_ms'].iloc[neig_index] + neig_out_time*1000
                # print("直行车到达交点的时刻：", neig_trj_time_at_intersection)

                # 统计两辆车到冲突点的时间
                agent_trj_model[direction+'_'+'jiaodian_time_ms'] = agent_trj_time_at_intersection
                agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = agent_trj_time_at_intersection - agent_trj_model['time_ms']
                neig_trj_model[direction+'_'+'jiaodian_time_ms'] = neig_trj_time_at_intersection
                neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']

                # 在agent这一行存储交互对象到达交点的时刻
                agent_trj_model[direction + '_' + 'jiaodian_time_ms_neig'] = neig_trj_time_at_intersection
                agent_trj_model[direction + '_' + 'delat_jiaodian_time_ms_neig'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']


                # 保留每辆车到达冲突点之前的时刻
                agent_trj_model_new = agent_trj_model[agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
                neig_trj_model_new = neig_trj_model[neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
                agent_trj_model_new.index = range(len(agent_trj_model_new))
                neig_trj_model_new.index = range(len(neig_trj_model_new))

                if time < agent_trj_time_at_intersection:
                    agent_index_new = agent_trj_model_new[agent_trj_model_new['time_ms'] == time].index[0]
                else:
                    agent_index_new = None
                if time < neig_trj_time_at_intersection:
                    neig_index_new = neig_trj_model_new[neig_trj_model_new['time_ms'] == time].index[0]
                else:
                    neig_index_new = None

                if time < agent_trj_time_at_intersection:
                    # 如果当前时刻还没有到达冲突点，计算指标
                    # 计算agent当前时刻到冲突点的距离
                    trj_length_agent = float(0)
                    for i in range(len(agent_trj_model_new) - 1):
                        dis_agent = np.sqrt(
                            (agent_trj_model_new[type + '_x'][i] - agent_trj_model_new[type + '_x'][i + 1]) ** 2
                            + (agent_trj_model_new[type + '_y'][i] - agent_trj_model_new[type + '_y'][i + 1]) ** 2)
                        trj_length_agent = trj_length_agent + dis_agent
                    # print(trj)

                    if time == agent_trj_model_new['time_ms'].min():
                        agent_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_agent
                    else:
                        go_dis_agent = 0
                        for j in range(1, 1 + agent_index_new):
                            go_dis_agent = go_dis_agent + np.sqrt(
                                (agent_trj_model_new[type + '_x'][j] - agent_trj_model_new[type + '_x'][j - 1]) ** 2
                                + (agent_trj_model_new[type + '_y'][j] - agent_trj_model_new[type + '_y'][j - 1]) ** 2)
                        agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] = trj_length_agent - go_dis_agent

                if time < neig_trj_time_at_intersection:
                    # 如果当前时刻还没有到达冲突点，计算指标
                    # 计算neig当前时刻到冲突点的距离
                    trj_length_neig = float(0)
                    for i in range(len(neig_trj_model_new) - 1):
                        dis_neig = np.sqrt(
                            (neig_trj_model_new[type + '_x'][i] - neig_trj_model_new[type + '_x'][i + 1]) ** 2
                            + (neig_trj_model_new[type + '_y'][i] - neig_trj_model_new[type + '_y'][i + 1]) ** 2)
                        trj_length_neig = trj_length_neig + dis_neig
                    # print(trj)
                    if time == neig_trj_model_new['time_ms'].min():
                        neig_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_neig
                    else:
                        go_dis_neig = 0
                        for j in range(1, 1 + neig_index_new):
                            go_dis_neig = go_dis_neig + np.sqrt(
                                (neig_trj_model_new[type + '_x'][j] - neig_trj_model_new[type + '_x'][j - 1]) ** 2
                                + (neig_trj_model_new[type + '_y'][j] - neig_trj_model_new[type + '_y'][j - 1]) ** 2)
                        neig_trj_model_new[direction+'_'+'chongtu_dis'][neig_index_new] = trj_length_neig - go_dis_neig

                # 计算agent和neig分别的期望加速度
                if (time < agent_trj_time_at_intersection) \
                        and (time < neig_trj_time_at_intersection):

                    ad_rush = 2 * (agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] - agent_trj_model_new[type + '_v'][
                        agent_index_new] *
                                   neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001 + 2) / (
                                      (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001) ** 2)
                    ad_yield = 2 * (agent_trj_model_new[direction+'_'+'chongtu_dis'][agent_index_new] - agent_trj_model_new[type + '_v'][
                        agent_index_new] *
                                    neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001 - 2) / (
                                       (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][neig_index_new] * 0.001) ** 2)
                    agent_trj_model_new[direction+'_'+'ad_rush'][agent_index_new] = ad_rush
                    agent_trj_model_new[direction+'_'+'ad_yield'][agent_index_new] = ad_yield
                    agent_trj_model_new[direction+'_'+'ad_rush_delta'][agent_index_new] = agent_trj_model_new[type + '_acc'][
                                                                                agent_index_new] - ad_rush
                    agent_trj_model_new[direction+'_'+'ad_yield_delta'][agent_index_new] = agent_trj_model_new[type + '_acc'][
                                                                                 agent_index_new] - ad_yield
                    agent_trj_model_new[direction+'_'+'dydt'][agent_index_new] = (agent_trj_model_new[type + '_acc'][
                                                                        agent_index_new] - ad_yield) / (ad_rush - ad_yield)
                    agent_trj_model_new[direction + '_dongtai_id'] = neig_trj_model_new['agent_id'][0]

            # if dot_product_agent < 0 and dot_product_neig > 0:  # agent过了交点，但是neig还没有
            #     # 获取最小距离对应的坐标
            #     time_agent_x = agent_trj_model[type + '_x'][int(time / 100)]
            #     time_agent_y = agent_trj_model[type + '_y'][int(time / 100)]
            #     intersection_x = jiaodianx
            #     intersection_y = jiaodiany
            #     not_go_chongtu_agent = False  # 判断这个agent有没有通过冲突点  True代表这个agent没有通过冲突点
            #     if agent_trj_model['direction'][0] == 'left':
            #         # agent是左转车
            #         if (time_agent_x < intersection_x) and (time_agent_y < intersection_y):
            #             # 这个agent没有通过冲突点
            #             not_go_chongtu_agent = True
            #     if agent_trj_model['direction'][0] == 'straight':
            #         # agent是直行车
            #         if time_agent_x > intersection_x:
            #             # 这个agent没有通过冲突点
            #             not_go_chongtu_agent = True
            #     if not_go_chongtu_agent == True:
            #         # print("近似交点坐标：", intersection_x, intersection_y)
            #         # 找到agent到达交点的时刻
            #         agent_out_time = np.sqrt((agent_x - jiaodianx) ** 2 + (agent_y - jiaodiany) ** 2) / agent_v
            #         agent_trj_time_at_intersection = agent_trj_model['time_ms'].iloc[
            #                                              len(agent_trj_model) - 1] - agent_out_time * 1000
            #
            #         # 找到neig到达交点的时刻
            #         neig_out_time = np.sqrt((neig_x - jiaodianx) ** 2 + (neig_y - jiaodiany) ** 2) / neig_v
            #         neig_trj_time_at_intersection = neig_trj_model['time_ms'].iloc[
            #                                             len(neig_trj_model) - 1] + neig_out_time * 1000
            #         # print("直行车到达交点的时刻：", neig_trj_time_at_intersection)
            #
            #         # 统计两辆车到冲突点的时间
            #         agent_trj_model[direction+'_'+'jiaodian_time_ms'] = agent_trj_time_at_intersection
            #         agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = agent_trj_time_at_intersection - agent_trj_model['time_ms']
            #         neig_trj_model[direction+'_'+'jiaodian_time_ms'] = neig_trj_time_at_intersection
            #         neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']
            #
            #         # 保留每辆车到达冲突点之前的时刻
            #         agent_trj_model_new = agent_trj_model[agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
            #         neig_trj_model_new = neig_trj_model[neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
            #         agent_trj_model_new.index = range(len(agent_trj_model_new))
            #         neig_trj_model_new.index = range(len(neig_trj_model_new))
            #
            #         if time <= agent_trj_time_at_intersection:
            #             # 如果当前时刻还没有到达冲突点，计算指标
            #             # 计算agent当前时刻到冲突点的距离
            #             trj_length_agent = float(0)
            #             for i in range(len(agent_trj_model_new) - 1):
            #                 dis_agent = np.sqrt(
            #                     (agent_trj_model_new[type + '_x'][i] - agent_trj_model_new[type + '_x'][i + 1]) ** 2
            #                     + (agent_trj_model_new[type + '_y'][i] - agent_trj_model_new[type + '_y'][i + 1]) ** 2)
            #                 trj_length_agent = trj_length_agent + dis_agent
            #             # print(trj)
            #
            #             if time == 0:
            #                 agent_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_agent
            #             else:
            #                 go_dis_agent = 0
            #                 for j in range(1, 1 + int(time / 100)):
            #                     go_dis_agent = go_dis_agent + np.sqrt(
            #                         (agent_trj_model_new[type + '_x'][j] - agent_trj_model_new[type + '_x'][j - 1]) ** 2
            #                         + (agent_trj_model_new[type + '_y'][j] - agent_trj_model_new[type + '_y'][j - 1]) ** 2)
            #                 agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] = trj_length_agent - go_dis_agent
            #
            #         if time <= neig_trj_time_at_intersection:
            #             # 如果当前时刻还没有到达冲突点，计算指标
            #             # 计算neig当前时刻到冲突点的距离
            #             trj_length_neig = float(0)
            #             for i in range(len(neig_trj_model_new) - 1):
            #                 dis_neig = np.sqrt(
            #                     (neig_trj_model_new[type + '_x'][i] - neig_trj_model_new[type + '_x'][i + 1]) ** 2
            #                     + (neig_trj_model_new[type + '_y'][i] - neig_trj_model_new[type + '_y'][i + 1]) ** 2)
            #                 trj_length_neig = trj_length_neig + dis_neig
            #             # print(trj)
            #             if time == 0:
            #                 neig_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_neig
            #             else:
            #                 go_dis_neig = 0
            #                 for j in range(1, 1 + int(time / 100)):
            #                     go_dis_neig = go_dis_neig + np.sqrt(
            #                         (neig_trj_model_new[type + '_x'][j] - neig_trj_model_new[type + '_x'][j - 1]) ** 2
            #                         + (neig_trj_model_new[type + '_y'][j] - neig_trj_model_new[type + '_y'][j - 1]) ** 2)
            #                 neig_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] = trj_length_neig - go_dis_neig
            #
            #         # 计算agent和neig分别的期望加速度
            #         if (agent_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] > 0) and (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] > 0):
            #             ad_rush = 2 * (
            #                         agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] - agent_trj_model_new[type + '_v'][
            #                     int(time / 100)] *
            #                         neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001 + 2) / (
            #                               (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001) ** 2)
            #             ad_yield = 2 * (
            #                         agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] - agent_trj_model_new[type + '_v'][
            #                     int(time / 100)] *
            #                         neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001 - 2) / (
            #                                (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001) ** 2)
            #             agent_trj_model_new[direction+'_'+'ad_rush'][int(time / 100)] = ad_rush
            #             agent_trj_model_new[direction+'_'+'ad_yield'][int(time / 100)] = ad_yield
            #             agent_trj_model_new[direction+'_'+'ad_rush_delta'][int(time / 100)] = agent_trj_model_new[type + '_acc'][
            #                                                                         int(time / 100)] - ad_rush
            #             agent_trj_model_new[direction+'_'+'ad_yield_delta'][int(time / 100)] = agent_trj_model_new[type + '_acc'][
            #                                                                          int(time / 100)] - ad_yield
            #             agent_trj_model_new[direction+'_'+'dydt'][int(time / 100)] = (agent_trj_model_new[type + '_acc'][
            #                                                                 int(time / 100)] - ad_yield) / (ad_rush - ad_yield)
            #
            #
            # if dot_product_agent > 0 and dot_product_neig < 0:  # neig过了交点，但是agent还没有
            #     # 获取最小距离对应的坐标
            #     time_neig_x = neig_trj_model[type + '_x'][int(time / 100)]
            #     time_neig_y = neig_trj_model[type + '_y'][int(time / 100)]
            #     intersection_x = jiaodianx
            #     intersection_y = jiaodiany
            #     not_go_chongtu_neig = False  # 判断这个agent有没有通过冲突点  True代表这个agent没有通过冲突点
            #     if neig_trj_model['direction'][0] == 'left':
            #         # neig是左转车
            #         if (time_neig_x < intersection_x) and (time_neig_y < intersection_y):
            #             # 这个neig没有通过冲突点
            #             not_go_chongtu_neig = True
            #     if neig_trj_model['direction'][0] == 'straight':
            #         # neig是直行车
            #         if time_neig_x > intersection_x:
            #             # 这个neig没有通过冲突点
            #             not_go_chongtu_neig = True
            #     if not_go_chongtu_neig == True:
            #         print("近似交点坐标：", intersection_x, intersection_y,'当前时刻的位置', time_neig_x, time_neig_y)
            #         # 找到agent到达交点的时刻
            #         agent_out_time = np.sqrt((agent_x - jiaodianx) ** 2 + (agent_y - jiaodiany) ** 2) / agent_v
            #         agent_trj_time_at_intersection = agent_trj_model['time_ms'].iloc[
            #                                              len(agent_trj_model) - 1] + agent_out_time * 1000
            #         # 找到neig到达交点的时刻
            #         neig_out_time = np.sqrt((neig_x - jiaodianx) ** 2 + (neig_y - jiaodiany) ** 2) / neig_v
            #         neig_trj_time_at_intersection = neig_trj_model['time_ms'].iloc[
            #                                             len(neig_trj_model) - 1] - neig_out_time * 1000
            #         # print("直行车到达交点的时刻：", neig_trj_time_at_intersection)
            #
            #         # 统计两辆车到冲突点的时间
            #         agent_trj_model[direction+'_'+'jiaodian_time_ms'] = agent_trj_time_at_intersection
            #         agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = agent_trj_time_at_intersection - agent_trj_model[
            #             'time_ms']
            #         neig_trj_model[direction+'_'+'jiaodian_time_ms'] = neig_trj_time_at_intersection
            #         neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] = neig_trj_time_at_intersection - neig_trj_model['time_ms']
            #
            #         # 保留每辆车到达冲突点之前的时刻
            #         agent_trj_model_new = agent_trj_model[agent_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
            #         neig_trj_model_new = neig_trj_model[neig_trj_model[direction+'_'+'delat_jiaodian_time_ms'] >= 0]
            #         agent_trj_model_new.index = range(len(agent_trj_model_new))
            #         neig_trj_model_new.index = range(len(neig_trj_model_new))
            #
            #         if time <= agent_trj_time_at_intersection:
            #             # 如果当前时刻还没有到达冲突点，计算指标
            #             # 计算agent当前时刻到冲突点的距离
            #             trj_length_agent = float(0)
            #             for i in range(len(agent_trj_model_new) - 1):
            #                 dis_agent = np.sqrt(
            #                     (agent_trj_model_new[type + '_x'][i] - agent_trj_model_new[type + '_x'][i + 1]) ** 2
            #                     + (agent_trj_model_new[type + '_y'][i] - agent_trj_model_new[type + '_y'][i + 1]) ** 2)
            #                 trj_length_agent = trj_length_agent + dis_agent
            #             # print(trj)
            #
            #             if time == 0:
            #                 agent_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_agent
            #             else:
            #                 go_dis_agent = 0
            #                 for j in range(1, 1 + int(time / 100)):
            #                     go_dis_agent = go_dis_agent + np.sqrt(
            #                         (agent_trj_model_new[type + '_x'][j] - agent_trj_model_new[type + '_x'][j - 1]) ** 2
            #                         + (agent_trj_model_new[type + '_y'][j] - agent_trj_model_new[type + '_y'][
            #                             j - 1]) ** 2)
            #                 agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] = trj_length_agent - go_dis_agent
            #
            #         if time <= neig_trj_time_at_intersection:
            #             # 如果当前时刻还没有到达冲突点，计算指标
            #             # 计算neig当前时刻到冲突点的距离
            #             trj_length_neig = float(0)
            #             for i in range(len(neig_trj_model_new) - 1):
            #                 dis_neig = np.sqrt(
            #                     (neig_trj_model_new[type + '_x'][i] - neig_trj_model_new[type + '_x'][i + 1]) ** 2
            #                     + (neig_trj_model_new[type + '_y'][i] - neig_trj_model_new[type + '_y'][i + 1]) ** 2)
            #                 trj_length_neig = trj_length_neig + dis_neig
            #             # print(trj)
            #             if time == 0:
            #                 neig_trj_model_new[direction+'_'+'chongtu_dis'][0] = trj_length_neig
            #             else:
            #                 go_dis_neig = 0
            #                 for j in range(1, 1 + int(time / 100)):
            #                     go_dis_neig = go_dis_neig + np.sqrt(
            #                         (neig_trj_model_new[type + '_x'][j] - neig_trj_model_new[type + '_x'][j - 1]) ** 2
            #                         + (neig_trj_model_new[type + '_y'][j] - neig_trj_model_new[type + '_y'][
            #                             j - 1]) ** 2)
            #                 neig_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] = trj_length_neig - go_dis_neig
            #
            #         # 计算agent和neig分别的期望加速度
            #         print('agent_trj_model_new:',time, agent_trj_model_new)
            #         if (agent_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] > 0) and (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] > 0):
            #             ad_rush = 2 * (
            #                     agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] - agent_trj_model_new[type + '_v'][
            #                 int(time / 100)] *
            #                     neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001 + 2) / (
            #                               (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001) ** 2)
            #             ad_yield = 2 * (
            #                     agent_trj_model_new[direction+'_'+'chongtu_dis'][int(time / 100)] - agent_trj_model_new[type + '_v'][
            #                 int(time / 100)] *
            #                     neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][int(time / 100)] * 0.001 - 2) / (
            #                                (neig_trj_model_new[direction+'_'+'delat_jiaodian_time_ms'][
            #                                     int(time / 100)] * 0.001) ** 2)
            #             agent_trj_model_new[direction+'_'+'ad_rush'][int(time / 100)] = ad_rush
            #             agent_trj_model_new[direction+'_'+'ad_yield'][int(time / 100)] = ad_yield
            #             agent_trj_model_new[direction+'_'+'ad_rush_delta'][int(time / 100)] = agent_trj_model_new[type + '_acc'][
            #                                                                         int(time / 100)] - ad_rush
            #             agent_trj_model_new[direction+'_'+'ad_yield_delta'][int(time / 100)] = agent_trj_model_new[type + '_acc'][
            #                                                                          int(time / 100)] - ad_yield
            #             agent_trj_model_new[direction+'_'+'dydt'][int(time / 100)] = (agent_trj_model_new[type + '_acc'][
            #                                                                 int(time / 100)] - ad_yield) / (
            #                                                                    ad_rush - ad_yield)

    # print('agent_trj_model_new:', time, agent_trj_model_new)
    if type_jiaodian == 'point':
        if (time < agent_trj_model_new['time_ms'].max()) \
                and (time < neig_trj_model_new['time_ms'].max()):
            return True, agent_trj_model_new.loc[agent_index]
        else:
            return False, agent_trj_model_new.loc[0]
    elif type_jiaodian == 'yanshen':  # 第二类延申出去得到的交点
        if (time <= agent_trj_model_new['time_ms'].max()) \
                and (time <= neig_trj_model_new['time_ms'].max()):
            return True, agent_trj_model_new.loc[agent_index]
        else:
            return False, agent_trj_model_new.loc[0]
    else:  # 没有交点
        return False, agent_trj_model_new.loc[0]
# generate_expert_trj_all = pd.read_csv(r'E:\wsh-科研\nvn_xuguan_sind\sinD_nvn_xuguan\MA_Intersection_straight\results_evaluate\v0\用于做分布图和计算KL散度的数据\85_生成轨迹和专家轨迹.csv')

generate_expert_trj_all['GT_AVE'] = None

generate_expert_trj_all['left_interaction_agent_id'] = None
generate_expert_trj_all['left_interaction_agent_dis'] = None
generate_expert_trj_all['GT_left'] = None
generate_expert_trj_all['right_interaction_agent_id'] = None
generate_expert_trj_all['right_interaction_agent_dis'] = None
generate_expert_trj_all['GT_right'] = None

generate_expert_trj_all['left_dongtai_id'] = None
generate_expert_trj_all['left_jiaodian_time_ms'] = None
generate_expert_trj_all['left_delat_jiaodian_time_ms'] = None
generate_expert_trj_all['left_jiaodian_time_ms_neig'] = None
generate_expert_trj_all['left_delat_jiaodian_time_ms_neig'] = None
generate_expert_trj_all['left_chongtu_dis'] = None
generate_expert_trj_all['left_ad_rush'] = None
generate_expert_trj_all['left_ad_yield'] = None
generate_expert_trj_all['left_ad_rush_delta'] = None
generate_expert_trj_all['left_ad_yield_delta'] = None
generate_expert_trj_all['left_dydt'] = None


generate_expert_trj_all['right_dongtai_id'] = None
generate_expert_trj_all['right_jiaodian_time_ms'] = None
generate_expert_trj_all['right_delat_jiaodian_time_ms'] = None
generate_expert_trj_all['right_jiaodian_time_ms_neig'] = None
generate_expert_trj_all['right_delat_jiaodian_time_ms_neig'] = None
generate_expert_trj_all['right_chongtu_dis'] = None
generate_expert_trj_all['right_ad_rush'] = None
generate_expert_trj_all['right_ad_yield'] = None
generate_expert_trj_all['right_ad_rush_delta'] = None
generate_expert_trj_all['right_ad_yield_delta'] = None
generate_expert_trj_all['right_dydt'] = None


model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]  # 划分场景
data_all_model_generate = pd.DataFrame() # 存放计算完期望加速度的左转车和直行车

for one_model in model_datass:
    # print('one_model.columns',one_model.columns)
    # 遍历这里的每一辆agent
    trj_datass = [name[1] for name in one_model.groupby(['agent_id'])]
    for trj0 in trj_datass:
        # print('lentrj:',len(trj0))
        trj0['time_ms'] = range(0, len(trj0) * 100, 100)
        trj0.index = range(len(trj0))
        # print('lentrj078:',trj0['time_ms'][78])
        trj = trj0[trj0['generate_x']!=-4]
        # print('lentrj78:', trj['time_ms'][78])
        trj_max_time = trj['time_ms'].max()

        if len(trj)!=0:
            # print('目前正在处理的model和轨迹为：', trj['model_id'][trj['time_ms'].idxmin()],
            #       trj['agent_id'][trj['time_ms'].idxmin()])
            trj_id = trj['agent_id'][trj['time_ms'].idxmin()]
            if trj_id <= 2:
                agent_direction = 'left'
            else:
                agent_direction = 'straight'
            other_potential_trjs = one_model[one_model['agent_id']!=trj_id]
            other_trjs = [name[1] for name in other_potential_trjs.groupby(['agent_id'])]
            other_trjs_new_df = pd.DataFrame()  # 存放有时间数据的agent之外的其他轨迹数据
            for other_trj_id, other_trj in enumerate(other_trjs):
                # 计算距离
                other_trj.index = range(len(other_trj))
                other_trj['time_ms'] = range(0, len(other_trj) * 100, 100)
                other_trj_use = other_trj[other_trj['generate_x']!=-4]
                other_trj_use.sort_values(by='time_ms', inplace=True)
                # other_trj_use.index = range(len(other_trj_use))
                if len(other_trj_use)!=0:
                    other_trjs_new_df = pd.concat([other_trjs_new_df, other_trj_use], axis=0)
                other_trj_new = other_trj[other_trj['time_ms'] <= trj_max_time]
                # distance = np.sqrt((trj['generate_x'] - other_trj_new['generate_x']) ** 2 + (trj['generate_y'] - other_trj_new['generate_y']) ** 2)
                # 存储距离结果
                # trj[f'generate_distance_to_trj_{other_trj_id}'] = distance.values
            # 计算了每个时刻和左前方交互对象的GT和右前方交互对象的GT，还有动态决策相关的参数
            # print('trj:',len(trj),trj['time_ms'].idxmin(),trj['time_ms'].idxmax()+1,trj)

            # 计算每一时刻agent和每个其他agent的GT
            for i, row in trj.iterrows():
                # 找到在agent视野范围内的数据
                # trj = trj.copy()
                use_GT = []  # 存放这个时刻的主要交互对象的GT
                # agent_x = trj[trj.index == 76]['generate_x'][76]
                agent_x = trj.loc[i,'generate_x']
                # print('agent_x：',agent_x)
                agent_y = trj.loc[i,'generate_y']
                agent_v = trj.loc[i,'generate_v']
                angle = trj.loc[i,'generate_angle']  # 上一时刻的角度 -90~270
                agent_angle_last = angle
                time_agent = trj.loc[i,'time_ms']
                # print('trj分析:',trj[trj.index >= 78])
                # print('76时刻的time:',i,trj['time_ms'][76])

                potential_interaction_data = other_trjs_new_df[abs(other_trjs_new_df['time_ms'] - time_agent) < 100]
                potential_interaction_data.index = range(len(potential_interaction_data))

                if len(potential_interaction_data) > 0:
                    for jj in range(len(potential_interaction_data)):
                        jiaohu_agent_x = potential_interaction_data['generate_x'][jj]
                        jiaohu_agent_y = potential_interaction_data['generate_y'][jj]
                        jiaohu_agent_v = potential_interaction_data['generate_v'][jj]
                        jiaohu_agent_angle_last = potential_interaction_data['generate_angle'][jj]

                        jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_v,
                                                          agent_angle_last, agent_direction,
                                                          jiaohu_agent_x, jiaohu_agent_y,
                                                          jiaohu_agent_v,
                                                          jiaohu_agent_angle_last)

                        # if same_jiaohu_agent_GT_value is not None:
                        use_GT.append(jiaohu_agent_GT_value)

                    use_GT_list_0 = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                    use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                    rew_minGT_mapped = 0
                    print('use_GT_list:', use_GT_list)
                    if len(use_GT_list) != 0:
                        # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        rew_aveGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                        # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                    else:
                        rew_aveGT = None

                    trj['GT_AVE'][i] = rew_aveGT

                # 判断这辆交互车是否在agent的视野前方，且距离小于15m
                # print('potential_interaction_data:',len(potential_interaction_data))
                if len(potential_interaction_data) > 0:

                    interaction_data_left = pd.DataFrame()  # 存放真正的视野左侧的交互数据
                    interaction_data_right = pd.DataFrame()  # 存放真正的视野右侧的交互数据
                    for jj in range(len(potential_interaction_data)):
                        a = np.array([potential_interaction_data['generate_x'][jj] - agent_x,
                                      potential_interaction_data['generate_y'][jj] - agent_y])
                        b = np.zeros(2)
                        if 0 <= angle < 90:  # tan>0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif angle == 90:
                            b = np.array([0, 2])
                        elif 90 < angle <= 180:  # tan<0
                            b = np.array([-1, -1 * math.tan(math.radians(angle))])
                        elif 180 < angle < 270:  # tan>0
                            b = np.array([-1, -1 * math.tan(math.radians(angle))])
                        elif angle == 270:  # 负无穷
                            b = np.array([0, -2])
                        elif 270 < angle <= 360:  # tan<0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif -90 < angle < 0:  # tan<0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif angle == -90:
                            b = np.array([0, -2])

                        Lb = np.sqrt(b.dot(b))
                        La = np.sqrt(a.dot(a))

                        cos_angle = np.dot(a, b) / (La * Lb)
                        cross = np.cross((b[0], b[1]),
                                         (potential_interaction_data['generate_x'][jj] - agent_x,
                                          potential_interaction_data['generate_y'][jj] - agent_y))
                        angle_hudu = np.arccos(cos_angle)
                        angle_jiaodu = angle_hudu * 360 / 2 / np.pi

                        # dis = np.sqrt((potential_interaction_data['x'][jj] - agent_x) ** 2
                        #               + (potential_interaction_data['y'][jj] - agent_y) ** 2)

                        if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 在车辆左侧
                            one = potential_interaction_data[potential_interaction_data.index == jj]
                            interaction_data_left = pd.concat([interaction_data_left, one], axis=0)

                        if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 在车辆右侧
                            one = potential_interaction_data[potential_interaction_data.index == jj]
                            interaction_data_right = pd.concat([interaction_data_right, one], axis=0)


                    if len(interaction_data_left) > 0:
                        interaction_data_left['distance'] = None
                        interaction_data_left.index = range(len(interaction_data_left))
                        interaction_data_left['distance'] = np.sqrt(
                            (interaction_data_left['generate_y'] - agent_y) ** 2 + (
                                    interaction_data_left['generate_x'] - agent_x) ** 2)
                        interaction_data_left['distance'] = pd.to_numeric(interaction_data_left["distance"],
                                                                          errors='coerce')  # 将distance这一列的类型转化为float
                        if interaction_data_left['distance'].min() <= 15:  # 如果这和agent距离最近的车和agent的距离在15m之内
                            neig_left_Series = interaction_data_left.loc[
                                interaction_data_left['distance'].idxmin()]  # 找到四辆距离k车最近的
                            neig_left = pd.DataFrame(neig_left_Series).T
                            neig_left.index = range(len(neig_left))
                            trj['left_interaction_agent_id'][i] = neig_left['agent_id'][0]
                            trj['left_interaction_agent_dis'][i] = interaction_data_left['distance'].min()

                            # 计算trj和这个交互对象的GT
                            time_trj = trj[trj['time_ms']==time_agent].copy()
                            GT_value = Cal_GT(time_trj, neig_left, 'expert')
                            trj['GT_left'][i] = GT_value

                            # print('左侧交互对象计算完GT之后的time_ms78',i,trj['time_ms'][78])

                        # print('interaction_data_left:',interaction_data_left)
                        neig_left_Series_dongtai = interaction_data_left.loc[interaction_data_left['distance'].idxmin()]  # 找到四辆距离k车最近的
                        neig_left_dongtai = pd.DataFrame(neig_left_Series_dongtai).T
                        neig_left_dongtai.index = range(len(neig_left_dongtai))
                        # 计算trj和这个交互对象的动态交互参数
                        neig_left_id_dongtai = neig_left_dongtai['agent_id'][0]
                        neig_left_trj_dongtai = other_trjs_new_df[other_trjs_new_df['agent_id']==neig_left_id_dongtai]
                        # print('time_agent:',time_agent)
                        trj_jiaoju_left = trj.copy()
                        label, new_row_left_dongtai = Jiaohu_change(trj_jiaoju_left, neig_left_trj_dongtai,'generate',time_agent,'left')
                        new_row_left_dongtai_df = pd.DataFrame(new_row_left_dongtai).T
                        # print('new_row_right_dongtai_df:', new_row_right_dongtai_df)
                        new_row_left_dongtai_df.index = range(i, i + len(new_row_left_dongtai_df))
                        # 将新行替代原始行
                        if label == True:
                            trj.loc[i] = new_row_left_dongtai_df.loc[i]
                        # print('左侧交互对象计算完dongtaijiaohu之后的time_ms78', i, trj['time_ms'][78])


                        # trj['left_dongtai_id'][i] = neig_left_id_dongtai


                    if len(interaction_data_right) > 0:
                        interaction_data_right['distance'] = None
                        interaction_data_right.index = range(len(interaction_data_right))
                        interaction_data_right['distance'] = np.sqrt(
                            (interaction_data_right['generate_y'] - agent_y) ** 2 + (
                                    interaction_data_right['generate_x'] - agent_x) ** 2)

                        interaction_data_right['distance'] = pd.to_numeric(interaction_data_right["distance"],
                                                                           errors='coerce')  # 将distance这一列的类型转化为float

                        if interaction_data_right['distance'].min() <= 15:  # 如果这和agent距离最近的车和agent的距离在15m之内
                            neig_right_Series = interaction_data_right.loc[
                                interaction_data_right['distance'].idxmin()]  # 找到四辆距离k车最近的
                            neig_right = pd.DataFrame(neig_right_Series).T
                            neig_right.index = range(len(neig_right))
                            trj['right_interaction_agent_id'][i] = neig_right['agent_id'][0]
                            trj['right_interaction_agent_dis'][i] = interaction_data_right['distance'].min()
                            # 计算trj和这个交互对象的GT
                            time_trj = trj[trj['time_ms'] == time_agent].copy()
                            GT_value = Cal_GT(time_trj, neig_right, 'expert')
                            trj['GT_left'][i] = GT_value

                            # print('右侧交互对象计算完GT之前的time_ms78', i, trj['time_ms'][78])

                            # print('右侧交互对象计算完GT之后的time_ms78', i, trj['time_ms'][78])


                        # 计算trj和这个交互对象的动态交互参数
                        neig_right_Series_dongtai = interaction_data_right.loc[
                            interaction_data_right['distance'].idxmin()]  # 找到四辆距离k车最近的
                        neig_right_dongtai = pd.DataFrame(neig_right_Series_dongtai).T
                        neig_right_dongtai.index = range(len(neig_right_dongtai))
                        # 计算trj和这个交互对象的动态交互参数
                        neig_right_id_dongtai = neig_right_dongtai['agent_id'][0]
                        neig_right_trj_dongtai = other_trjs_new_df[other_trjs_new_df['agent_id'] == neig_right_id_dongtai]
                        # print('Jiaohu_change这里的time_agent:',i,time_agent)

                        trj_jiaohu_right = trj.copy()
                        # print('右侧交互对象计算完dongtaijiaohu之前的time_ms78', i, trj['time_ms'][78])
                        label, new_row_right_dongtai = Jiaohu_change(trj_jiaohu_right, neig_right_trj_dongtai, 'generate', time_agent, 'right')
                        new_row_right_dongtai_df = pd.DataFrame(new_row_right_dongtai).T
                        # print('new_row_right_dongtai_df:', new_row_right_dongtai_df)
                        new_row_right_dongtai_df.index = range(i,i+len(new_row_right_dongtai_df))
                        # print('new_row_right_dongtai_df_i:', i, new_row_right_dongtai_df['time_ms'][i])


                        if label == True:
                            # print('trj.iloc[i]', i, trj['time_ms'][i])
                            # print('new_row_right_dongtai_df.index==i', pd.DataFrame(new_row_right_dongtai_df[new_row_right_dongtai_df.index==i]))
                            trj.loc[i] = new_row_right_dongtai_df.loc[i]
                            # print('赋值完之后的trji:',trj.loc[i])
                        # if trj['time_ms'][78]!=7800:
                        #     print('时间不对的：',i,trj['time_ms'][78])
                        # print('右侧交互对象计算完dongtaijiaohu之后的time_ms78', i, trj['time_ms'][78])

            data_all_model_generate = pd.concat([data_all_model_generate,trj],axis=0)

data_all_model_generate['type'] = 'generate'

data_all_model_expert = pd.DataFrame() # 存放计算完期望加速度的左转车和直行车

for one_model in model_datass:
    # print('one_model.columns',one_model.columns)
    # 遍历这里的每一辆agent
    trj_datass = [name[1] for name in one_model.groupby(['agent_id'])]
    for trj0 in trj_datass:
        trj0.index = range(len(trj0))
        trj0['time_ms'] = range(0, len(trj0) * 100, 100)
        trj = trj0[(trj0['expert_x'] != -4) & (trj0['expert_x'].notna())]

        trj_max_time = trj['time_ms'].max()
        if len(trj) != 0:
            # print('trj:', len(trj), trj['time_ms'].idxmin(), trj['time_ms'].idxmax() + 1, trj)
            trj_id = trj['agent_id'][trj['time_ms'].idxmin()]
            if trj_id <= 2:
                agent_direction = 'left'
            else:
                agent_direction = 'straight'
            other_potential_trjs = one_model[one_model['agent_id']!=trj_id]
            other_trjs = [name[1] for name in other_potential_trjs.groupby(['agent_id'])]
            other_trjs_new_df = pd.DataFrame()  # 存放有时间数据的agent之外的其他轨迹数据
            for other_trj_id, other_trj in enumerate(other_trjs):
                # 计算距离
                other_trj.index = range(len(other_trj))
                other_trj['time_ms'] = range(0, len(other_trj) * 100, 100)
                other_trj_use = other_trj[other_trj['expert_x'] != -4]
                other_trj_use.sort_values(by='time_ms', inplace=True)
                # other_trj_use.index = range(len(other_trj_use))
                if len(other_trj_use) != 0:
                    other_trjs_new_df = pd.concat([other_trjs_new_df, other_trj_use], axis=0)
                other_trj_new = other_trj[other_trj['time_ms'] <= trj_max_time]
                # distance = np.sqrt((trj['expert_x'] - other_trj_new['expert_x']) ** 2 + (trj['expert_y'] - other_trj_new['expert_y']) ** 2)
                # 存储距离结果
                # trj[f'expert_distance_to_trj_{other_trj_id}'] = distance.values

            # 计算每一时刻agent和每个其他agent的GT
            for i in range(trj['time_ms'].idxmin(), trj['time_ms'].idxmax() + 1):
                # 找到在agent视野范围内的数据
                use_GT = []  # 存放这个时刻的主要交互对象的GT
                agent_x = trj.loc[i, 'expert_x']
                agent_y = trj.loc[i, 'expert_y']
                agent_v = trj.loc[i, 'expert_v']
                angle = trj.loc[i, 'expert_angle']  # 上一时刻的角度 -90~270
                agent_angle_last = angle
                time_agent = trj.loc[i, 'time_ms']

                # print('len(trj):', len(trj), 'time_agent:',time_agent)
                potential_interaction_data = other_trjs_new_df[abs(other_trjs_new_df['time_ms'] - time_agent) < 100]

                # 判断这辆交互车是否在agent的视野前方，且距离小于15m
                potential_interaction_data.index = range(len(potential_interaction_data))

                if len(potential_interaction_data) > 0:
                    for jj in range(len(potential_interaction_data)):
                        jiaohu_agent_x = potential_interaction_data['generate_x'][jj]
                        jiaohu_agent_y = potential_interaction_data['generate_y'][jj]
                        jiaohu_agent_v = potential_interaction_data['generate_v'][jj]
                        jiaohu_agent_angle_last = potential_interaction_data['generate_angle'][jj]

                        jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_v,
                                                          agent_angle_last, agent_direction,
                                                          jiaohu_agent_x, jiaohu_agent_y,
                                                          jiaohu_agent_v,
                                                          jiaohu_agent_angle_last)
                        # if same_jiaohu_agent_GT_value is not None:
                        use_GT.append(jiaohu_agent_GT_value)

                    use_GT_list_0 = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                    use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                    rew_minGT_mapped = 0
                    print('use_GT_list:', use_GT_list)
                    if len(use_GT_list) != 0:
                        # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        rew_aveGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                        # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                    else:
                        rew_aveGT = None

                    trj['GT_AVE'][i] = rew_aveGT

            # 计算了每个时刻和左前方交互对象的GT和右前方交互对象的GT，还有动态决策相关的参数
                # 判断这辆交互车是否在agent的视野前方，且距离小于15m
                if len(potential_interaction_data) > 0:
                    interaction_data_left = pd.DataFrame()  # 存放真正的视野左侧的交互数据
                    interaction_data_right = pd.DataFrame()  # 存放真正的视野右侧的交互数据
                    for jj in range(len(potential_interaction_data)):
                        a = np.array([potential_interaction_data['expert_x'][jj] - agent_x,
                                      potential_interaction_data['expert_y'][jj] - agent_y])
                        b = np.zeros(2)
                        if 0 <= angle < 90:  # tan>0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif angle == 90:
                            b = np.array([0, 2])
                        elif 90 < angle <= 180:  # tan<0
                            b = np.array([-1, -1 * math.tan(math.radians(angle))])
                        elif 180 < angle < 270:  # tan>0
                            b = np.array([-1, -1 * math.tan(math.radians(angle))])
                        elif angle == 270:  # 负无穷
                            b = np.array([0, -2])
                        elif 270 < angle <= 360:  # tan<0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif -90 < angle < 0:  # tan<0
                            b = np.array([1, math.tan(math.radians(angle))])
                        elif angle == -90:
                            b = np.array([0, -2])

                        Lb = np.sqrt(b.dot(b))
                        La = np.sqrt(a.dot(a))

                        cos_angle = np.dot(a, b) / (La * Lb)
                        cross = np.cross((b[0], b[1]),
                                         (potential_interaction_data['expert_x'][jj] - agent_x,
                                          potential_interaction_data['expert_y'][jj] - agent_y))
                        angle_hudu = np.arccos(cos_angle)
                        angle_jiaodu = angle_hudu * 360 / 2 / np.pi

                        # dis = np.sqrt((potential_interaction_data['x'][jj] - agent_x) ** 2
                        #               + (potential_interaction_data['y'][jj] - agent_y) ** 2)

                        if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 在车辆左侧
                            one = potential_interaction_data[potential_interaction_data.index == jj]
                            interaction_data_left = pd.concat([interaction_data_left, one], axis=0)

                        if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 在车辆右侧
                            one = potential_interaction_data[potential_interaction_data.index == jj]
                            interaction_data_right = pd.concat([interaction_data_right, one], axis=0)

                    if len(interaction_data_left) > 0:
                        interaction_data_left['distance'] = None
                        interaction_data_left.index = range(len(interaction_data_left))
                        interaction_data_left['distance'] = np.sqrt(
                            (interaction_data_left['expert_y'] - agent_y) ** 2 + (
                                    interaction_data_left['expert_x'] - agent_x) ** 2)
                        interaction_data_left['distance'] = pd.to_numeric(interaction_data_left["distance"],
                                                                          errors='coerce')  # 将distance这一列的类型转化为float
                        if interaction_data_left['distance'].min() <= 15:  # 如果这和agent距离最近的车和agent的距离在15m之内
                            neig_left_Series = interaction_data_left.loc[
                                interaction_data_left['distance'].idxmin()]  # 找到四辆距离k车最近的
                            neig_left = pd.DataFrame(neig_left_Series).T
                            neig_left.index = range(len(neig_left))
                            trj['left_interaction_agent_id'][i] = neig_left['agent_id'][0]
                            trj['left_interaction_agent_dis'][i] = interaction_data_left['distance'].min()
                            # 计算trj和这个交互对象的GT
                            time_trj = trj[trj['time_ms']==time_agent]
                            GT_value = Cal_GT(time_trj,neig_left,'expert')
                            trj['GT_left'][i] = GT_value

                        neig_left_Series_dongtai = interaction_data_left.loc[
                            interaction_data_left['distance'].idxmin()]  # 找到四辆距离k车最近的
                        neig_left_dongtai = pd.DataFrame(neig_left_Series_dongtai).T
                        neig_left_dongtai.index = range(len(neig_left_dongtai))
                        # 计算trj和这个交互对象的动态交互参数
                        neig_left_id_dongtai = neig_left_dongtai['agent_id'][0]
                        neig_left_trj_dongtai = other_trjs_new_df[other_trjs_new_df['agent_id'] == neig_left_id_dongtai]
                        # print('time_agent:',time_agent)
                        trj_jiaohu_left_expert = trj.copy()
                        # print('trj_jiaohu_left_expert:',trj_jiaohu_left_expert)
                        # print('neig_left_trj_dongtai:',neig_left_trj_dongtai)
                        label, new_row_left_dongtai = Jiaohu_change(trj_jiaohu_left_expert, neig_left_trj_dongtai, 'expert', time_agent, 'left')
                        new_row_left_dongtai_df = pd.DataFrame(new_row_left_dongtai).T
                        # print('new_row_right_dongtai_df:', new_row_right_dongtai_df)
                        new_row_left_dongtai_df.index = range(i, i + len(new_row_left_dongtai_df))
                        # 将新行替代原始行
                        if label == True:
                            trj.loc[i] = new_row_left_dongtai_df.loc[i]

                        # trj['left_dongtai_id'][i] = neig_left_id_dongtai


                    if len(interaction_data_right) > 0:
                        interaction_data_right['distance'] = None
                        interaction_data_right.index = range(len(interaction_data_right))
                        interaction_data_right['distance'] = np.sqrt(
                            (interaction_data_right['expert_y'] - agent_y) ** 2 + (
                                    interaction_data_right['expert_x'] - agent_x) ** 2)

                        interaction_data_right['distance'] = pd.to_numeric(interaction_data_right["distance"],
                                                                           errors='coerce')  # 将distance这一列的类型转化为float
                        if interaction_data_right['distance'].min() <= 15:  # 如果这和agent距离最近的车和agent的距离在15m之内
                            neig_right_Series = interaction_data_right.loc[
                                interaction_data_right['distance'].idxmin()]  # 找到四辆距离k车最近的
                            neig_right = pd.DataFrame(neig_right_Series).T
                            neig_right.index = range(len(neig_right))
                            trj['right_interaction_agent_id'][i] = neig_right['agent_id'][0]
                            trj['right_interaction_agent_dis'][i] = interaction_data_right['distance'].min()
                            # 计算trj和这个交互对象的GT
                            time_trj = trj[trj['time_ms'] == time_agent]
                            GT_value = Cal_GT(time_trj, neig_right, 'expert')
                            trj['GT_right'][i] = GT_value

                        # 计算trj和这个交互对象的动态交互参数
                        neig_right_Series_dongtai = interaction_data_right.loc[
                            interaction_data_right['distance'].idxmin()]  # 找到四辆距离k车最近的
                        neig_right_dongtai = pd.DataFrame(neig_right_Series_dongtai).T
                        neig_right_dongtai.index = range(len(neig_right_dongtai))
                        # 计算trj和这个交互对象的动态交互参数
                        neig_right_id_dongtai = neig_right_dongtai['agent_id'][0]
                        neig_right_trj_dongtai = other_trjs_new_df[
                            other_trjs_new_df['agent_id'] == neig_right_id_dongtai]
                        # print('time_agent:',time_agent)
                        trj_jiaohu_right_expert = trj.copy()
                        label, new_row_right_dongtai = Jiaohu_change(trj_jiaohu_right_expert, neig_right_trj_dongtai, 'expert', time_agent,
                                                              'right')
                        new_row_right_dongtai_df = pd.DataFrame(new_row_right_dongtai).T
                        # print('new_row_right_dongtai_df:', new_row_right_dongtai_df)
                        new_row_right_dongtai_df.index = range(i, i + len(new_row_right_dongtai_df))
                        # 将新行替代原始行
                        if label == True:
                            trj.loc[i] = new_row_right_dongtai_df.loc[i]
                        # 将新行替代原始行
                        # if label == True:
                        #     trj.iloc[i] = new_row_right_dongtai
                        # trj['right_dongtai_id'][i] = neig_right_id_dongtai

            data_all_model_expert = pd.concat([data_all_model_expert,trj],axis=0)

data_all_model_expert['type'] = 'expert'

data_all_model = pd.concat([data_all_model_generate, data_all_model_expert],axis=0)

data_all_model.to_csv(f"{root+ '/专家和生成轨迹desried_acc'}/{scenario_test}_生成和专家轨迹的期望加速度.csv")



# for k in sv:
#     plt.scatter(vehicles[k][:,0],vehicles[k][:,1], s = 15)
#     plt.scatter(vehicles[k][0,0],vehicles[k][0,1], c=5, s = 20)

# %
# k = 17

# fig = plt.figure(figsize=(6,3/4*6))
# ax = fig.gca()
# for k in range(num_agents):
#     plt.scatter(vehicles[k][:,0],vehicles[k][:,1], s = 15)

#     plt.scatter(vehicles[k][0,0],vehicles[k][0,1], c=5, s = 20)
# plt.xlim([980,1060])
# plt.ylim([970,1030])
# plt.plot(x,y)
# plt.xlim([0,1])
# plt.ylim([0,0.75])

# a = np.squeeze(sample_trajs[0]['ac'])  # 原本是sample_trajs
# b = np.squeeze(sample_trajs[0]['ob'])
# c = np.squeeze(sample_trajs[0]['rew'])

# %%
# %%


# n_obs = 24
# num_agents = 18
# results = []
# for sample_trajs in sample_trajss:
#     # sample_trajs = sample_trajss[8]
#     speed_gen_all = []
#     speed_gen_all_in = []
#     speed_gen_all_out = []
#     test_index = []
#     rmse_all = []
#     for sample_traj in sample_trajs:
#         index = sorted(list(range(0, n_obs*num_agents, n_obs))+list(range(1, n_obs*num_agents+1, n_obs)))
#         trajectories = sample_traj['all_ob'][:,index]
#         speeds_gen = np.array([sample_traj['ob'][j][:,:4] for j in range(num_agents)])
#         for speed_gen in speeds_gen:
#             speed_gen_all += [(speed_gen[i,2] **2 + speed_gen[i,3] **2) **0.5 *15 for i in range(179) if speed_gen[i,0]>0]
#             speed_gen_all_in += [(speed_gen[i,2] **2 + speed_gen[i,3] **2) **0.5 *15 for i in range(179) if in_insection(speed_gen[i,0], speed_gen[i,1])]
#             speed_gen_all_out += [(speed_gen[i,2] **2 + speed_gen[i,3] **2) **0.5 *15 for i in range(179) if (not in_insection(speed_gen[i,0], speed_gen[i,1]) and speed_gen[i,0]>0)]
#         iii = 1000
#         for ii in range(117):
#             trajs_expert = np.squeeze([experts[ii]['ob'][j][:,:2] for j in range(num_agents)])
#             if trajectories[0,0] != 0 and trajectories[0,0] in trajs_expert[:,0,0]:
#                 iii=ii
#                 test_index.append(iii)
#         # iii = 9#107
#         expert = np.squeeze([experts[iii]['ob'][j][:,:2] for j in range(num_agents)])
#         rmse = []
#
#         for i in range(num_agents):
#             traj_gen = trajectories[:,[0+i*2,1+i*2]]
#
#             traj_index = np.where(expert[:,0,0] == traj_gen[0,0])
#             if (traj_index[0].size)>0:
#                 traj_obs = expert[traj_index[0][0]]
#             else:
#                 traj_obs = np.zeros([179,2])
#
#
#             mask = np.logical_and(traj_obs[:,0] != 0 , traj_gen[:,0] != 0)
#             mask[0] = True
#
#             rmse.append (mean_squared_error(traj_obs[mask], traj_gen[mask], squared=False))
#
#         rmse = np.array(rmse)
#         rmse_all.append(np.mean(rmse[rmse>0]))
#
#     speed_gen_distr_in = np.histogram(speed_gen_all_in, bins=np.arange(20))
#     speed_gen_distr_in_test =  speed_gen_distr_in[0]/np.sum(speed_gen_distr_in[0])
#
#     speed_gen_distr = np.histogram(speed_gen_all, bins=np.arange(20))
#     speed_gen_distr_test =  speed_gen_distr[0]/np.sum(speed_gen_distr[0])
#
#     speed_gen_distr_out = np.histogram(speed_gen_all_out, bins=np.arange(20))
#     speed_gen_distr_out_test =  speed_gen_distr_out[0]/np.sum(speed_gen_distr_out[0])
#
#     dis = distance.jensenshannon(speed_distr_test, speed_gen_distr_test)
#     dis_in = distance.jensenshannon(speed_distr_in_test, speed_gen_distr_in_test)
#     dis_out = distance.jensenshannon(speed_distr_out_test, speed_gen_distr_out_test)
#
#     results.append([np.mean(rmse_all), dis, dis_in, dis_out])
#
#     # print('rmse:', np.mean(rmse_all))
#     # print('JS dis:', dis)
#     # print('JS dis_in:', dis_in)
#     # print('JS dis_out:', dis_out)
#
#     # plt.plot(speed_gen_distr_test)
#     # plt.plot(speed_gen_distr_in_test)
#     # plt.plot(speed_gen_distr_out_test)
#     # plt.legend(['overall','in','out'])
# aa = np.array(results)
# #%%
# speed_all = []
# speed_all_in = []
# speed_all_out = []
#
# for ii in test_index:
#     trajs_expert = np.squeeze([experts[ii]['ob'][j][:,:2] for j in range(num_agents)])
#     actions = np.squeeze(experts[ii]['ac'])
#     speeds = np.array([ (experts[ii]['ob'][j][:,:4]) for j in range(num_agents)])
#     for speed in speeds:
#         speed_all += [(speed[i,2] **2 + speed[i,3] **2) **0.5 *15 for i in range(179) if speed[i,0]>0]
#         speed_all_in += [(speed[i,2] **2 + speed[i,3] **2) **0.5 *15 for i in range(179) if in_insection(speed[i,0], speed[i,1])]
#         speed_all_out += [(speed[i,2] **2 + speed[i,3] **2) **0.5 *15 for i in range(179) if (not in_insection(speed[i,0], speed[i,1]) and speed[i,0]>0)]
#
# speed_distr = np.histogram(speed_all, bins=np.arange(20))
# speed_distr_test =  speed_distr[0]/np.sum(speed_distr[0])
#
# speed_distr_in = np.histogram(speed_all_in, bins=np.arange(20))
# speed_distr_in_test =  speed_distr_in[0]/np.sum(speed_distr_in[0])
#
# speed_distr_out = np.histogram(speed_all_out, bins=np.arange(20))
# speed_distr_out_test =  speed_distr_out[0]/np.sum(speed_distr_out[0])
#
# # plt.plot(speed_distr2)
# plt.plot(speed_distr_test)
# plt.plot(speed_distr_in_test)
# plt.plot(speed_distr_out_test)
# plt.legend(['overall','in','out'])
# # plt.hist(speed_distr[0])
# # plt.hist(speed_all)
# # plt.hist(speed_all_in)
# # plt.hist(speed_all_out)
#
#
#
# #%% plot_reward
#
# path = 'multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
#        '/s-100/l-0.0001-b-50-d-0.1-c-489-l2-0.1-iter-5-r-0.0/seed-13/'
#
#
# discriminator = get_dis(path, model, env_id)
#
#
# #%%
# import scipy
# for _ in range(5):
#     a =  [scipy.stats.norm.sf (np.random.normal(0,0.5)) for _ in range(360)]
#
#     b= 1
#     for i in range(360): b = b*a[i]
#
#     print(math.log(b,10))
#
# #%%
# expert = [experts[0]['ob'][j][:,:] for j in range(num_agents)]
# t=20  # 1 20 40 60; 100 130 140
# sv = 2  #11, 2
#
# fig = plt.figure(figsize=(4,3/4*4))
# # fig = plt.figure(figsize=(8,6))
# ax = fig.gca()
# for k in range(num_agents):
#
#     if k !=sv:
#     # plt.scatter(expert[k][t:t+5,0],expert[k][t:t+5,1])
#         plt.scatter(expert[k][t,0],expert[k][t,1], c=2, s = 8)
#         if expert[k][t,2] !=0:
#             plt.arrow (expert[k][t,0],expert[k][t,1], expert[k][t,2]/5,expert[k][t,3]/5,shape='full', color='grey', length_includes_head=False, zorder=0,head_width=0.015)
#         # print(expert[k][t,:4])
#     # plt.plot(expert[sv][t:t+5,0],expert[sv][t:t+5,1], c=5)
#     plt.scatter(expert[sv][t,0],expert[sv][t,1], c=5, s = 20)
#
# # plt.xlim([980,1060])
# # plt.ylim([970,1030])
# plt.plot(x,y)
# # plt.xlabel('(m)')
# # plt.ylabel('(m)')
#
# plt.xlim([0,1])
# plt.ylim([0,0.75])
#
# plt.show()
#
#
# mean_reward = []
# #%%
# prob = []
# accum_reward = []
# for i in range(20):
#     expert = [sample_trajss[0][i]['ob'][j][:,:] for j in range(num_agents)]
#
#     # plt.scatter(expert[sv][:,0],expert[sv][:,1], s = 15)
#     # plt.scatter(expert[sv][0,0],expert[sv][0,1],c=5, s = 20)
#
#     # plt.plot(x,y)
#     # plt.xlim([0,1])
#     # plt.ylim([0,0.75])
#
#     reward = []
#     # sv=11
#     for n_v in range(7):
#         reward.append([])
#         reward[-1] = np.zeros(180)
#         for t in range(179):
#             state = expert[sv][t]
#             # if state[0]!=0:
#             reward[-1][t] = discriminator[n_v].get_reward(state,
#                                 np.array([0,0]),
#                                 np.array([1,0]),
#                                 np.array([0,0]),
#                                 discrim_score=False)
#                 # else: reward[-1][t] = 0
#     reward = np.array(reward)
#
#     mean_reward.append(np.mean(reward, axis =0) )
#     accum_reward.append( [sum(mean_reward[-1][:i]) for i in range(1,179)])
#
#
#     a = len(np.where(expert[sv][:,0]!=0)[0]) *2
#     a=20 *2
#     aa =  [scipy.stats.norm.sf (np.random.normal(0,0.5)) for _ in range(a)]
#
#     b= 1
#     for i in range(a): b = b*aa[i]
#
#     prob.append(math.log(b,10))
#     # plt.show()
#
# plt.scatter(prob, [accum_reward[i][20] for i in range(20)])
#
# #%%
# for i in range(40,60):
#     plt.plot(accum_reward[i])
#     # plt.plot(mean_reward[i])
#     # plt.legend(range(0,6))
#
# # for n_v in range(18):
# #     did=str(n_v)+'_00001'
# #     path2 = path +'d_'+did
# #     discriminator[n_v].load(path2)
#
# #%%
# reward = []
# for n_v in range(7):
#
#     # state = np.zeros(22)
#     state = expert[sv][t]
#     # state[8] = state[2] - state[8]
#     # state[9] = state[3] - state[9]
#     reward.append([])
#     # reward[n_v] = [[] for _ in range(10)]
#
#     for acce in range(1):
#         reward[-1] = np.zeros([20,20])
#         for i in range(20):
#             for j in range(20):
#
#
#
#                 # state[0] = 0.7
#                 # state[1] = 0.2
#
#                 # state[2] =  0
#                 # state[3] = 0.3
#
#                 # state[4] = 0
#                 # state[5] = 0.3
#
#                 state[2] = i * 0.05 -0.5
#                 state[3] = -j *0.05 +0.5
#
#
#                 # for u in []
#
#                 # state[8] = state[2] - state[8]
#                 # state[9] = state[3] - state[9]
#                 # state[8] = i * 0.05 -0.5
#                 # state[9] = -j *0.05 +0.5
#
#
#                 reward[-1][i,j] = discriminator[n_v].get_reward(state,
#                                     np.array([0,0]),
#                                     np.array([1,0]),
#                                     np.array([0,0]),
#                                     discrim_score=False)
#                 # state[8] = state[2] - state[8]
#                 # state[9] = state[3] - state[9]
# reward = np.array(reward)
# mean_reward = np.mean(reward, axis =0)
# # mean_reward = np.mean(mean_reward, axis = 0)
#
#
#
# plt.imshow(mean_reward)
# # plt.imshow(mean_reward, vmin = -0.25, vmax = -0.23)
# plt.colorbar()
# plt.xticks([0,10,20],[-7.5,0,7.5])
# plt.yticks([0,10,20],[7.5,0,-7.5])
# plt.xlabel('v_x (m/s)')
# plt.ylabel('v_y (m/s)')
#
# #%%
#
# for n_v in range(18):
#     did=str(n_v)+'_00001'
#     path2 = path +'d_'+did
#     discriminator[n_v].load(path2)
#
#
# n1 = list(range(0,7))
# n2 = list(range(7,10))
# n3 = list(range(10,14))
# n4 = list(range(14,18))
#
# reward_expert = []
# for ii in range(100):
#     expert = [experts[ii]['ob'][j][:,:] for j in range(num_agents)]
#     reward_expert.append ([])
#     # for k in range(num_agents):
#
#     #     plt.scatter(expert[k][:,0],expert[k][:,1], s = 15)
#     #     plt.scatter(expert[k][0,0],expert[k][0,1], c=5, s = 20)
#
#     # plt.plot(x,y)
#     # plt.xlim([0,1])
#     # plt.ylim([0,0.75])
#
#     for n_v in range(18):
#        reward_expert[-1].append([])
#        reward = np.zeros([179,18])
#        for t in range(179):
#
#            state = expert[n_v][t,:]
#            for nn in [n1,n2,n3,n4]:
#
#                if n_v in nn and state[0] !=0:
#                     for k in nn:
#
#                         reward[t,k] = discriminator[k].get_reward(state,
#                                             np.array([0,0]),
#                                             np.array([1,0]),
#                                             np.array([0,0]),
#                                             discrim_score=False)
#                     break
#        reward = np.mean(reward[:,reward[0,:]!=0], axis = 1)
#        reward_expert[-1][-1] = reward
#
#
#            # elif n_v >8 and state[0] !=0:
#            #      for k in range(9,16):
#
#            #          reward[t,k-9] = discriminator[k].get_reward(state,
#            #                               np.array([0,0]),
#            #                               np.array([1,0]),
#            #                               np.array([0,0]),
#            #                               discrim_score=False)
#
#
# # reward_expert = np.load('reward_expert_4_004.npy', allow_pickle=True)
# # np.save( 'reward_expert_4_004',reward_expert )
# #%%
# index = 4
#
# aa = []
# bb = []
# cc = []
# dd = []
# for ii in  range(100):
#     expert = [experts[ii]['ob'][j][:,:] for j in range(num_agents)]
#     for k in range(num_agents):
#         for t in range(179):
#             if expert[k][t,0]!=0 and k in n1:
#                 # variable = (expert[k][t,index]**2+expert[k][t,index+1]**2)**0.5
#                 variable = expert[k][t,index]
#                 if variable != 0 and variable != 1:
#                     aa.append([variable, reward_expert[ii][k][t]])
#             elif expert[k][t,0]!=0 and k in n2:
#                 # variable = (expert[k][t,index]**2+expert[k][t,index+1]**2)**0.5
#
#                 variable = expert[k][t,index]
#                 if variable != 0 and variable != 1:
#                     bb.append([variable, reward_expert[ii][k][t]])
#             elif expert[k][t,0]!=0 and k in n3:
#                 # variable = (expert[k][t,index]**2+expert[k][t,index+1]**2)**0.5
#                 variable = expert[k][t,index]
#                 if variable != 0 and variable != 1:
#                     cc.append([variable, reward_expert[ii][k][t]])
#             elif expert[k][t,0]!=0 and k in n4:
#                 # variable = (expert[k][t,index]**2+expert[k][t,index+1]**2)**0.5
#                 variable = expert[k][t,index]
#                 if variable != 0 and variable != 1:
#                     dd.append([variable, reward_expert[ii][k][t]])
#
# # plt.xlim([0,1])
# aa = np.array(aa)
# bb = np.array(bb)
# cc = np.array(cc)
# dd = np.array(dd)
#
# plt.scatter(aa[:,0],aa[:,1], s = 5)
# plt.show()
# plt.scatter(bb[:,0],bb[:,1], s = 5)
# plt.show()
# plt.scatter(cc[:,0],cc[:,1], s = 5)
# plt.show()
# plt.scatter(dd[:,0],dd[:,1], s = 5)
# # plt.ylim([-0.12,-0.11])
# # plt.xlim([0,0.4])
#
#
# #%%
# n_v =2 #11
# expert = [experts[0]['ob'][j][:,:] for j in range(num_agents)]
# n1 = list(range(0,7))
# n2 = list(range(7,10))
# n3 = list(range(10,14))
# n4 = list(range(14,18))
#
#
# reward = np.zeros([179,18])
# for t in range(179):
#
#     state = expert[n_v][t,:]
#     # state[state == 1] = 0
#     for nn in [n1,n2,n3,n4]:
#
#         if n_v in nn and state[0] !=0:
#              for k in nn:
#
#                  reward[t,k] = discriminator[k].get_reward(state,
#                                      np.array([0,0]),
#                                      np.array([1,0]),
#                                      np.array([0,0]),
#                                      discrim_score=False)
#              break
# reward = np.mean(reward[:,reward[0,:]!=0], axis = 1)
# # reward_expert[-1][-1] = reward
# #%%
# index = 4
# mask = np.logical_and (expert[n_v][:,index]!=0, expert[n_v][:,index]!=1)
# plt.scatter(expert[n_v][:,index][mask],reward[mask], s = 5)
#
#
# #%%
# for n_v in range(18):
#     did=str(n_v)+'_00001'
#     path2 = path +'d_'+did
#     discriminator[n_v].load(path2)
#
#
# n1 = list(range(0,7))
# n2 = list(range(7,10))
# n3 = list(range(10,14))
# n4 = list(range(14,18))
# #%%
# from SALib.sample import saltelli,morris
# from SALib.analyze import sobol,morris
# from SALib.test_functions import Ishigami
#
# #[0.46,0.48]
# b1 = [[0,1] , [0.36,0.44], [0,1], [0,0.1], [0,0.8] , [0,0.1] , [0,5], [0,2], [0,0.1],[0,0.8],[0,0.1], [0,1],
#       [0,0.3],[0,0.4] , [0,0.6],[0,0.6], [0,0.4],[0,0.6], [0,0.5],[0,0.8], [0,0.1],[0,0.8],[0,0.1], [0,1]]
#
# b2 =  [[0,1] , [0.46,0.48], [0,1], [0,0.1], [0,0.8] , [0,0.1] ,[0,5],[0,2], [0,0.1],[0,0.8],[0,0.1], [0,1],
#       [0,0.3],[0,0.4] , [0,0.6],[0,0.6], [0,0.3],[0,0.6], [0,0.5],[0,0.6], [0,0.1],[0,0.8],[0,0.1], [0,1]]
#
# b3 = [[0.36, 0.44] , [0,0.75], [0,0.05], [0, 1], [0,0.03] , [0,0.7] ,[0,5],[0,0.2], [0,0.1],[0,0.6],[0,0.1], [0,1],
#       [0,0.6],[0,0.3] , [0,0.6],[0,0.5], [0,0.3],[0,0.35], [0,0.6],[0,0.6], [0,0.1],[0,0.6],[0,0.1], [0,1]]
#
# b4 =  [[0.5,0.66] , [0,0.75], [0,0.15], [0, 1], [0,0.1] , [0,0.6] ,[0,5],[0,0.2], [0,0.1],[0,0.6],[0,0.1], [0,0.8],
#       [0,0.6],[0,0.4] , [0,0.8],[0,0.7], [0,0.4],[0,0.4], [0,0.6],[0,0.5], [0,0.1],[0,0.7],[0,0.1], [0,1]]
#
# # Define the model inputs
# problem = {
#     'num_vars': 24,
#     'names': ['x'+str(i+1) for i in range(24)]
# ,
#     'bounds': b4
# }
#
# # Generate samples
# param_values = saltelli.sample(problem, 256)
#
# nn = n4
# Y = []
# for i in range(len(param_values)):
#     state = param_values[i,:]
#
#     reward = np.zeros([1,18])
#
#
#         # state[state == 1] = 0
#
#
#
#     for k in nn:
#
#         reward[0,k] = discriminator[k].get_reward(state,
#                             np.array([0,0]),
#                             np.array([1,0]),
#                             np.array([0,0]),
#                             discrim_score=False)
#
#     reward = np.mean(reward[:,reward[0,:]!=0], axis = 1)
#     Y.append(reward[0])
#     # Run model (example)
# # Y = Ishigami.evaluate(param_values)
# Y = np.array(Y)
# # Perform analysis
# Si = sobol.analyze(problem, Y, print_to_console=True)
#
# aa1 =(Si['ST'][np.argsort(-Si['ST'])[:8]])
# aa2 =(np.array(a) [(-Si['ST']).argsort()[:8]])
# #%%
# problem = {
#     'num_vars': 22,
#     'names': ['x'+str(i+1) for i in range(22)],
#     'bounds': b4
# }
#
# param_values = morris.sample(problem, 1000, num_levels=4)
# nn = n4
# Y = []
# for i in range(len(param_values)):
#     state = param_values[i,:]
#
#     reward = np.zeros([1,18])
#
#
#         # state[state == 1] = 0
#
#
#
#     for k in nn:
#
#         reward[0,k] = discriminator[k].get_reward(state,
#                             np.array([0,0]),
#                             np.array([1,0]),
#                             np.array([0,0]),
#                             discrim_score=False)
#
#     reward = np.mean(reward[:,reward[0,:]!=0], axis = 1)
#     Y.append(reward[0])
#     # Run model (example)
# # Y = Ishigami.evaluate(param_values)
# Y = np.array(Y)
# # Y = Ishigami.evaluate(X)
# Si = morris.analyze(problem, param_values, Y, conf_level=0.95,
#                     print_to_console=True, num_levels=4)
# # Print the first-order sensitivity indices
# # print(Si['S1'])
# # param_values = []
# # aa = []
# # for ii in  range(100):
# #     expert = [experts[ii]['ob'][j][:,:] for j in range(num_agents)]
# #     for k in range(num_agents):
# #         for t in range(179):
# #             if expert[k][t,0]!=0 and k in n1:
# #                 # variable = (expert[k][t,index]**2+expert[k][t,index+1]**2)**0.5
# #                 variable = expert[k][t,0]
# #                 param_values.append(expert[k][t,:])
# #                 if variable != 0 and variable != 1:
# #                     aa.append( reward_expert[ii][k][t])
#
# # param_values =np.array(param_values)
# # aa=np.array(aa)
# a = ['x position',
# 'y position',
# 'x speed',
# 'y speed',
# 'x distance to destination',
# 'y distance to destination',
# 'distance to boundary',
# 'distance to stop line',
# 'Leading/lateral distance ',
# 'Leading/longitudinal distance ',
# 'Leading/lateral speed difference ',
# 'Leading/longitudinal speed difference  ',
# 'Left/lateral distance ',
# 'Left/longitudinal distance' ,
# 'Left/lateral speed difference ',
# 'Left/longitudinal speed difference  ',
# 'Right/lateral distance ',
# 'Right/longitudinal distance ',
# 'Right/lateral speed difference ',
# 'Right/longitudinal speed difference ' ,
# 'Opposite/lateral distance ',
# 'Opposite/longitudinal distance ',
# 'Opposite/lateral speed difference ',
# 'Opposite/longitudinal speed difference'  ]
# Si['ST'][np.argsort(-Si['ST'])[:8]]
# np.array(a) [(-Si['ST']).argsort()[:8]]