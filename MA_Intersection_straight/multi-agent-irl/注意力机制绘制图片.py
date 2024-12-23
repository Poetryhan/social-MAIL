# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:29:31 2020

@author: uqjsun9
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from irl.render import makeModel, render, get_dis
from irl.mack.kfac_discriminator_airl import Discriminator
import pickle as pkl
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from shapely import geometry
from scipy.spatial import distance
from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from utils.DataReader import read_tracks_all, read_tracks_meta, read_light

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

    configs.add_argument('--path', default="D:/Study/同济大学/博三/面向自动驾驶测试的仿真/数据/SinD/github/SinD-main/SinD-main/Data/",
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
                                     (2.10, 27.47), (-4.7, 25.39),
                                     (-4.7, 6.41),
                                     (3.87, 4.70),
                                     (7.29, -2.64),
                                     (22.26, -2.57),
                                     (25.23, 3.74),
                                     (33.61, 6.41),
                                     (33.83, 25.76),
                                     (24.56, 28.06),
                                     (20.34, 34.73)])


def in_insection(x_true, y_true):
    # x = x * 39 - 5
    # y = y * 38 - 3
    point = geometry.Point(x_true, y_true)
    if poly.contains(point):
        return True
    else:
        return False


experts = pkl.load(open(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs\MA_Intersection_straight' \
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1.pkl','rb'))

num_agents = 8
num_left = 3
num_straight = 5

# % rendering
results = []
bleu_results = []
env_id = 'trj_intersection_4'
# model = makeModel(env_id)

# mids=['0100']
# %%
sample_trajss = []
mids = ['0310']
best_model = 11
print('mids:', len(mids))
# 上一行的print结果如下 mids: ['0001', '0100', '0200', '0300', '0400', '0489']

# mids=['0001']

mid = mids[0]
path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight' \
       r'\multi-agent-irl\irl\mack\multi-agent-trj\logger\airl\trj_intersection_4\decentralized' \
       r'\v13\l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0\seed-13\m_0' + mid   # 之前出的图用的v1


import scipy
import math

#%% 绘制生成场景的位置图
scenario_test = 79
root_attention = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight\results_evaluate' \
              fr'\v31\attention\{scenario_test}'
# 创建文件夹（如果不存在）
os.makedirs(root_attention, exist_ok=True)

print('experts[scenario_test_id][ac]', np.shape(experts[scenario_test]['ac'])) # (2, 139, 2)
expert = [experts[scenario_test]['ob'][j][:,:] for j in range(num_agents)] # 专家轨迹的第0个场景的所有agent的观测值obs
# expert_ac = [experts[scenario_test_id]['ac'][j] for j in range(num_agents)]

# 生成轨迹
generate_path_read = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
       fr'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\专门生成生成轨迹的数据' \
           fr'\{scenario_test}west_left\sample_trajss_{scenario_test}west_left.pkl'
f_read = open(generate_path_read,'rb')
generate_trajss = pkl.load(f_read)
generate_best_model = generate_trajss[best_model][0]
generate = [generate_best_model['ob'][j][:,:] for j in range(num_agents)]

path_landmark = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                        r'\ATT-social-iniobs_rlyouhua' \
                        r'\MA_Intersection_straight\AV_test\DATA\sinD_nvnxuguan_9jiaohu_landmarks_buxianshi.pkl'
f_landmark = open(path_landmark, 'rb')
all_landmark_trj = pkl.load(f_landmark)
landmark = [all_landmark_trj[scenario_test]['ob'][j][:,:] for j in range(30)]

t_attention = 20  # 1 20 40 60; 100 130 140  第20个时刻
sv = 0  #11, 2  主车的agent的编号，左转车

fig = plt.figure(figsize=(4,3/4*4))
# fig = plt.figure(figsize=(8,6))
ax = fig.gca()
for k in range(num_agents):
    if k !=sv:
    # plt.scatter(expert[k][t:t+5,0],expert[k][t:t+5,1])
    # 速度：左转车用蓝色箭头，直行车用红色箭头，landmark用黑色箭头
    # 位置：agent用黑色方块，landmark用空心圆
        if generate[k][t_attention,0] !=0:
            if k <= 2:
                speed_color = 'blue'
            else:
                speed_color = 'red'
            plt.scatter(generate[k][t_attention,0]*38-4,generate[k][t_attention,1]*23+14, c='black', s=8, zorder=100, marker='s') # 其他agent的点位置
            if generate[k][t_attention,2] !=0:  #  Vx[m/s] ！= 0
                '''
                下面这行代码画的是TRB论文上某个时间点其他agent的速度方向(带箭头)
                这段代码使用 matplotlib.pyplot.arrow 函数在图上绘制箭头。具体参数的含义如下：
                expert[k][t,0], expert[k][t,1]: 箭头的起始点的 x 和 y 坐标，即箭头的位置。
                expert[k][t,2]/5, expert[k][t,3]/5: 箭头的方向，由两个分量表示。expert[k][t,2] 是 x 方向的分量，expert[k][t,3] 是 y 方向的分量。箭头的长度和方向由这两个分量决定。在这里，将这两个分量除以 5，可能是为了缩小箭头的长度。
                shape='full': 箭头的形状，这里设置为 'full' 表示箭头是一个完整的三角形。
                color='grey': 箭头的颜色，这里设置为灰色。
                length_includes_head=False: 表示箭头的长度是否包括箭头头部。在这里设置为 False，表示不包括头部，箭头的长度只计算箭尾到箭头的尖端。
                zorder=0: 表示图中绘制的顺序，数值越大的图层会被绘制在数值较小的图层之上。
                head_width=0.015: 箭头头部的宽度。
                总体来说，这段代码是在图上以箭头的形式表示某一点到下一点的方向。箭头的起始位置和方向是通过 expert 中的数据确定的。
                '''
                # print('expert[k][t,2]/5,expert[k][t,3]/5:',(expert[k][t,2]*22-14)/100,(expert[k][t,3]*10-1)/100)
                plt.arrow(generate[k][t_attention, 0]*38-4, generate[k][t_attention, 1] * 23 + 14,
                          (generate[k][t_attention, 2] * 21 - 14),
                          (generate[k][t_attention, 3] * 12 - 2), shape='full', color=speed_color, length_includes_head=False,
                          zorder=100, head_width=1.5)
                print('场景：',scenario_test,'agent:',k,'第：',t_attention,'时刻','横向速度：',
                      generate[k][t_attention, 2] * 21 - 14,'纵向速度：',generate[k][t_attention, 3] * 12 - 2,
                      '合速度：',np.sqrt((generate[k][t_attention, 2] * 21 - 14)**2+(generate[k][t_attention, 3] * 12 - 1)**2))

    else:
        if generate[k][t_attention,0] !=0:
            # plt.scatter(generate[k][t_attention,0]*38-4,generate[k][t_attention,1]*23+14, c='black', s=10, zorder=100, marker='s') # 其他agent的点位置
            if generate[k][t_attention,2] !=0:  #  Vx[m/s] ！= 0
                if k <= 2:
                    speed_color = 'blue'
                else:
                    speed_color = 'red'
                plt.scatter(generate[sv][t_attention, 0]*38-4, generate[sv][t_attention, 1]*23+14, c='black', s=8, zorder=100)  # 主车agent的点
                plt.arrow(generate[sv][t_attention, 0]*38-4, generate[sv][t_attention, 1]*23+14, (generate[sv][t_attention, 2] * 21 - 14),
                          (generate[sv][t_attention, 3] * 12 - 2), shape='full', color=speed_color, length_includes_head=False, zorder=100,
                          head_width=1.5)
                print('场景：', scenario_test, 'sv_agent:', sv, '第：', t_attention, '时刻', '横向速度：',
                      generate[sv][t_attention, 2] * 21 - 14, '纵向速度：', generate[sv][t_attention, 3] * 12 - 1,
                      '合速度：', np.sqrt((generate[sv][t_attention, 2] * 21 - 14) ** 2 + (generate[sv][t_attention, 3] * 12 - 1) ** 2))


# landmark的个数
for ld in range(30):
    if landmark[ld][t_attention, 0] != 0:
        x_real = landmark[ld][t_attention, 0] * 39 - 5
        y_real = landmark[ld][t_attention, 1] * 38 - 3
        in_insection_label = in_insection(x_real, y_real)
        if in_insection_label == True:
            plt.scatter(landmark[ld][t_attention, 0] * 39 - 5, landmark[ld][t_attention, 1] * 38 - 3, edgecolors='black',
                facecolors='none', s=8, zorder=100, marker='s')  # 其他agent的点位置
            if landmark[ld][t_attention, 2] != 0:  # Vx[m/s] ！= 0
                '''
                总体来说，这段代码是在图上以箭头的形式表示某一点到下一点的方向。
                '''
                speed_color = 'black'
                # print('expert[k][t,2]/5,expert[k][t,3]/5:',(expert[k][t,2]*22-14)/100,(expert[k][t,3]*10-1)/100)
                plt.arrow(landmark[ld][t_attention, 0] * 39 - 5, landmark[ld][t_attention, 1] * 38 - 3,
                          (landmark[ld][t_attention, 2] * 31 - 16),
                          (landmark[ld][t_attention, 3] * 21 - 10), shape='full', color=speed_color, length_includes_head=False,
                          zorder=100, head_width=1.5)
                print('场景：', scenario_test, 'landmark:', ld, '第：', t_attention, '时刻', '横向速度：',
                      landmark[ld][t_attention, 2] * 31 - 16, '纵向速度：', landmark[ld][t_attention, 3] * 21 - 10,
                      '合速度：',
                      np.sqrt((landmark[ld][t_attention, 2] * 31 - 16) ** 2 + (landmark[ld][t_attention, 3] * 21 - 10) ** 2))


config = args_parser()

map_path = glob.glob(config['path']+'/*.osm')[0]
if use_lanelet2_lib:
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
    laneletmap = lanelet2.io.load(map_path, projector)
    map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
else:
    # 这里你需要将之前的画交叉口边界的代码整合到这里
    map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

plt.savefig(root_attention + '\轨迹点处速度\%s_generate.png' % (str(scenario_test) + '_' + 'agent_' + str(sv) + '_time_' + str(t_attention)), dpi=300)
plt.close()

# 绘制注意力

generate_ob = [generate_best_model['ob'][j][:,:] for j in range(num_agents)] # 专家轨迹的第0个场景的所有agent的观测值obs
generate_ac = [generate_best_model['ac'][j] for j in range(num_agents)]
print('generate_obs:', generate_ob, np.shape(generate_ob),'generate_acc:',np.shape(generate_ac))  # expert_obs: (2, 139, 14) expert_acc: (2, 139, 2)

# 分析车sv的注意力
state = generate_ob[sv][t_attention] # 主车在t时刻的obs
state_2 = state.reshape(-1, len(state))
next_state = generate_ob[sv][t_attention + 1]
next_state_2 = next_state.reshape(-1, len(next_state))



# 绘制主agent SV 的注意力

attention_spatial = [generate_best_model['all_attention_weight_spatial'][j][:,:,:,:] for j in range(num_agents)]
attention_temporal = [generate_best_model['all_attention_weight_temporal'][j][:,:,:] for j in range(num_agents)]

print('attention_spatial:',np.shape(attention_spatial))  # (8, 230, 21, 10, 10)
print('attention_temporal:',np.shape(attention_temporal))  # (8, 230, 21, 21)

# 读取主要讨论的agent的空间注意力

attention_spatial_sv_time = attention_spatial[sv][t_attention]  # 这个shape应该是（21，10，10）
attention_space = []
attention_space_retained_indices = []
print('attention_spatial_sv_time:',np.shape(attention_spatial_sv_time))  # (21, 10, 10)
for i in range(attention_spatial_sv_time.shape[0]):  # 这里的范围应该是21
    attention_spatial_sv_time_i = attention_spatial_sv_time[i]  # 这里的shape应该是（10，10）
    print('attention_spatial_sv_time_i:',np.shape(attention_spatial_sv_time_i))  # (10,10)
    attention_spatial_sv_time_i_onehang = attention_spatial_sv_time_i[0]  # 这里的shape应该是（10，），因为取了sv对其他对象的注意力
    print('attention_spatial_sv_time_i_onehang:', np.shape(attention_spatial_sv_time_i_onehang))  # (10,)

    # 检查是否所有元素都是0.1
    if np.all(attention_spatial_sv_time_i_onehang == 0.1):
        non_zero_attention = np.array([])  # 如果全部都是0.1，则返回一个空数组
        retained_indices = np.array([])  # 空数组表示没有保留的索引
    else:
        # 否则，去掉为0的元素
        retained_indices = np.where(attention_spatial_sv_time_i_onehang != 0)[0]
        non_zero_attention = attention_spatial_sv_time_i_onehang[retained_indices]

    print('non_zero_attention:', np.shape(non_zero_attention), non_zero_attention)  # 打印非零数据的shape和数据
    print('retained_indices:', retained_indices)  # 打印保留的索引

    attention_space.append(non_zero_attention)
    attention_space_retained_indices.append(retained_indices)
print('attention_space:',np.shape(attention_space),attention_space)  # 期待的shape是（21，10）
print('attention_space_retained_indices:',np.shape(attention_space_retained_indices),attention_space_retained_indices)

# 去掉空数组
filtered_attention_space = [arr for arr in attention_space if arr.size > 0]

# 构建一个 (21, 2) 的画布，并用 NaN 填充
heatmap_data = np.full((21, 2), np.nan)
for i, arr in enumerate(attention_space):
    if arr.size > 0:
        heatmap_data[i, :arr.size] = arr
    else:
        heatmap_data[i, :] = np.nan

# 生成热力图
heatmap_data[11,0] = 0.2
heatmap_data[11,1] = 0.8
plt.figure(figsize=(12, 6))
plt.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)


# 添加网格
plt.grid(color='black', linestyle='-', linewidth=1)
plt.colorbar(label='Attention Value')
plt.title('Heatmap of Attention Values')
plt.xlabel('Attention Index')
plt.ylabel('Data Index')
plt.yticks(ticks=np.arange(21), labels=np.arange(1, 22))
plt.xticks(ticks=np.arange(2), labels=np.arange(1, 3))

# 设置网格间隔和位置
plt.gca().set_xticks(np.arange(-0.5, 2.5, 1))
plt.gca().set_yticks(np.arange(0.5, 21, 1))
plt.gca().grid(which='major', color='black', linestyle='-', linewidth=1)
plt.savefig(root_attention + '\轨迹点处空间关注度\%s_generate.png' % (str(scenario_test) + '_' + 'agent_' + str(sv) + '_time_' + str(t_attention)), dpi=300)

plt.show()


# 读取主要讨论的agent的时间注意力

attention_temporal_sv_time = attention_temporal[sv][t_attention]  # 这个shape应该是（21，21）
attention_time = []
attention_time_retained_indices = []
print('attention_temporal_sv_time:',np.shape(attention_temporal_sv_time))  # (21, 21)

attention_temporal_sv_time_i_onehang = attention_temporal_sv_time[-1]  # 这里的shape应该是（21,）
print('attention_temporal_sv_time_i_onehang:',np.shape(attention_temporal_sv_time_i_onehang))  # (21,)


attention_time.append(attention_temporal_sv_time_i_onehang)
print('attention_time:',np.shape(attention_time),attention_time)  # (1, 21)

# 构建一个 (1, 21) 的画布，并用 NaN 填充
heatmap_data_time = np.full((1, 21), np.nan)
for i, arr in enumerate(attention_time):
    if arr.size > 0:
        heatmap_data_time[i, :arr.size] = arr
    else:
        heatmap_data_time[i, :] = np.nan

# 生成热力图
heatmap_data_time[0,20] = 0.5
# heatmap_data_time[11,1] = 0.8
plt.figure(figsize=(12, 6))
plt.imshow(heatmap_data_time, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)

# 添加网格
plt.grid(color='black', linestyle='-', linewidth=1)
plt.colorbar(label='Attention Value')
plt.title('Heatmap of Attention Values')
plt.xlabel('Attention Index')
plt.ylabel('Data Index')
plt.yticks(ticks=np.arange(1), labels=np.arange(1, 2))
plt.xticks(ticks=np.arange(21), labels=np.arange(1, 22))

# 设置网格间隔和位置
plt.gca().set_xticks(np.arange(0.5, 21, 1))
plt.gca().set_yticks(np.arange(1, 2, 1))

plt.gca().grid(which='major', color='black', linestyle='-', linewidth=1)
plt.savefig(root_attention + '\轨迹点处时间关注度\%s_generate.png' % (str(scenario_test) + '_' + 'agent_' + str(sv) + '_time_' + str(t_attention)), dpi=300)

plt.show()





#
# # 需要用模型计算attention
# # 空间维度
# # 提取当前时刻和前 20 个时刻的数据，不够 20 个时刻的部分用零填充
# obs_lstm_sv_t = np.zeros((1, 21, 57))
# if t_attention >= 20:
#     obs_lstm_sv_t = [generate_ob[sv][t_attention - 20:t_attention + 1]]
# else:
#     obs_lstm_sv_t[0][20 - t_attention:] = generate_ob[sv][:t_attention + 1]
# print('generate_ac:',np.shape(generate_ac),generate_ac[0],generate_ac[7][t_attention])
# a_v = np.zeros((1, 14))
# # a_v = np.concatenate([generate_ac[i][t_attention] for i in range(num_agents) if i != sv], axis=1)
# obs_lstm = np.zeros((1, 21*8, 57))
# is_training = False
# # 这里处理ob_lstm（8, batch, 21, 46）得到mask_atime（21，batch,10,10）, mask_times（batch,21,21）
# k_ob_lstm = obs_lstm_sv_t  # (21, 57)
# k_ob_lstm = np.array(k_ob_lstm)
# print('k_ob_lstm', np.shape(k_ob_lstm))
# num_batch = k_ob_lstm.shape[0]
# # 提取出第二个维度的大小，即 21
# num_time = k_ob_lstm.shape[1]
# # print('num_batch:', num_batch, 'num_time:', num_time)
# num_features = 10  # 每个序列的参数的个数
# step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# mask_atime_all = []  # 存放对于第k个agent来说，每一个时刻的mask矩阵，shape为（21，batch，10,10）
# mask_times = np.ones([num_batch, 21, 21], dtype=bool)  # 存放对于第k个agent来说，在时刻维度的mask矩阵，shape为（batch，21,21）
# for time_i in range(num_time):
#     k_ob_lstm_ONE_TIME = k_ob_lstm[:, time_i, :]  # (batch,46)
#     # print('k_ob_lstm_ONE_TIME:', np.shape(k_ob_lstm_ONE_TIME))
#     # 把k_ob_lstm_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
#     # 定义每个时间步长的特征数
#     # 以下是numpy版本下的代码
#     sub_inputsRag_all_np = []  # 存放每一个时刻的拆分之后的数据
#     # 遍历每个样本
#     for j in range(num_batch):
#         # 创建一个零张量
#         sub_inputsRag_j_np = np.zeros([0, num_features], dtype=np.float32)
#         # 记录当前位置
#         current_pos_np = 0
#         # 遍历每个特征的长度
#         for k_, step_size in enumerate(step_sizes_np):
#             # 截取当前时间步长的特征
#             feature_slice = k_ob_lstm_ONE_TIME[j, current_pos_np: current_pos_np + step_size]
#             # print('feature_slice:',np.shape(feature_slice))
#             # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
#             if feature_slice.shape[0] < num_features:
#                 pad_size = num_features - feature_slice.shape[0]
#                 feature_slice = np.append(feature_slice, np.zeros(pad_size))
#             # 在垂直方向堆叠
#             # print('feature_slice:',np.shape(feature_slice))
#             sub_inputsRag_j_np = np.concatenate(
#                 [sub_inputsRag_j_np, np.expand_dims(feature_slice, axis=0)], axis=0)
#             # 最后一步会得到(10,10)
#             # 更新当前位置
#             current_pos_np += step_size
#         sub_inputsRag_all_np.append(sub_inputsRag_j_np)
#         # 最后会得到（batch，10,10）
#
#     # 在垂直方向堆叠，形成 RaggedTensor
#     sub_inputsRag_np = np.stack(sub_inputsRag_all_np, axis=0)  # （nbatch, 10, 10)
#     # print('sub_inputsRag_np:', np.shape(sub_inputsRag_np))
#     # 形成这个时刻的mask （batch，10，10）
#     mask_atime = np.ones([num_batch, 10, 10], dtype=bool)
#     for j_mask in range(num_batch):
#         for i_mask in range(10):
#             if sub_inputsRag_np[j_mask, i_mask, 0] == 0:
#                 # 说明这个交互对象没有/agent没有
#                 mask_atime[j_mask, i_mask, :] = False
#                 mask_atime[j_mask, :, i_mask] = False
#
#     mask_atime_all.append(mask_atime)
# mask_atime_all_new = np.stack(mask_atime_all, axis=0)  # （21, nbatch, 10, 10)
#
# # 时间维度上的mask （batch,21,21）
# # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
# for i_batch in range(num_batch):
#     if k_ob_lstm[i_batch][20][0] == 0:  # k_ob_lstm (batch,21,46)
#         # 说明这个agent还没往前走或者是无效的，那么所有的都是要掩码的
#         mask_times[i_batch, :, :] = False
#     else:
#         # 说明这个agent往前走了，只需要找到在哪个时刻往前走的就可以了
#         for time_i_batch in range(num_time):
#             if k_ob_lstm[i_batch][time_i_batch][0] != 0:
#                 mask_times[i_batch, 0:time_i_batch, 0:time_i_batch] = False
#                 break  # 退出循环，已经找到第一个有效的时刻
# # print('测试下时间维度上的mask对不对,batch 0：',k_ob_lstm[0],mask_times[0])
# # print('测试下一个时间上的mask对不对,batch 0,最后一个时刻：', mask_atime_all_new[20][0])
# print('mask_atime_all_new:',np.shape(mask_atime_all_new),mask_atime_all_new)
# print('mask_times：', np.shape(mask_times),mask_times)
# model = makeModel(env_id)
# model.load(path)
# _, _, _, atten_weights_spatial, atten_weights_temporal = model.attention_step(sv, obs_lstm_sv_t, state, obs_lstm, a_v, is_training, mask_atime_all_new, mask_times)
#
# # 主agent SV 在t_attention时刻的注意力
# print('atten_weights_spatial:',np.shape(atten_weights_spatial), atten_weights_spatial)  # (21, 1, 10, 10)
# print('atten_weights_temporal:',np.shape(atten_weights_temporal), atten_weights_temporal)  # (1, 21, 21)
#
#
