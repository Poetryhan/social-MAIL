import os
import re
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib as mpl
import cv2
import glob
from PIL import Image
import pickle

"""
绘制序贯多车交互的专家轨迹视频
"""


from irl.render import makeModel, render, get_dis
from irl.mack.kfac_discriminator_airl import Discriminator
import pickle as pkl
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from shapely import geometry
from scipy.spatial import distance
from scipy import interpolate

from matplotlib.widgets import Button, Slider
from utils.DataReader import read_tracks_all, read_tracks_meta, read_light

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

# 输入的trj_in为该场景所有车辆轨迹信息（提取了我们需要的列）,只针对一帧来画图

def plot_top_view_ani_with_lidar_label(trj_in, scenario_landmark_trj, scenario_label_in, time_id_in, frame_id_in):
    # this function plots one single frame of the top view video
    # trj_in is a pandas with three columns(obj_id, frame_label, local_time_stamp, global_center_x, global_center_y,
    # length, width, heading)
    # trj_in is all the trajectories within one segment
    # seg_id_in is the current segment id
    # trj_in['center_x'] = trj_in['center_x'] - trj_in['center_x'].min()
    # trj_in['center_y'] = trj_in['center_y'] - trj_in['center_y'].min()
    unique_veh_id = pd.unique(trj_in['id']).tolist()
    plt.figure(figsize=(18, 13.5))
    plt.figure()
    plt.xlabel('center x (m)', fontsize=10)
    plt.ylabel('center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim(-5, 34)
    plt.ylim(14, 35)
    # max_range = max(trj_in['global_center_x'].max(), )
    title_name = '_Scenario label_' + str(scenario_label_in)
    plt.title(title_name, loc='left')
    # 设置横纵坐标轴刻度,round为四舍五入函数
    plt.xticks(
        np.arange(round(min(float(scenario_landmark_trj['x_real'].min()),float(trj_in['x_real'].min()))) - 10,
                  round(max(float(scenario_landmark_trj['x_real'].max()),float(trj_in['x_real'].max()))) + 10, 20),
        fontsize=5)
    plt.yticks(
        np.arange(round(min(float(scenario_landmark_trj['y_real'].min()), float(trj_in['y_real'].min()))) - 10,
                  round(max(float(scenario_landmark_trj['y_real'].max()), float(trj_in['y_real'].max()))) + 10, 20),
        fontsize=5)
    font = {'family': 'Times New Roman', 'size': 5}
    # plt.legend()
    ax = plt.gca()
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

    # find out the global heading of ego vehicle first, use it to transform other vehicles' local heading to global heading
    # ego：AV车
    # ego_veh_trj = trj_in.loc[trj_in['is_AV'] == 1, :]
    # ego_current_heading = ego_veh_trj.loc[ego_veh_trj['frame_label'] == frame_id, 'heading'].values[0]
    # get all the trajectories until current frame
    for single_veh_id in unique_veh_id:
        single_veh_trj = trj_in[trj_in['id'] == single_veh_id]
        single_veh_trj = single_veh_trj.copy()
        # single_veh_trj['frame_label'] = range(0, 139)
        # single_veh_trj.loc[:, 'frame_label'] = range(0, 41)

        single_veh_trj = single_veh_trj[(abs(single_veh_trj['time'] - time_id_in) == 0)&(single_veh_trj['x_real']!=-4)]

        if len(single_veh_trj) > 1:
            single_veh_trj_2 = pd.DataFrame(single_veh_trj.loc[abs(single_veh_trj['time'] - time_id_in).idxmin()]).T
            # print('有多条的single_veh_trj_2:', time_id_in, single_veh_trj_2)
        else:
            single_veh_trj_2 = single_veh_trj
            # print('只有一条的single_veh_trj_2:', time_id_in, single_veh_trj_2)

        if len(single_veh_trj_2) > 0:
            ts = ax.transData
            coords = [single_veh_trj_2['x_real'].iloc[0], single_veh_trj_2['y_real'].iloc[0]]

            veh_local_direction = single_veh_trj_2['direction'].iloc[0]
            if veh_local_direction == 'left':
                # 先绘制same agent交互对象
                coords_jiaohu_same_agent = [single_veh_trj_2['same_agent_1_x_real'].iloc[0],
                                            single_veh_trj_2['same_agent_1_y_real'].iloc[0]]
                if coords_jiaohu_same_agent[0] != -4:

                    same_agent_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                      & (trj_in['id'] != single_veh_id)
                                                      & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(same_agent_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        same_agent_trj_potential['distance'] = np.sqrt(
                            (same_agent_trj_potential['x_real'] - coords_jiaohu_same_agent[0]) ** 2
                            + (same_agent_trj_potential['y_real'] - coords_jiaohu_same_agent[1]) ** 2)
                        same_agent_trj_potential.index = range(len(same_agent_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = same_agent_trj_potential['distance'].idxmin()

                        # 获取最近距离的行

                        print('左转车有same agent')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        same_agent_jiaohu_Series = same_agent_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        same_agent_jiaohu = pd.DataFrame(same_agent_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_sameagent = round(single_veh_trj_2['weight_same1'].iloc[0],3)
                        heading_angle_sameagent = same_agent_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_sameagent = mpl.transforms.Affine2D().rotate_deg_around(same_agent_jiaohu['x_real'].iloc[0],
                                                                                   same_agent_jiaohu['y_real'].iloc[0],
                                                                                   heading_angle_sameagent)
                        t_sameagent = tr_sameagent + ts
                        # note that exact xy needs to bo calculated
                        sameagent_length = same_agent_jiaohu['length'].iloc[0]
                        sameagent_width = same_agent_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(same_agent_jiaohu['x_real'].iloc[0] - 0.5 * sameagent_length,
                                same_agent_jiaohu['y_real'].iloc[0] - 0.5 * sameagent_width),
                            width=sameagent_length,
                            height=sameagent_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_sameagent,
                            transform=t_sameagent, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(same_agent_jiaohu['x_real'].iloc[0] + 2,
                                             same_agent_jiaohu['y_real'].iloc[0] + 2,
                                             str(same_agent_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white', rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

                # 再绘制left agent 1 交互对象
                coords_jiaohu_left_agent1 = [single_veh_trj_2['left_agent_1_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_1_y_real'].iloc[0]]

                if coords_jiaohu_left_agent1[0] != -4:
                    left_agent1_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    print('left_agent1_trj_potential:', left_agent1_trj_potential)
                    if len(left_agent1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent1_trj_potential['distance'] = np.sqrt(
                            (left_agent1_trj_potential['x_real'] - coords_jiaohu_left_agent1[0]) ** 2
                            + (left_agent1_trj_potential['y_real'] - coords_jiaohu_left_agent1[1]) ** 2)

                        left_agent1_trj_potential.index = range(len(left_agent1_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent1_trj_potential['distance'].idxmin()
                        print('左转车有left agent1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent1_jiaohu_Series = left_agent1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent1_jiaohu = pd.DataFrame(left_agent1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent1 = round(single_veh_trj_2['weight_left1'].iloc[0],3)
                        heading_angle_leftagent1 = left_agent1_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_leftagent1 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent1_jiaohu['x_real'].iloc[0],
                            left_agent1_jiaohu['y_real'].iloc[0],heading_angle_leftagent1)
                        t_leftagent1 = tr_leftagent1 + ts
                        # note that exact xy needs to bo calculated
                        leftagent1_length = left_agent1_jiaohu['length'].iloc[0]
                        leftagent1_width = left_agent1_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent1_jiaohu['x_real'].iloc[0] - 0.5 * leftagent1_length,
                                left_agent1_jiaohu['y_real'].iloc[0] - 0.5 * leftagent1_width),
                            width=leftagent1_length,
                            height=leftagent1_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent1,
                            transform=t_leftagent1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent1_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent1_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent1_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white', rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

                # 再绘制left agent2交互对象
                coords_jiaohu_left_agent2 = [single_veh_trj_2['left_agent_2_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_2_y_real'].iloc[0]]
                if coords_jiaohu_left_agent2[0] != -4:
                    left_agent2_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(left_agent2_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent2_trj_potential['distance'] = np.sqrt(
                            (left_agent2_trj_potential['x_real'] - coords_jiaohu_left_agent2[0]) ** 2
                            + (left_agent2_trj_potential['y_real'] - coords_jiaohu_left_agent2[1]) ** 2)

                        left_agent2_trj_potential.index = range(len(left_agent2_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent2_trj_potential['distance'].idxmin()
                        print('左转车有left agent2')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent2_jiaohu_Series = left_agent2_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent2_jiaohu = pd.DataFrame(left_agent2_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent2 = round(single_veh_trj_2['weight_left2'].iloc[0],3)
                        heading_angle_leftagent2 = left_agent2_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_leftagent2 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent2_jiaohu['x_real'].iloc[0],
                            left_agent2_jiaohu['y_real'].iloc[0],heading_angle_leftagent2)
                        t_leftagent2 = tr_leftagent2 + ts
                        # note that exact xy needs to bo calculated
                        leftagent2_length = left_agent2_jiaohu['length'].iloc[0]
                        leftagent2_width = left_agent2_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent2_jiaohu['x_real'].iloc[0] - 0.5 * leftagent2_length,
                                left_agent2_jiaohu['y_real'].iloc[0] - 0.5 * leftagent2_width),
                            width=leftagent2_length,
                            height=leftagent2_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent2,
                            transform=t_leftagent2, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent2_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent2_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent2_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制left agent3交互对象
                coords_jiaohu_left_agent3 = [single_veh_trj_2['left_agent_3_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_3_y_real'].iloc[0]]
                if coords_jiaohu_left_agent3[0] != -4:
                    left_agent3_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(left_agent3_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent3_trj_potential['distance'] = np.sqrt(
                            (left_agent3_trj_potential['x_real'] - coords_jiaohu_left_agent3[0]) ** 2
                            + (left_agent3_trj_potential['y_real'] - coords_jiaohu_left_agent3[1]) ** 2)

                        left_agent3_trj_potential.index = range(len(left_agent3_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent3_trj_potential['distance'].idxmin()
                        print('左转车有left agent3')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent3_jiaohu_Series = left_agent3_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent3_jiaohu = pd.DataFrame(left_agent3_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent3 = round(single_veh_trj_2['weight_left3'].iloc[0],3)
                        heading_angle_leftagent3 = left_agent3_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_leftagent3 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent3_jiaohu['x_real'].iloc[0],
                            left_agent3_jiaohu['y_real'].iloc[0],heading_angle_leftagent3)
                        t_leftagent3 = tr_leftagent3 + ts
                        # note that exact xy needs to bo calculated
                        leftagent3_length = left_agent3_jiaohu['length'].iloc[0]
                        leftagent3_width = left_agent3_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent3_jiaohu['x_real'].iloc[0] - 0.5 * leftagent3_length,
                                left_agent3_jiaohu['y_real'].iloc[0] - 0.5 * leftagent3_width),
                            width=leftagent3_length,
                            height=leftagent3_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent3,
                            transform=t_leftagent3, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent3_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent3_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent3_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent 1 交互对象
                coords_jiaohu_right_agent1 = [single_veh_trj_2['right_agent_1_x_real'].iloc[0],
                                              single_veh_trj_2['right_agent_1_y_real'].iloc[0]]
                if coords_jiaohu_right_agent1[0] != -4:
                    right_agent1_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    print('right_agent1_trj_potential:', right_agent1_trj_potential)
                    if len(right_agent1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent1_trj_potential['distance'] = np.sqrt(
                            (right_agent1_trj_potential['x_real'] - coords_jiaohu_right_agent1[0]) ** 2
                            + (right_agent1_trj_potential['y_real'] - coords_jiaohu_right_agent1[1]) ** 2)

                        right_agent1_trj_potential.index = range(len(right_agent1_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent1_trj_potential['distance'].idxmin()
                        print('左转车有right agent1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent1_jiaohu_Series = right_agent1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent1_jiaohu = pd.DataFrame(right_agent1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent1 = round(single_veh_trj_2['weight_right1'].iloc[0],3)
                        heading_angle_rightagent1 = right_agent1_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_rightagent1 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent1_jiaohu['x_real'].iloc[0],
                            right_agent1_jiaohu['y_real'].iloc[0],heading_angle_rightagent1)
                        t_rightagent1 = tr_rightagent1 + ts
                        # note that exact xy needs to bo calculated
                        rightagent1_length = right_agent1_jiaohu['length'].iloc[0]
                        rightagent1_width = right_agent1_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent1_jiaohu['x_real'].iloc[0] - 0.5 * rightagent1_length,
                                right_agent1_jiaohu['y_real'].iloc[0] - 0.5 * rightagent1_width),
                            width=rightagent1_length,
                            height=rightagent1_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent1,
                            transform=t_rightagent1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent1_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent1_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent1_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent2交互对象
                coords_jiaohu_right_agent2 = [single_veh_trj_2['right_agent_2_x_real'].iloc[0],
                                              single_veh_trj_2['right_agent_2_y_real'].iloc[0]]
                if coords_jiaohu_right_agent2[0] != -4:
                    right_agent2_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(right_agent2_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent2_trj_potential['distance'] = np.sqrt(
                            (right_agent2_trj_potential['x_real'] - coords_jiaohu_right_agent2[0]) ** 2
                            + (right_agent2_trj_potential['y_real'] - coords_jiaohu_right_agent2[1]) ** 2)

                        right_agent2_trj_potential.index = range(len(right_agent2_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent2_trj_potential['distance'].idxmin()
                        print('左转车有right agent2')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent2_jiaohu_Series = right_agent2_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent2_jiaohu = pd.DataFrame(right_agent2_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent2 = round(single_veh_trj_2['weight_right2'].iloc[0],3)
                        heading_angle_rightagent2 = right_agent2_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_rightagent2 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent2_jiaohu['x_real'].iloc[0],
                            right_agent2_jiaohu['y_real'].iloc[0],heading_angle_rightagent2)
                        t_rightagent2 = tr_rightagent2 + ts
                        # note that exact xy needs to bo calculated
                        rightagent2_length = right_agent2_jiaohu['length'].iloc[0]
                        rightagent2_width = right_agent2_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent2_jiaohu['x_real'].iloc[0] - 0.5 * rightagent2_length,
                                right_agent2_jiaohu['y_real'].iloc[0] - 0.5 * rightagent2_width),
                            width=rightagent2_length,
                            height=rightagent2_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent2,
                            transform=t_rightagent2, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent2_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent2_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent2_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent3交互对象
                coords_jiaohu_right_agent3 = [single_veh_trj_2['left_agent_3_x_real'].iloc[0],
                                              single_veh_trj_2['left_agent_3_y_real'].iloc[0]]
                if coords_jiaohu_right_agent3[0] != -4:
                    right_agent3_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(right_agent3_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent3_trj_potential['distance'] = np.sqrt(
                            (right_agent3_trj_potential['x_real'] - coords_jiaohu_right_agent3[0]) ** 2
                            + (right_agent3_trj_potential['y_real'] - coords_jiaohu_right_agent3[1]) ** 2)

                        right_agent3_trj_potential.index = range(len(right_agent3_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent3_trj_potential['distance'].idxmin()
                        print('左转车有right agent3')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent3_jiaohu_Series = right_agent3_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent3_jiaohu = pd.DataFrame(right_agent3_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent3 = round(single_veh_trj_2['weight_right3'].iloc[0],3)
                        heading_angle_rightagent3 = right_agent3_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_rightagent3 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent3_jiaohu['x_real'].iloc[0],
                            right_agent3_jiaohu['y_real'].iloc[0],
                            heading_angle_rightagent3)
                        t_rightagent3 = tr_rightagent3 + ts
                        # note that exact xy needs to bo calculated
                        rightagent3_length = right_agent3_jiaohu['length'].iloc[0]
                        rightagent3_width = right_agent3_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent3_jiaohu['x_real'].iloc[0] - 0.5 * rightagent3_length,
                                right_agent3_jiaohu['y_real'].iloc[0] - 0.5 * rightagent3_width),
                            width=rightagent3_length,
                            height=rightagent3_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent3,
                            transform=t_rightagent3, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent3_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent3_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent3_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制left landmark1 交互对象
                coords_jiaohu_left_landmark1 = [single_veh_trj_2['left_landmark_1_x_real'].iloc[0],
                                                single_veh_trj_2['left_landmark_1_y_real'].iloc[0]]
                if coords_jiaohu_left_landmark1[0] != -5:
                    left_landmark1_trj_potential = scenario_landmark_trj[(scenario_landmark_trj['x_real'] != -5)
                                                                         & (abs(
                        scenario_landmark_trj['time'] - time_id_in) == 0)]
                    if len(left_landmark1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_landmark1_trj_potential['distance'] = np.sqrt(
                            (left_landmark1_trj_potential['x_real'] - coords_jiaohu_left_landmark1[0]) ** 2
                            + (left_landmark1_trj_potential['y_real'] - coords_jiaohu_left_landmark1[1]) ** 2)

                        left_landmark1_trj_potential.index = range(len(left_landmark1_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_landmark1_trj_potential['distance'].idxmin()
                        print('左转车有left landmark1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_landmark1_jiaohu_Series = left_landmark1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_landmark1_jiaohu = pd.DataFrame(left_landmark1_jiaohu_Series).T
                        print('left_landmark1_jiaohu:', type(left_landmark1_jiaohu), left_landmark1_jiaohu)
                        # 进行相应的处理...
                        # temp_alpha_leftlandmark1 = round(single_veh_trj_2['weight_left_landmark1'].iloc[0],3)
                        heading_angle_leftlandmark1 = left_landmark1_jiaohu['angle_last_real'].iloc[0]
                        tr_leftlandmark1 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_landmark1_jiaohu['x_real'].iloc[0],
                            left_landmark1_jiaohu['y_real'].iloc[0],
                            heading_angle_leftlandmark1)
                        t_leftlandmark1 = tr_leftlandmark1 + ts
                        # note that exact xy needs to bo calculated
                        leftlandmark1_length = 1
                        leftlandmark1_width = 1
                        ax.add_patch(patches.Rectangle(
                            xy=(left_landmark1_jiaohu['x_real'].iloc[0] - 0.5 * leftlandmark1_length,
                                left_landmark1_jiaohu['y_real'].iloc[0] - 0.5 * leftlandmark1_width),
                            width=leftlandmark1_length,
                            height=leftlandmark1_width,
                            linewidth=1,
                            facecolor='purple',
                            edgecolor='purple',
                            # alpha=temp_alpha_leftlandmark1,
                            transform=t_leftlandmark1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_landmark1_jiaohu['x_real'].iloc[0] + 2,
                                             left_landmark1_jiaohu['y_real'].iloc[0] + 2,
                                             str('ld') + str(int(left_landmark1_jiaohu['id'].iloc[0])),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right landmark1 交互对象
                coords_jiaohu_right_landmark1 = [single_veh_trj_2['right_landmark_1_x_real'].iloc[0],
                                                 single_veh_trj_2['right_landmark_1_y_real'].iloc[0]]
                if coords_jiaohu_right_landmark1[0] != -5:
                    right_landmark1_trj_potential = scenario_landmark_trj[(scenario_landmark_trj['x_real'] != -5)
                                                                          & (abs(
                        scenario_landmark_trj['time'] - time_id_in) == 0)]
                    if len(right_landmark1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_landmark1_trj_potential['distance'] = np.sqrt(
                            (right_landmark1_trj_potential['x_real'] - coords_jiaohu_right_landmark1[0]) ** 2
                            + (right_landmark1_trj_potential['y_real'] - coords_jiaohu_right_landmark1[1]) ** 2)
                        right_landmark1_trj_potential.index = range(len(right_landmark1_trj_potential))

                        # 找到距离最小的行索引
                        min_distance_idx = right_landmark1_trj_potential['distance'].idxmin()
                        print('左转车有right landmark1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_landmark1_jiaohu_Series = right_landmark1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_landmark1_jiaohu = pd.DataFrame(right_landmark1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightlandmark1 = round(single_veh_trj_2['weight_right_landmark1'].iloc[0],3)
                        heading_angle_rightlandmark1 = right_landmark1_jiaohu['angle_last_real'].iloc[0]
                        tr_rightlandmark1 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_landmark1_jiaohu['x_real'].iloc[0],
                            right_landmark1_jiaohu['y_real'].iloc[0],
                            heading_angle_rightlandmark1)
                        t_rightlandmark1 = tr_rightlandmark1 + ts
                        # note that exact xy needs to bo calculated
                        rightlandmark1_length = 1
                        rightlandmark1_width = 1
                        ax.add_patch(patches.Rectangle(
                            xy=(right_landmark1_jiaohu['x_real'].iloc[0] - 0.5 * rightlandmark1_length,
                                right_landmark1_jiaohu['y_real'].iloc[0] - 0.5 * rightlandmark1_width),
                            width=rightlandmark1_length,
                            height=rightlandmark1_width,
                            linewidth=1,
                            facecolor='purple',
                            edgecolor='purple',
                            # alpha=temp_alpha_rightlandmark1,
                            transform=t_rightlandmark1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_landmark1_jiaohu['x_real'].iloc[0] + 2,
                                             right_landmark1_jiaohu['y_real'].iloc[0] + 2,
                                             str('ld') + str(int(right_landmark1_jiaohu['id'].iloc[0])),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])


                # 自身左转车的颜色和透明度
                temp_alpha_agent = 0.99
                temp_facecolor_agent = 'blue'
            else:
                # 先绘制same agent交互对象
                coords_jiaohu_same_agent = [single_veh_trj_2['same_agent_1_x_real'].iloc[0],
                                            single_veh_trj_2['same_agent_1_y_real'].iloc[0]]
                if coords_jiaohu_same_agent[0] != -4:

                    same_agent_trj_potential = trj_in[(trj_in['direction'] == 'straight') & (trj_in['x_real'] != -4)
                                                      & (trj_in['id'] != single_veh_id)
                                                      & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(same_agent_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        same_agent_trj_potential['distance'] = np.sqrt(
                            (same_agent_trj_potential['x_real'] - coords_jiaohu_same_agent[0]) ** 2
                            + (same_agent_trj_potential['y_real'] - coords_jiaohu_same_agent[1]) ** 2)
                        same_agent_trj_potential.index = range(len(same_agent_trj_potential))

                        # 找到距离最小的行索引
                        min_distance_idx = same_agent_trj_potential['distance'].idxmin()

                        # 获取最近距离的行

                        print('直行车有same agent')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        same_agent_jiaohu_Series = same_agent_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        same_agent_jiaohu = pd.DataFrame(same_agent_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_sameagent = round(single_veh_trj_2['weight_same1'].iloc[0],3)
                        heading_angle_sameagent = same_agent_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_sameagent = mpl.transforms.Affine2D().rotate_deg_around(same_agent_jiaohu['x_real'].iloc[0],
                                                                                   same_agent_jiaohu['y_real'].iloc[0],
                                                                                   heading_angle_sameagent)
                        t_sameagent = tr_sameagent + ts
                        # note that exact xy needs to bo calculated
                        sameagent_length = same_agent_jiaohu['length'].iloc[0]
                        sameagent_width = same_agent_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(same_agent_jiaohu['x_real'].iloc[0] - 0.5 * sameagent_length,
                                same_agent_jiaohu['y_real'].iloc[0] - 0.5 * sameagent_width),
                            width=sameagent_length,
                            height=sameagent_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_sameagent,
                            transform=t_sameagent, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(same_agent_jiaohu['x_real'].iloc[0] + 2,
                                             same_agent_jiaohu['y_real'].iloc[0] + 2,
                                             str(same_agent_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white', rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

                # 再绘制left agent 1 交互对象
                coords_jiaohu_left_agent1 = [single_veh_trj_2['left_agent_1_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_1_y_real'].iloc[0]]

                if coords_jiaohu_left_agent1[0] != -4:
                    left_agent1_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    print('left_agent1_trj_potential:', left_agent1_trj_potential)
                    if len(left_agent1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent1_trj_potential['distance'] = np.sqrt(
                            (left_agent1_trj_potential['x_real'] - coords_jiaohu_left_agent1[0]) ** 2
                            + (left_agent1_trj_potential['y_real'] - coords_jiaohu_left_agent1[1]) ** 2)

                        left_agent1_trj_potential.index = range(len(left_agent1_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent1_trj_potential['distance'].idxmin()
                        print('直行车有left agent1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent1_jiaohu_Series = left_agent1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent1_jiaohu = pd.DataFrame(left_agent1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent1 = round(single_veh_trj_2['weight_left1'].iloc[0],3)
                        heading_angle_leftagent1 = left_agent1_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_leftagent1 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent1_jiaohu['x_real'].iloc[0],
                            left_agent1_jiaohu['y_real'].iloc[0],
                            heading_angle_leftagent1)
                        t_leftagent1 = tr_leftagent1 + ts
                        # note that exact xy needs to bo calculated
                        leftagent1_length = left_agent1_jiaohu['length'].iloc[0]
                        leftagent1_width = left_agent1_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent1_jiaohu['x_real'].iloc[0] - 0.5 * leftagent1_length,
                                left_agent1_jiaohu['y_real'].iloc[0] - 0.5 * leftagent1_width),
                            width=leftagent1_length,
                            height=leftagent1_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent1,
                            transform=t_leftagent1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent1_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent1_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent1_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white', rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

                # 再绘制left agent2交互对象
                coords_jiaohu_left_agent2 = [single_veh_trj_2['left_agent_2_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_2_y_real'].iloc[0]]
                if coords_jiaohu_left_agent2[0] != -4:
                    left_agent2_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(left_agent2_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent2_trj_potential['distance'] = np.sqrt(
                            (left_agent2_trj_potential['x_real'] - coords_jiaohu_left_agent2[0]) ** 2
                            + (left_agent2_trj_potential['y_real'] - coords_jiaohu_left_agent2[1]) ** 2)

                        left_agent2_trj_potential.index = range(len(left_agent2_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent2_trj_potential['distance'].idxmin()
                        print('直行车有left agent2')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent2_jiaohu_Series = left_agent2_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent2_jiaohu = pd.DataFrame(left_agent2_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent2 = round(single_veh_trj_2['weight_left2'].iloc[0],3)
                        heading_angle_leftagent2 = left_agent2_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_leftagent2 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent2_jiaohu['x_real'].iloc[0],
                            left_agent2_jiaohu['y_real'].iloc[0],
                            heading_angle_leftagent2)
                        t_leftagent2 = tr_leftagent2 + ts
                        # note that exact xy needs to bo calculated
                        leftagent2_length = left_agent2_jiaohu['length'].iloc[0]
                        leftagent2_width = left_agent2_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent2_jiaohu['x_real'].iloc[0] - 0.5 * leftagent2_length,
                                left_agent2_jiaohu['y_real'].iloc[0] - 0.5 * leftagent2_width),
                            width=leftagent2_length,
                            height=leftagent2_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent2,
                            transform=t_leftagent2, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent2_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent2_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent2_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制left agent3交互对象
                coords_jiaohu_left_agent3 = [single_veh_trj_2['left_agent_3_x_real'].iloc[0],
                                             single_veh_trj_2['left_agent_3_y_real'].iloc[0]]
                if coords_jiaohu_left_agent3[0] != -4:
                    left_agent3_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                       & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(left_agent3_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_agent3_trj_potential['distance'] = np.sqrt(
                            (left_agent3_trj_potential['x_real'] - coords_jiaohu_left_agent3[0]) ** 2
                            + (left_agent3_trj_potential['y_real'] - coords_jiaohu_left_agent3[1]) ** 2)

                        left_agent3_trj_potential.index = range(len(left_agent3_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = left_agent3_trj_potential['distance'].idxmin()
                        print('直行车有left agent3')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_agent3_jiaohu_Series = left_agent3_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_agent3_jiaohu = pd.DataFrame(left_agent3_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_leftagent3 = round(single_veh_trj_2['weight_left3'].iloc[0],3)
                        heading_angle_leftagent3 = left_agent3_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_leftagent3 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_agent3_jiaohu['x_real'].iloc[0],
                            left_agent3_jiaohu['y_real'].iloc[0],
                            heading_angle_leftagent3)
                        t_leftagent3 = tr_leftagent3 + ts
                        # note that exact xy needs to bo calculated
                        leftagent3_length = left_agent3_jiaohu['length'].iloc[0]
                        leftagent3_width = left_agent3_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(left_agent3_jiaohu['x_real'].iloc[0] - 0.5 * leftagent3_length,
                                left_agent3_jiaohu['y_real'].iloc[0] - 0.5 * leftagent3_width),
                            width=leftagent3_length,
                            height=leftagent3_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_leftagent3,
                            transform=t_leftagent3, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_agent3_jiaohu['x_real'].iloc[0] + 2,
                                             left_agent3_jiaohu['y_real'].iloc[0] + 2,
                                             str(left_agent3_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent 1 交互对象
                coords_jiaohu_right_agent1 = [single_veh_trj_2['right_agent_1_x_real'].iloc[0],
                                              single_veh_trj_2['right_agent_1_y_real'].iloc[0]]
                if coords_jiaohu_right_agent1[0] != -4:
                    right_agent1_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    print('right_agent1_trj_potential:', right_agent1_trj_potential)
                    if len(right_agent1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent1_trj_potential['distance'] = np.sqrt(
                            (right_agent1_trj_potential['x_real'] - coords_jiaohu_right_agent1[0]) ** 2
                            + (right_agent1_trj_potential['y_real'] - coords_jiaohu_right_agent1[1]) ** 2)

                        right_agent1_trj_potential.index = range(len(right_agent1_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent1_trj_potential['distance'].idxmin()
                        print('直行车有right agent1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent1_jiaohu_Series = right_agent1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent1_jiaohu = pd.DataFrame(right_agent1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent1 = round(single_veh_trj_2['weight_right1'].iloc[0],3)
                        heading_angle_rightagent1 = right_agent1_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_rightagent1 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent1_jiaohu['x_real'].iloc[0],
                            right_agent1_jiaohu['y_real'].iloc[0],
                            heading_angle_rightagent1)
                        t_rightagent1 = tr_rightagent1 + ts
                        # note that exact xy needs to bo calculated
                        rightagent1_length = right_agent1_jiaohu['length'].iloc[0]
                        rightagent1_width = right_agent1_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent1_jiaohu['x_real'].iloc[0] - 0.5 * rightagent1_length,
                                right_agent1_jiaohu['y_real'].iloc[0] - 0.5 * rightagent1_width),
                            width=rightagent1_length,
                            height=rightagent1_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent1,
                            transform=t_rightagent1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent1_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent1_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent1_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent2交互对象
                coords_jiaohu_right_agent2 = [single_veh_trj_2['right_agent_2_x_real'].iloc[0],
                                              single_veh_trj_2['right_agent_2_y_real'].iloc[0]]
                if coords_jiaohu_right_agent2[0] != -4:
                    right_agent2_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(right_agent2_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent2_trj_potential['distance'] = np.sqrt(
                            (right_agent2_trj_potential['x_real'] - coords_jiaohu_right_agent2[0]) ** 2
                            + (right_agent2_trj_potential['y_real'] - coords_jiaohu_right_agent2[1]) ** 2)

                        right_agent2_trj_potential.index = range(len(right_agent2_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent2_trj_potential['distance'].idxmin()
                        print('直行车有right agent2')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent2_jiaohu_Series = right_agent2_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent2_jiaohu = pd.DataFrame(right_agent2_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent2 = round(single_veh_trj_2['weight_right2'].iloc[0],3)
                        heading_angle_rightagent2 = right_agent2_jiaohu['angle_now_real'].iloc[0]  # angle_now_real heading_angle_last1_real
                        tr_rightagent2 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent2_jiaohu['x_real'].iloc[0],
                            right_agent2_jiaohu['y_real'].iloc[0],
                            heading_angle_rightagent2)
                        t_rightagent2 = tr_rightagent2 + ts
                        # note that exact xy needs to bo calculated
                        rightagent2_length = right_agent2_jiaohu['length'].iloc[0]
                        rightagent2_width = right_agent2_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent2_jiaohu['x_real'].iloc[0] - 0.5 * rightagent2_length,
                                right_agent2_jiaohu['y_real'].iloc[0] - 0.5 * rightagent2_width),
                            width=rightagent2_length,
                            height=rightagent2_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent2,
                            transform=t_rightagent2, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent2_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent2_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent2_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right agent3交互对象
                coords_jiaohu_right_agent3 = [single_veh_trj_2['left_agent_3_x_real'].iloc[0],
                                              single_veh_trj_2['left_agent_3_y_real'].iloc[0]]
                if coords_jiaohu_right_agent3[0] != -4:
                    right_agent3_trj_potential = trj_in[(trj_in['direction'] == 'left') & (trj_in['x_real'] != -4)
                                                        & (abs(trj_in['time'] - time_id_in) == 0)]
                    if len(right_agent3_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_agent3_trj_potential['distance'] = np.sqrt(
                            (right_agent3_trj_potential['x_real'] - coords_jiaohu_right_agent3[0]) ** 2
                            + (right_agent3_trj_potential['y_real'] - coords_jiaohu_right_agent3[1]) ** 2)

                        right_agent3_trj_potential.index = range(len(right_agent3_trj_potential))
                        # 找到距离最小的行索引
                        min_distance_idx = right_agent3_trj_potential['distance'].idxmin()
                        print('直行车有right agent3')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_agent3_jiaohu_Series = right_agent3_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_agent3_jiaohu = pd.DataFrame(right_agent3_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightagent3 = round(single_veh_trj_2['weight_right3'].iloc[0],3)
                        heading_angle_rightagent3 = right_agent3_jiaohu['angle_now_real'].iloc[0]  # angle_now_real  heading_angle_last1_real
                        tr_rightagent3 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_agent3_jiaohu['x_real'].iloc[0],
                            right_agent3_jiaohu['y_real'].iloc[0],
                            heading_angle_rightagent3)
                        t_rightagent3 = tr_rightagent3 + ts
                        # note that exact xy needs to bo calculated
                        rightagent3_length = right_agent3_jiaohu['length'].iloc[0]
                        rightagent3_width = right_agent3_jiaohu['width'].iloc[0]
                        ax.add_patch(patches.Rectangle(
                            xy=(right_agent3_jiaohu['x_real'].iloc[0] - 0.5 * rightagent3_length,
                                right_agent3_jiaohu['y_real'].iloc[0] - 0.5 * rightagent3_width),
                            width=rightagent3_length,
                            height=rightagent3_width,
                            linewidth=1,
                            facecolor='green',
                            edgecolor='green',
                            # alpha=temp_alpha_rightagent3,
                            transform=t_rightagent3, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_agent3_jiaohu['x_real'].iloc[0] + 2,
                                             right_agent3_jiaohu['y_real'].iloc[0] + 2,
                                             str(right_agent3_jiaohu['id'].iloc[0]),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制left landmark1 交互对象
                coords_jiaohu_left_landmark1 = [single_veh_trj_2['left_landmark_1_x_real'].iloc[0],
                                                single_veh_trj_2['left_landmark_1_y_real'].iloc[0]]
                if coords_jiaohu_left_landmark1[0] != -5:
                    left_landmark1_trj_potential = scenario_landmark_trj[(scenario_landmark_trj['x_real'] != -5)
                                                                         & (abs(
                        scenario_landmark_trj['time'] - time_id_in) == 0)]
                    if len(left_landmark1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        left_landmark1_trj_potential['distance'] = np.sqrt(
                            (left_landmark1_trj_potential['x_real'] - coords_jiaohu_left_landmark1[0]) ** 2
                            + (left_landmark1_trj_potential['y_real'] - coords_jiaohu_left_landmark1[1]) ** 2)
                        left_landmark1_trj_potential.index = range(len(left_landmark1_trj_potential))
                        # 找到距离最小的行索引
                        print('left_landmark1_trj_potential:',left_landmark1_trj_potential)
                        min_distance_idx = left_landmark1_trj_potential['distance'].idxmin()
                        print('直行车有left landmark1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        left_landmark1_jiaohu_Series = left_landmark1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        left_landmark1_jiaohu = pd.DataFrame(left_landmark1_jiaohu_Series).T
                        print('left_landmark1_jiaohu:',type(left_landmark1_jiaohu), left_landmark1_jiaohu)
                        # 进行相应的处理...
                        # temp_alpha_leftlandmark1 = round(single_veh_trj_2['weight_left_landmark1'].iloc[0],3)
                        heading_angle_leftlandmark1 = left_landmark1_jiaohu['angle_last_real'].iloc[0]
                        tr_leftlandmark1 = mpl.transforms.Affine2D().rotate_deg_around(
                            left_landmark1_jiaohu['x_real'].iloc[0],
                            left_landmark1_jiaohu['y_real'].iloc[0],
                            heading_angle_leftlandmark1)
                        t_leftlandmark1 = tr_leftlandmark1 + ts
                        # note that exact xy needs to bo calculated
                        leftlandmark1_length = 1
                        leftlandmark1_width = 1
                        ax.add_patch(patches.Rectangle(
                            xy=(left_landmark1_jiaohu['x_real'].iloc[0] - 0.5 * leftlandmark1_length,
                                left_landmark1_jiaohu['y_real'].iloc[0] - 0.5 * leftlandmark1_width),
                            width=leftlandmark1_length,
                            height=leftlandmark1_width,
                            linewidth=1,
                            facecolor='purple',
                            edgecolor='purple',
                            # alpha=temp_alpha_leftlandmark1,
                            transform=t_leftlandmark1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(left_landmark1_jiaohu['x_real'].iloc[0] + 2,
                                             left_landmark1_jiaohu['y_real'].iloc[0] + 2,
                                             str('ld') + str(int(left_landmark1_jiaohu['id'].iloc[0])),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 再绘制right landmark1 交互对象
                coords_jiaohu_right_landmark1 = [single_veh_trj_2['right_landmark_1_x_real'].iloc[0],
                                                 single_veh_trj_2['right_landmark_1_y_real'].iloc[0]]
                if coords_jiaohu_right_landmark1[0] != -5:
                    right_landmark1_trj_potential = scenario_landmark_trj[(scenario_landmark_trj['x_real'] != -5)
                                                                          & (abs(
                        scenario_landmark_trj['time'] - time_id_in) == 0)]
                    if len(right_landmark1_trj_potential) != 0:
                        # 计算每一行的欧几里得距离
                        right_landmark1_trj_potential['distance'] = np.sqrt(
                            (right_landmark1_trj_potential['x_real'] - coords_jiaohu_right_landmark1[0]) ** 2
                            + (right_landmark1_trj_potential['y_real'] - coords_jiaohu_right_landmark1[1]) ** 2)
                        right_landmark1_trj_potential.index = range(len(right_landmark1_trj_potential))

                        # 找到距离最小的行索引
                        min_distance_idx = right_landmark1_trj_potential['distance'].idxmin()
                        print('直行车有right landmark1')
                        # 这里进行满足条件后的处理，比如获取相应的行数据等
                        right_landmark1_jiaohu_Series = right_landmark1_trj_potential.iloc[min_distance_idx]  # 最接近的行数据
                        right_landmark1_jiaohu = pd.DataFrame(right_landmark1_jiaohu_Series).T
                        # 进行相应的处理...
                        # temp_alpha_rightlandmark1 = round(single_veh_trj_2['weight_right_landmark1'].iloc[0],3)
                        heading_angle_rightlandmark1 = right_landmark1_jiaohu['angle_last_real'].iloc[0]
                        tr_rightlandmark1 = mpl.transforms.Affine2D().rotate_deg_around(
                            right_landmark1_jiaohu['x_real'].iloc[0],
                            right_landmark1_jiaohu['y_real'].iloc[0],
                            heading_angle_rightlandmark1)
                        t_rightlandmark1 = tr_rightlandmark1 + ts
                        # note that exact xy needs to bo calculated
                        rightlandmark1_length = 1
                        rightlandmark1_width = 1
                        ax.add_patch(patches.Rectangle(
                            xy=(right_landmark1_jiaohu['x_real'].iloc[0] - 0.5 * rightlandmark1_length,
                                right_landmark1_jiaohu['y_real'].iloc[0] - 0.5 * rightlandmark1_width),
                            width=rightlandmark1_length,
                            height=rightlandmark1_width,
                            linewidth=1,
                            facecolor='purple',
                            edgecolor='purple',
                            # alpha=temp_alpha_rightlandmark1,
                            transform=t_rightlandmark1, zorder=1000))
                        # add vehicle local id for only vehicle object
                        # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
                        temp_text = plt.text(right_landmark1_jiaohu['x_real'].iloc[0] + 2,
                                             right_landmark1_jiaohu['y_real'].iloc[0] + 2,
                                             str('ld') + str(int(right_landmark1_jiaohu['id'].iloc[0])),
                                             style='italic',
                                             weight='heavy', ha='center', va='center', color='white',
                                             rotation=0,
                                             size=5, zorder=1000)
                        temp_text.set_path_effects(
                            [path_effects.Stroke(linewidth=1, foreground='black'),
                             path_effects.Normal()])

                # 直行车自身的颜色和透明度
                temp_alpha_agent = 0.99
                temp_facecolor_agent = 'red'

            heading_angle = single_veh_trj_2['angle_now_real'].iloc[0]  # 场景2的数据delta_theta是错误的，所以用heading_angle_last1_real代替angle_now_real，其他场景用正确的来做
            # if veh_local_direction == 'left':
            #     print('heading_angle:',heading_angle)
            # transform for other vehicles, note that the ego global heading should be added to current local heading
            tr_agent = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)

            t_agent = tr_agent + ts
            # note that exact xy needs to bo calculated
            veh_length = single_veh_trj_2['length'].iloc[0]
            veh_width = single_veh_trj_2['width'].iloc[0]
            ax.add_patch(patches.Rectangle(
                xy=(single_veh_trj_2['x_real'].iloc[0] - 0.5 * veh_length,
                    single_veh_trj_2['y_real'].iloc[0] - 0.5 * veh_width),
                width=veh_length,
                height=veh_width,
                linewidth=0.1,
                facecolor=temp_facecolor_agent,
                edgecolor='black',
                alpha=temp_alpha_agent,
                transform=t_agent, zorder=100))
            # add vehicle local id for only vehicle object
            # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
            temp_text = plt.text(single_veh_trj_2['x_real'].iloc[0]+2,
                                 single_veh_trj_2['y_real'].iloc[0]+2, str(single_veh_id), style='italic',
                                 weight='heavy', ha='center', va='center', color='white', rotation=0,
                                 size=5,zorder=100)
            temp_text.set_path_effects(
                [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    unique_landmark_id = pd.unique(scenario_landmark_trj['id']).tolist()
    for single_landmark_id in unique_landmark_id:
        single_landmark_trj = scenario_landmark_trj[scenario_landmark_trj['id'] == single_landmark_id]
        single_landmark_trj = single_landmark_trj.copy()

        single_landmark_trj = single_landmark_trj[abs(single_landmark_trj['time'] - time_id_in) == 0]

        if len(single_landmark_trj) > 1:
            single_landmark_trj_2 = pd.DataFrame(single_landmark_trj.loc[abs(single_landmark_trj['time'] - time_id_in).idxmin()]).T
            # print('有多条的single_veh_trj_2:', time_id_in, single_veh_trj_2)
        else:
            single_landmark_trj_2 = single_landmark_trj
            # print('只有一条的single_veh_trj_2:', time_id_in, single_veh_trj_2)

        if len(single_landmark_trj_2) > 0:
            ts_landmark = ax.transData

            coords_landmark = [single_landmark_trj_2['x_real'].iloc[0], single_landmark_trj_2['y_real'].iloc[0]]

            heading_angle_landmark = single_landmark_trj_2['angle_last_real'].iloc[0]
            # print('heading_angle:',heading_angle)
            # transform for other vehicles, note that the ego global heading should be added to current local heading
            tr_landmark = mpl.transforms.Affine2D().rotate_deg_around(coords_landmark[0], coords_landmark[1], heading_angle_landmark)

            t_landmark = tr_landmark + ts_landmark
            # note that exact xy needs to bo calculated
            # 暂时用虚假的大小代替landmark的形状，等后续重新提取landmark的时候，注意提取车辆类型和landmark
            landmark_length = 1
            landmark_width = 1

            ax.add_patch(patches.Rectangle(
                xy=(single_landmark_trj_2['x_real'].iloc[0] - 0.5 * landmark_length,
                    single_landmark_trj_2['y_real'].iloc[0] - 0.5 * landmark_width),
                width=landmark_length,
                height=landmark_width,
                linewidth=1,
                facecolor='white',
                edgecolor='black',
                alpha=1,
                transform=t_landmark,zorder=101))
            # add vehicle local id for only vehicle object
            # if single_veh_trj['obj_type'].iloc[0] == 1 or single_veh_trj['obj_type'].iloc[0] == 3:
            temp_text = plt.text(single_landmark_trj_2['x_real'].iloc[0]+2,
                                 single_landmark_trj_2['y_real'].iloc[0]+2, str('ld_'+ str(single_landmark_id)), style='italic',
                                 weight='heavy', ha='center', va='center', color='white', rotation=0,
                                 size=5,zorder=100)
            temp_text.set_path_effects(
                [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])


    trj_save_name = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                    r'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频\figure_save' \
                    r'\temp_top_view_figure\top_scenario_label_' + str(scenario_label_in) + '_frame_' + \
                    str(frame_id_in) + '_trajectory.jpg'
    plt.savefig(trj_save_name, dpi=600)
    plt.close('all')


def top_view_video_generation(start_frame_num, start_time, scenario_label):
    # this function experts one top view video based on top view figures from one segment
    img_array = []
    for num in range(start_frame_num,
                     start_frame_num + len(os.listdir(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                                      r'\MA_Intersection_straight\results_evaluate\v13'
                                                      r'\有注意力的视频\figure_save\temp_top_view_figure/'))):
        image_filename = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                         r'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频' \
                         r'\figure_save/temp_top_view_figure/' \
                         + 'top_scenario_label_' + str(scenario_label) + \
                         '_frame_' + str(num) + '_trajectory.jpg'
        # print(image_filename)
        img = Image.open(image_filename)  # Image loaded successfully.
        img = np.array(img)
        # img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # img

    video_save_name_0 = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                      fr'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频' \
                      fr'\figure_save\top_view_video\expert\{scenario_label}'

    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(video_save_name_0):
        os.makedirs(video_save_name_0)
    video_save_name = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                      fr'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频' \
                      fr'\figure_save\top_view_video\expert\{scenario_label}\'_scenario_label_{scenario_label}.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('top view video made success')
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                      r'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频\figure_save'
                                      r'\temp_top_view_figure\\', "*.jpg"))
    for f in filelist:
        os.remove(f)


def fill_in_size_parameter(scenario_trj_in):
    scenario_trj_out = pd.DataFrame()
    obj_trj_list = [name[1] for name in scenario_trj_in.groupby('agent_id')]
    for obj_trj in obj_trj_list:
        length_list = pd.unique(obj_trj['length']).tolist()
        width_list = pd.unique(obj_trj['width']).tolist()
        for single_length in length_list:
            if not pd.isnull(single_length):
                length = single_length
        for single_width in width_list:
            if not pd.isnull(single_width):
                width = single_width
        # obj_trj['length'] = length
        # obj_trj['width'] = width
        scenario_trj_out = pd.concat([scenario_trj_out, obj_trj], axis=0, ignore_index=True)
    return scenario_trj_out


if __name__ == '__main__':
    # ******************** Stage 2: visualization of lidar information ********************

    # ---------- process calculated trajectories from the csv lidar information file ----------
    # local_veh_id_generation_label = 1
    # a local vehicle ID will make each object more identifiable  # 生成一个新id，改为1则可以运行

    # to determine if the top view video should be expertd in this run
    top_view_video_generation_label = 1  # 改为1
    if top_view_video_generation_label:
        save_step = 416
        '''
        # 删除if else自己写
        if version_1_2_label:  # 如果变量version_1_2_label存在
            save_segment_id_start = 1000  # 场景编号的起始编号
            total_steps = 2  # 轨迹文件数
        else:
            save_segment_id_start = 0
            total_steps = 11
            '''
        # scenario_label = 9
        # scenario_name = str(scenario_label) + 'east_left'

        scenario_label = 23

        # all_expert_trj = pkl.load(open(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
        #                         r'\ATT-social-iniobs\MA_Intersection_straight'
        #                         r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1.pkl', 'rb'))
        #
        # path_landmark = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
        #               r'\ATT-social-iniobs\MA_Intersection_straight' \
        #               r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_landmarks_buxianshi.pkl'  # [95,97,6,14,6]

        all_expert_trj = pkl.load(
            open(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_social_dayu1.pkl', 'rb'))
        path_landmark = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_landmarks.pkl'  # [95,97,6,14,6]

        f_landmark = open(path_landmark, 'rb')
        all_landmark_trj = pickle.load(f_landmark)


        # all_expert_trj包含每一个场景的各类参数，第一层是场景编号，第二层是各类信息（包括ob attention_weight 等等），第三层是各类信息的具体值

        # delete previous frame jpgs (might or might not exist)
        filelist = glob.glob(os.path.join(
            r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
            r'\MA_Intersection_straight\results_evaluate\v13\有注意力的视频\figure_save\temp_top_view_figure\\',
            "*.jpg"))  # 存放中间步骤的图的路径
        for f in filelist:
            os.remove(f)
            print('删除了文件')

        # temp_trj['timestamp_ms'] = range(165)
        # or single_model_label == 5 or single_model_label == 15:
        #     or single_scenario_label == 6 or single_scenario_label == 6:
        # if single_scenario_label == 4:  # expert 只需要弄一个model就好，因为都是一样的
        scenario_print = 'Top view video now in phase: ' + str(scenario_label)
        print(scenario_print)

        scenario_trj_ob = all_expert_trj[scenario_label]['ob']

        scenario_trj_ac = all_expert_trj[scenario_label]['ac']
        print('scenario_trj_ob:',type(scenario_trj_ob),len(scenario_trj_ob))
        print('scenario_trj_ac:', type(scenario_trj_ac), len(scenario_trj_ac))

        left_scenario_use = pd.DataFrame()  # 存储该场景下有用的左转车轨迹数据
        straight_scenario_use = pd.DataFrame()  # 存储该场景下有用的直行车轨迹数据

        # 提取左转车
        for left_i in range(3):
            left_trj_i_array = scenario_trj_ob[left_i][:,:-1]
            left_trj_i = pd.DataFrame(left_trj_i_array)
            print('left_trj_i:',type(left_trj_i),left_trj_i.shape,left_trj_i)
            left_trj_i_use = left_trj_i[left_trj_i.iloc[:, 0] != 0]
                #  left_trj_i[left_trj_i[:, 0] != 0]
            print('left_trj_i_use:',type(left_trj_i_use),left_trj_i_use.shape,left_trj_i_use)

            left_ac_i_array = scenario_trj_ac[left_i]
            left_ac_i = pd.DataFrame(left_ac_i_array)
            print('left_ac_i:', type(left_ac_i),left_ac_i.shape,left_ac_i)
            left_ac_i_use = left_ac_i.loc[left_trj_i_use.index][1]  # 只有一列是delat_angle
            print('left_ac_i_use:', type(left_ac_i_use), left_ac_i_use.shape,left_ac_i_use)
            # 计算 angle_now 列，将第7列和 left_ac_i_use 相加
            left_trj_i_use['angle_now_real'] = left_trj_i_use.iloc[:, 6]*191-1 + ((2.8*(left_ac_i_use+1))/2) - 0.3
            left_trj_i_use['heading_angle_last1_real'] = left_trj_i_use.iloc[:, 6] * 191 - 1
            print('left_trj_i_use:', type(left_trj_i_use), left_trj_i_use.shape,left_trj_i_use)


            left_trj_i_use = left_trj_i_use.assign(id=left_i)
            print('left_i_use:',type(left_trj_i_use),left_trj_i_use.shape,left_trj_i_use)

            # 将索引转换为time列，并删除原始索引
            left_i_use_reset = left_trj_i_use.reset_index()

            # 将列名 'index' 更改为 'time'，位于第一列
            left_i_use_reset = left_i_use_reset.rename(columns={'index': 'time'})

            left_scenario_use = pd.concat([left_scenario_use, left_i_use_reset],axis=0)  # 纵向拼接
            print('left_scenario_use:', type(left_scenario_use), left_scenario_use.shape, left_scenario_use)


        left_scenario_use['direction'] = 'left'
        left_scenario_use['width'] = 1.6
        left_scenario_use['length'] = 4.0


        left_scenario_use.columns = ['time','x','y','vx','vy','dx','dy','heading_angle_last1','des','vx_last1','vy_last1',
                    'same_deltax_agent_1', 'same_deltay_agent_1', 'same_deltavx_agent_1', 'same_deltavy_agent_1',
                    'left_deltax_agent_1', 'left_deltay_agent_1', 'left_deltavx_agent_1', 'left_deltavy_agent_1',
                    'left_deltax_agent_2', 'left_deltay_agent_2', 'left_deltavx_agent_2', 'left_deltavy_agent_2',
                    'left_deltax_agent_3', 'left_deltay_agent_3', 'left_deltavx_agent_3', 'left_deltavy_agent_3',
                    'right_deltax_agent_1', 'right_deltay_agent_1', 'right_deltavx_agent_1', 'right_deltavy_agent_1',
                    'right_deltax_agent_2', 'right_deltay_agent_2', 'right_deltavx_agent_2', 'right_deltavy_agent_2',
                    'right_deltax_agent_3', 'right_deltay_agent_3', 'right_deltavx_agent_3', 'right_deltavy_agent_3',
                    'left_deltax_landmark_1', 'left_deltay_landmark_1', 'left_deltavx_landmark_1', 'left_deltavy_landmark_1',
                    'right_deltax_landmark_1', 'right_deltay_landmark_1', 'right_deltavx_landmark_1', 'right_deltavy_landmark_1',
                    'same_heading_last','left1_heading_last','left2_heading_last','left3_heading_last',
                    'right1_heading_last','right2_heading_last','right3_heading_last',
                    'landmark1_heading_last','landmark2_heading_last','min_distance_to_lane', 'last_delta_angle',
                    'angle_now_real','heading_angle_last1_real',
                            'id', 'direction', 'width', 'length']  # 九个交互对象的state，10+1*4+3*4+3*4+2*4

        left_scenario_use['x_real'] = left_scenario_use['x'] * 38 - 4
        left_scenario_use['y_real'] = left_scenario_use['y'] * 23 + 14

        # left_scenario_use['angle_now_real'] = left_scenario_use['angle_now'] * 191 - 1
        left_scenario_use['same_agent_1_x_real'] = None
        left_scenario_use['same_agent_1_y_real'] = None
        left_scenario_use['left_agent_1_x_real'] = None
        left_scenario_use['left_agent_1_y_real'] = None
        left_scenario_use['left_agent_2_x_real'] = None
        left_scenario_use['left_agent_2_y_real'] = None
        left_scenario_use['left_agent_3_x_real'] = None
        left_scenario_use['left_agent_3_y_real'] = None
        left_scenario_use['right_agent_1_x_real'] = None
        left_scenario_use['right_agent_1_y_real'] = None
        left_scenario_use['right_agent_2_x_real'] = None
        left_scenario_use['right_agent_2_y_real'] = None
        left_scenario_use['right_agent_3_x_real'] = None
        left_scenario_use['right_agent_3_y_real'] = None
        left_scenario_use['left_landmark_1_x_real'] = None
        left_scenario_use['left_landmark_1_y_real'] = None
        left_scenario_use['right_landmark_1_x_real'] = None
        left_scenario_use['right_landmark_1_y_real'] = None


        left_scenario_use['same_agent_1_x_real'] = np.where(
            left_scenario_use['same_deltax_agent_1'] == 0, -4,
            left_scenario_use['x_real'] - left_scenario_use['same_deltax_agent_1'] * 27 - 12)
        left_scenario_use['same_agent_1_y_real'] = np.where(
            left_scenario_use['same_deltay_agent_1'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use['same_deltay_agent_1'] * 18 - 14))

        left_scenario_use['left_agent_1_x_real'] = np.where(
            left_scenario_use['left_deltax_agent_1'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'left_deltax_agent_1'] * 30 - 15))
        left_scenario_use['left_agent_1_y_real'] = np.where(
            left_scenario_use['left_deltay_agent_1'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'left_deltay_agent_1'] * 14 - 7))
        left_scenario_use['left_agent_2_x_real'] = np.where(
            left_scenario_use['left_deltax_agent_2'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'left_deltax_agent_2'] * 30 - 15))
        left_scenario_use['left_agent_2_y_real'] = np.where(
            left_scenario_use['left_deltay_agent_2'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'left_deltay_agent_2'] * 14 - 7))
        left_scenario_use['left_agent_3_x_real'] = np.where(
            left_scenario_use['left_deltax_agent_3'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'left_deltax_agent_3'] * 5 - 14))
        left_scenario_use['left_agent_3_y_real'] = np.where(
            left_scenario_use['left_deltay_agent_3'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'left_deltay_agent_3'] * 4 - 6))

        left_scenario_use['right_agent_1_x_real'] = np.where(
            left_scenario_use['right_deltax_agent_1'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'right_deltax_agent_1'] * 30 - 15))
        left_scenario_use['right_agent_1_y_real'] = np.where(
            left_scenario_use['right_deltay_agent_1'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'right_deltay_agent_1'] * 24 - 15))
        left_scenario_use['right_agent_2_x_real'] = np.where(
            left_scenario_use['right_deltax_agent_2'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'right_deltax_agent_2'] * 30 - 15))
        left_scenario_use['right_agent_2_y_real'] = np.where(
            left_scenario_use['right_deltay_agent_2'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'right_deltay_agent_2'] * 24 - 15))
        left_scenario_use['right_agent_3_x_real'] = np.where(
            left_scenario_use['right_deltax_agent_3'] == 0, -4,
            left_scenario_use['x_real'] - (left_scenario_use[
            'right_deltax_agent_3'] * 4 - 15))
        left_scenario_use['right_agent_3_y_real'] = np.where(
            left_scenario_use['right_deltay_agent_3'] == 0, 14,
            left_scenario_use['y_real'] - (left_scenario_use[
            'right_deltay_agent_3'] * 4 + 2))

        left_scenario_use['left_landmark_1_x_real'] = np.where(
            left_scenario_use['left_deltax_landmark_1'] == 0, -5,
            left_scenario_use['x_real'] - (left_scenario_use[
            'left_deltax_landmark_1'] * 29 - 14))
        left_scenario_use['left_landmark_1_y_real'] = np.where(
            left_scenario_use['left_deltay_landmark_1'] == 0, -3,
            left_scenario_use['y_real'] - (left_scenario_use[
            'left_deltay_landmark_1'] * 30 - 15))
        left_scenario_use['right_landmark_1_x_real'] = np.where(
            left_scenario_use['right_deltax_landmark_1'] == 0, -5,
            left_scenario_use['x_real'] - (left_scenario_use[
            'right_deltax_landmark_1'] * 30 - 15))
        left_scenario_use['right_landmark_1_y_real'] = np.where(
            left_scenario_use['right_deltay_landmark_1'] == 0, -3,
            left_scenario_use['y_real'] - (left_scenario_use[
            'right_deltay_landmark_1'] * 29 - 15))


        # 提取直行车
        for straight_i in range(3,8):
            straight_trj_i_array = scenario_trj_ob[straight_i][:,:-1]
            straight_trj_i = pd.DataFrame(straight_trj_i_array)
            print('straight_trj_i:', type(straight_trj_i), straight_trj_i.shape, straight_trj_i)
            straight_trj_i_use = straight_trj_i[straight_trj_i.iloc[:, 0] != 0]
            #  straight_trj_i[straight_trj_i[:, 0] != 0]
            print('straight_trj_i_use:', type(straight_trj_i_use), straight_trj_i_use.shape, straight_trj_i_use)

            straight_ac_i_array = scenario_trj_ac[straight_i]
            straight_ac_i = pd.DataFrame(straight_ac_i_array)
            print('straight_ac_i:', type(straight_ac_i), straight_ac_i.shape, straight_ac_i)
            straight_ac_i_use = straight_ac_i.loc[straight_trj_i_use.index][1]  # 只有一列是delat_angle
            print('straight_ac_i_use:', type(straight_ac_i_use), straight_ac_i_use.shape, straight_ac_i_use)
            # 计算 angle_now 列，将第7列和 straight_ac_i_use 相加
            straight_trj_i_use['angle_now_real'] = straight_trj_i_use.iloc[:, 6] * 191 - 1 + ((2.4*(straight_ac_i_use+1))/2) - 1.2
            straight_trj_i_use['heading_angle_last1_real'] = straight_trj_i_use.iloc[:, 6] * 191 - 1

            print('straight_trj_i_use:', type(straight_trj_i_use), straight_trj_i_use.shape, straight_trj_i_use)


            straight_i_use = straight_trj_i_use  # 横向拼接
            straight_i_use = straight_i_use.assign(id=straight_i)
            print('straight_i_use:', type(straight_i_use), straight_i_use.shape, straight_i_use)

            # 将索引转换为time列，并删除原始索引
            straight_i_use_reset = straight_i_use.reset_index()

            # 将列名 'index' 更改为 'time'，位于第一列
            straight_i_use_reset = straight_i_use_reset.rename(columns={'index': 'time'})

            straight_scenario_use = pd.concat([straight_scenario_use, straight_i_use_reset], axis=0)  # 纵向拼接
            print('straight_scenario_use:', type(straight_scenario_use), straight_scenario_use.shape, straight_scenario_use)

        straight_scenario_use['direction'] = 'straight'
        straight_scenario_use['width'] = 1.6  # 1.8
        straight_scenario_use['length'] = 4.0  # 4.2

        straight_scenario_use.columns = ['time','x','y','vx','vy','dx','dy','heading_angle_last1','des','vx_last1','vy_last1',
                            'same_deltax_agent_1', 'same_deltay_agent_1', 'same_deltavx_agent_1', 'same_deltavy_agent_1',
                            'left_deltax_agent_1', 'left_deltay_agent_1', 'left_deltavx_agent_1', 'left_deltavy_agent_1',
                            'left_deltax_agent_2', 'left_deltay_agent_2', 'left_deltavx_agent_2', 'left_deltavy_agent_2',
                            'left_deltax_agent_3', 'left_deltay_agent_3', 'left_deltavx_agent_3', 'left_deltavy_agent_3',
                            'right_deltax_agent_1', 'right_deltay_agent_1', 'right_deltavx_agent_1', 'right_deltavy_agent_1',
                            'right_deltax_agent_2', 'right_deltay_agent_2', 'right_deltavx_agent_2', 'right_deltavy_agent_2',
                            'right_deltax_agent_3', 'right_deltay_agent_3', 'right_deltavx_agent_3', 'right_deltavy_agent_3',
                            'left_deltax_landmark_1', 'left_deltay_landmark_1', 'left_deltavx_landmark_1', 'left_deltavy_landmark_1',
                            'right_deltax_landmark_1', 'right_deltay_landmark_1', 'right_deltavx_landmark_1', 'right_deltavy_landmark_1',
                             'same_heading_last', 'left1_heading_last', 'left2_heading_last',
                             'left3_heading_last',
                             'right1_heading_last', 'right2_heading_last',
                             'right3_heading_last',
                             'landmark1_heading_last', 'landmark2_heading_last','min_distance_to_lane', 'last_delta_angle',
                            'angle_now_real', 'heading_angle_last1_real',
                            'id', 'direction', 'width', 'length']  # 九个交互对象的state，10+1*4+3*4+3*4+2*4

        straight_scenario_use['x_real'] = straight_scenario_use['x'] * 38 - 4
        straight_scenario_use['y_real'] = straight_scenario_use['y'] * 23 + 14

        straight_scenario_use['same_agent_1_x_real'] = None
        straight_scenario_use['same_agent_1_y_real'] = None
        straight_scenario_use['left_agent_1_x_real'] = None
        straight_scenario_use['left_agent_1_y_real'] = None
        straight_scenario_use['left_agent_2_x_real'] = None
        straight_scenario_use['left_agent_2_y_real'] = None
        straight_scenario_use['left_agent_3_x_real'] = None
        straight_scenario_use['left_agent_3_y_real'] = None
        straight_scenario_use['right_agent_1_x_real'] = None
        straight_scenario_use['right_agent_1_y_real'] = None
        straight_scenario_use['right_agent_2_x_real'] = None
        straight_scenario_use['right_agent_2_y_real'] = None
        straight_scenario_use['right_agent_3_x_real'] = None
        straight_scenario_use['right_agent_3_y_real'] = None
        straight_scenario_use['left_landmark_1_x_real'] = None
        straight_scenario_use['left_landmark_1_y_real'] = None
        straight_scenario_use['right_landmark_1_x_real'] = None
        straight_scenario_use['right_landmark_1_y_real'] = None

        straight_scenario_use['same_agent_1_x_real'] = np.where(
            straight_scenario_use['same_deltax_agent_1'] == 0, -4,
            straight_scenario_use['x_real'] - straight_scenario_use['same_deltax_agent_1'] * 27 - 12)
        straight_scenario_use['same_agent_1_y_real'] = np.where(
            straight_scenario_use['same_deltay_agent_1'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use['same_deltay_agent_1'] * 18 - 14))

        straight_scenario_use['left_agent_1_x_real'] = np.where(
            straight_scenario_use['left_deltax_agent_1'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'left_deltax_agent_1'] * 30 - 15))
        straight_scenario_use['left_agent_1_y_real'] = np.where(
            straight_scenario_use['left_deltay_agent_1'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'left_deltay_agent_1'] * 14 - 7))
        straight_scenario_use['left_agent_2_x_real'] = np.where(
            straight_scenario_use['left_deltax_agent_2'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'left_deltax_agent_2'] * 30 - 15))
        straight_scenario_use['left_agent_2_y_real'] = np.where(
            straight_scenario_use['left_deltay_agent_2'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'left_deltay_agent_2'] * 14 - 7))
        straight_scenario_use['left_agent_3_x_real'] = np.where(
            straight_scenario_use['left_deltax_agent_3'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'left_deltax_agent_3'] * 5 - 14))
        straight_scenario_use['left_agent_3_y_real'] = np.where(
            straight_scenario_use['left_deltay_agent_3'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'left_deltay_agent_3'] * 4 - 6))

        straight_scenario_use['right_agent_1_x_real'] = np.where(
            straight_scenario_use['right_deltax_agent_1'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'right_deltax_agent_1'] * 30 - 15))
        straight_scenario_use['right_agent_1_y_real'] = np.where(
            straight_scenario_use['right_deltay_agent_1'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'right_deltay_agent_1'] * 24 - 15))
        straight_scenario_use['right_agent_2_x_real'] = np.where(
            straight_scenario_use['right_deltax_agent_2'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'right_deltax_agent_2'] * 30 - 15))
        straight_scenario_use['right_agent_2_y_real'] = np.where(
            straight_scenario_use['right_deltay_agent_2'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'right_deltay_agent_2'] * 24 - 15))
        straight_scenario_use['right_agent_3_x_real'] = np.where(
            straight_scenario_use['right_deltax_agent_3'] == 0, -4,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'right_deltax_agent_3'] * 4 - 15))
        straight_scenario_use['right_agent_3_y_real'] = np.where(
            straight_scenario_use['right_deltay_agent_3'] == 0, 14,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'right_deltay_agent_3'] * 4 + 2))

        straight_scenario_use['left_landmark_1_x_real'] = np.where(
            straight_scenario_use['left_deltax_landmark_1'] == 0, -5,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'left_deltax_landmark_1'] * 29 - 14))
        straight_scenario_use['left_landmark_1_y_real'] = np.where(
            straight_scenario_use['left_deltay_landmark_1'] == 0, -3,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'left_deltay_landmark_1'] * 30 - 15))
        straight_scenario_use['right_landmark_1_x_real'] = np.where(
            straight_scenario_use['right_deltax_landmark_1'] == 0, -5,
            straight_scenario_use['x_real'] - (straight_scenario_use[
                                               'right_deltax_landmark_1'] * 30 - 15))
        straight_scenario_use['right_landmark_1_y_real'] = np.where(
            straight_scenario_use['right_deltay_landmark_1'] == 0, -3,
            straight_scenario_use['y_real'] - (straight_scenario_use[
                                               'right_deltay_landmark_1'] * 29 - 15))

        scenario_data = pd.concat([left_scenario_use,straight_scenario_use], axis=0)
        print('scenario_data:',scenario_data)
        scenario_data_file_path = f'D:/Study/同济大学/博三/面向自动驾驶测试的仿真/sinD_nvn_xuguan/ATT-social-iniobs' \
                                  f'/MA_Intersection_straight/results_evaluate/v13/有注意力的视频/社会倾向' \
                                  f'/专门生成生成轨迹的数据/{scenario_label}/scenario{scenario_label}_data.csv'
        scenario_data.to_csv(scenario_data_file_path)

        time_range = range(scenario_data['time'].min(), min(229, scenario_data['time'].max()+1))

        # 找到这个场景对应的landmarks
        scenario_landmark_trj = pd.DataFrame()
        for landmark_i in range(30): # 最多有30个landmark
            landmark_data_i_array = all_landmark_trj[scenario_label]['ob'][landmark_i][time_range, :]
            landmark_data_i = pd.DataFrame(landmark_data_i_array)
            landmark_data_i.index = time_range
            print('landmark_data_i:',landmark_data_i)
            if len(landmark_data_i[landmark_data_i.iloc[:, 0] != 0]) > 0:
                # 将索引转换为time列，并删除原始索引
                landmark_data_i_use_reset = landmark_data_i.reset_index()
                # 将列名 'index' 更改为 'time'，位于第一列
                landmark_data_i_use_reset = landmark_data_i_use_reset.rename(columns={'index': 'time'})
                landmark_data_i_use_reset['id'] = landmark_i
                scenario_landmark_trj = pd.concat([scenario_landmark_trj, landmark_data_i_use_reset], axis=0)

        scenario_landmark_trj.columns = ['time','x','y','vx','vy','ax','ay','heading_angle_last_1_guiyihua', 'id']

        scenario_landmark_trj['x_real'] = scenario_landmark_trj['x'] * 39 - 5
        scenario_landmark_trj['y_real'] = scenario_landmark_trj['y'] * 38 - 3
        scenario_landmark_trj['angle_last_real'] = scenario_landmark_trj['heading_angle_last_1_guiyihua'] * 360 - 90

        start_time = scenario_data['time'].min()
        end_time = scenario_data['time'].max()
        start_frame_num = 0
        end_frame_num = end_time - start_time
        for frame_id in range(start_frame_num, int(end_frame_num) + 1):  # 遍历一个场景内每一帧
            # 生成图片
            frame_time = start_time + frame_id
            plot_top_view_ani_with_lidar_label(scenario_data, scenario_landmark_trj, scenario_label, frame_time, frame_id)
        # ---------- video generation ----------
        top_view_video_generation(start_frame_num,start_time,scenario_label)  # 生成视频的函数
# save_segment_id_start += save_step
