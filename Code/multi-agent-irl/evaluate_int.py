# -*- coding: utf-8 -*-
"""
Created on Wang Shihan Dec 13:29:31 2023

@author: uqjsun9
"""
import numpy as np
import pandas as pd
import math
from irl.render import makeModel, render, get_dis
import pickle as pkl
import matplotlib.pyplot as plt
from shapely import geometry


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

import glob
import os
import argparse

def args_parser():
    configs = argparse.ArgumentParser(description='')

    configs.add_argument('--path', default="D:/Data/github/SinD-main/Data/",
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



experts = pkl.load(open(r'\Data\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl','rb'))
init_pointss = np.load(r'\Data\init_sinD_nvnxuguan_9jiaohu_social_dayu1_v2.npy', allow_pickle=True)

num_agents = 8
num_left = 3
num_straight = 5

# % rendering
env_id = 'trj_intersection_4'
mids = ['0001'] + [str(a).rjust(4, '0') for a in range(50, 2001, 50)]

root_save = r'Code\results_evaluate\s-90'
training_label = False

for scenario_test in range(2, 30):  # scenario_test_number
    results = []
    sample_trajss = []
    model = makeModel(env_id, scenario_test, training_label)
    for iteration in range(len(mids)):
        mid = mids[iteration]
        path = r'Code\multi-agent-irl\irl\mack\multi-agent-trj\logger\airl\trj_intersection_4\decentralized' \
               r'\s-90\seed-13\m_0' + mid
        sample_trajs = render(path, model, env_id, mid, scenario_test, training_label)
        sample_trajss.append(sample_trajs)

    scenario_test_west = str(scenario_test) + 'west_left'

    folder_path_with_scenario = os.path.join(root_save, 'attention_video\\trj_data\\%s' % str(scenario_test_west))
    os.makedirs(folder_path_with_scenario, exist_ok=True)

    scenario_name = "sample_trajss_" + str(scenario_test_west)

    pkl_file_path = os.path.join(folder_path_with_scenario, f"{scenario_name}.pkl")

    with open(pkl_file_path, 'wb') as f:
        pkl.dump(sample_trajss, f)

    root = os.path.join(root_save, 'trainingset-evaluate_results')
    os.makedirs(root, exist_ok=True)
    root_figure = os.path.join(root, 'trj_compare_figure')
    os.makedirs(root, exist_ok=True)

    actionss = []
    expert = []
    ini_point = []
    expert_ac = []
    sv = []

    for ii in range(scenario_test, scenario_test + 1):
        expert = [experts[ii]['ob'][j][:, :] for j in range(num_agents)]
        ini_point = init_pointss[ii]
        expert_trj_all = pd.DataFrame()
        fig = plt.figure(figsize=(6, 3 / 4 * 6))
        ax = fig.gca()
        for k in range(num_agents):
            if k not in sv:
                trj = pd.concat([pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                                axis=1)

                if ini_point[k][0] != 0:

                    x = expert[k][:, 0] * 38 - 4
                    y = expert[k][:, 1] * 23 + 14
                    non_zero_points = (x != -4) & (y != 14)
                    plt.scatter(x[non_zero_points], y[non_zero_points], s=10, zorder=100, c='none', marker='o', edgecolors='g')
                    plt.scatter(expert[k][0, 0] * 38 - 4, expert[k][0, 1] * 23 + 14, c='k', s=20, zorder=100,
                                marker='o')
                    expert_trj_all = pd.concat([expert_trj_all, trj], axis=0)  # 纵向拼接

        config = args_parser()
        map_path = glob.glob(config['path'] + '/*.osm')[0]
        if use_lanelet2_lib:
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            laneletmap = lanelet2.io.load(map_path, projector)
            map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
        else:
            map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

        os.makedirs(root_figure, exist_ok=True)
        plt.savefig(root_figure + '\%s.png' % ('real_scenario_' + str(ii) + ''), dpi=300)
        plt.close()
        actions = np.squeeze(experts[ii]['ac'])
        actionss.append(actions)

    yaw_rate = [[], []]
    acc_rate = [[], []]

    for ii in range(1):
        for j in range(8):
            for step in range(185):
                if j < 3:
                    acc_rate[0].append(((9.2 * (actionss[ii][j][step][0] + 1)) / 2) - 4.8)
                    yaw_rate[0].append(((2.8 * (actionss[ii][j][step][1] + 1)) / 2) - 0.3)
                elif 3 <= j:
                    acc_rate[1].append(((8.5 * (actionss[ii][j][step][0] + 1)) / 2) - 3.6)
                    yaw_rate[1].append(((2.4 * (actionss[ii][j][step][1] + 1)) / 2) - 1.2)

    sv = []
    generate_expert_trj_all = pd.DataFrame()
    generate_expert_acc_yaw_all = pd.DataFrame()

    for ii in range(41):
        sample_trajs = sample_trajss[ii]
        generate_expert_trj_one_model = pd.DataFrame()
        generate_expert_acc_yaw_one_model = pd.DataFrame()
        for i in range(1):
            traj_data = sample_trajs[i]
            vehicles = [traj_data["ob"][j] for j in range(num_agents)]
            actions = [traj_data["ac"][j] for j in range(num_agents)]
            yaw_rate_gerente = [[], []]
            acc_rate_gerente = [[], []]
            yaw_rate_expert = [[], []]
            acc_rate_expert = [[], []]

            for k in range(num_agents):
                if k < 3:
                    for step in range(230):
                        acc_rate_gerente[0].append(((9.2 * (actions[k][step][0] + 1)) / 2) - 4.8)
                        yaw_rate_gerente[0].append(((2.8 * (actions[k][step][1] + 1)) / 2) - 0.3)

                    for step in range(185):
                        acc_rate_expert[0].append(((9.2 * (actionss[i][k][step][0] + 1)) / 2) - 4.8)
                        yaw_rate_expert[0].append(((2.8 * (actionss[i][k][step][1] + 1)) / 2) - 0.3)

                if k not in sv and k < 3:
                    fig = plt.figure(figsize=(6, 3 / 4 * 6))

                    x_g = vehicles[k][:, 0] * 38 - 4
                    y_g = vehicles[k][:, 1] * 23 + 14

                    non_zero_points_g = (x_g != -4) & (y_g != 14)

                    x_e = expert[k][:, 0] * 38 - 4
                    y_e = expert[k][:, 1] * 23 + 14

                    non_zero_points_e = (x_e != -4) & (y_e != 14)

                    plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g], s=10, zorder=100, c='none', marker='o',
                                edgecolors='b')
                    plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                                marker='o', edgecolors='k')
                    plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, zorder=99, c='none', marker='o',
                                edgecolors='g')

                    config = args_parser()

                    map_path = glob.glob(config['path'] + '/*.osm')[0]
                    if use_lanelet2_lib:
                        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                        laneletmap = lanelet2.io.load(map_path, projector)
                        map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
                    else:
                        map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

                    plt.savefig(root_figure + '\%s.png' % (
                                'scenario_' + scenario_test_west + 'model_' + str(ii) + 'left_vehicle_' + str(k)), dpi=300)
                    plt.close()


                    generate_trj_left = pd.concat(
                        [pd.DataFrame(vehicles[k][:, 0] * 38 - 4), pd.DataFrame(vehicles[k][:, 1] * 23 + 14)],
                        axis=1)

                    generate_action_left = pd.concat(
                        [pd.DataFrame(acc_rate_gerente[0][0+k*230:230+k*230]), pd.DataFrame(yaw_rate_gerente[0][0+k*230:230+k*230])], axis=1)

                    generate_angle_left = pd.DataFrame(vehicles[k][:, 6] * 191 - 1)
                    generate_v_left = pd.DataFrame(np.sqrt((vehicles[k][:, 2].astype(float) * 21 - 14)**2 + (vehicles[k][:, 3].astype(float) * 12 - 2)**2))

                    expert_trj_left = pd.concat(
                        [pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                        axis=1)

                    expert_action_left = pd.concat(
                        [pd.DataFrame(acc_rate_expert[0][0+k*185:185+k*185]), pd.DataFrame(yaw_rate_expert[0][0+k*185:185+k*185])], axis=1)

                    expert_angle_left = pd.DataFrame(expert[k][:, 6] * 191 - 1)
                    expert_v_left = pd.DataFrame(np.sqrt((expert[k][:, 2] * 21 - 14) ** 2 + (expert[k][:, 3] * 12 - 2) ** 2))

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
                    generate_expert_trj_one_model = pd.concat([generate_expert_trj_one_model, generate_expert_trj_left],axis=0)


            for k in range(num_agents):
                if 3 <= k:
                    for step in range(230):
                        acc_rate_gerente[1].append(((8.5 * (actions[k][step][0] + 1)) / 2) - 3.6)
                        yaw_rate_gerente[1].append(((2.4 * (actions[k][step][1] + 1)) / 2) - 1.2)

                    for step in range(185):
                        acc_rate_expert[1].append(((8.5 * (actionss[i][k][step][0] + 1)) / 2) - 3.6)
                        yaw_rate_expert[1].append(((2.4 * (actionss[i][k][step][1] + 1)) / 2) - 1.2)

                if k not in sv and k >= 3:
                    fig = plt.figure(figsize=(6, 3 / 4 * 6))

                    x_g = vehicles[k][:, 0] * 38 - 4
                    y_g = vehicles[k][:, 1] * 23 + 14

                    non_zero_points_g = (x_g != -4) & (y_g != 14)

                    x_e = expert[k][:, 0] * 38 - 4
                    y_e = expert[k][:, 1] * 23 + 14

                    non_zero_points_e = (x_e != -4) & (y_e != 14)

                    plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g],
                                s=10, zorder=100, c='none', marker='o', edgecolors='b')
                    plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14,
                                s=20, zorder=100, c='none', marker='o', edgecolors='k')
                    plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, zorder=99, c='none', marker='o',
                                edgecolors='g')

                    config = args_parser()

                    map_path = glob.glob(config['path'] + '/*.osm')[0]
                    if use_lanelet2_lib:
                        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                        laneletmap = lanelet2.io.load(map_path, projector)
                        map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
                    else:
                        map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

                    plt.savefig(root_figure + '\%s.png' % (
                                'scenario_' + scenario_test_west + 'model_' + str(ii) + 'straight_vehicle_' + str(k)),
                                dpi=300)
                    plt.close()


                    generate_trj_straight = pd.concat(
                        [pd.DataFrame(vehicles[k][:, 0] * 38 - 4), pd.DataFrame(vehicles[k][:, 1] * 23 + 14)],
                        axis=1)

                    generate_action_straight = pd.concat(
                        [pd.DataFrame(acc_rate_gerente[1][0+(k-3)*230:230+(k-3)*230]), pd.DataFrame(yaw_rate_gerente[1][0+(k-3)*230:230+(k-3)*230])], axis=1)

                    generate_angle_straight = pd.DataFrame(vehicles[k][:, 6] * 191 - 1)
                    generate_v_straight = pd.DataFrame(np.sqrt((vehicles[k][:, 2].astype(float) * 21 - 14) ** 2 + (vehicles[k][:, 3].astype(float) * 12 - 2) ** 2))

                    expert_trj_straight = pd.concat(
                        [pd.DataFrame(expert[k][:, 0] * 38 - 4), pd.DataFrame(expert[k][:, 1] * 23 + 14)],
                        axis=1)

                    expert_action_straight = pd.concat(
                        [pd.DataFrame(acc_rate_expert[1][0+(k-3)*185:185+(k-3)*185]), pd.DataFrame(yaw_rate_expert[1][0+(k-3)*185:185+(k-3)*185])], axis=1)

                    expert_angle_straight = pd.DataFrame(expert[k][:, 6] * 191 - 1)
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

                    generate_expert_trj_one_model = pd.concat([generate_expert_trj_one_model, generate_expert_trj_straight], axis=0)

            fig_all = plt.figure(figsize=(6, 3 / 4 * 6))
            for k in range(num_agents):
                if k < 3:
                    if ini_point[k][0] != 0:
                        x_g = vehicles[k][:, 0] * 38 - 4
                        y_g = vehicles[k][:, 1] * 23 + 14

                        non_zero_points_g = (x_g != -4) & (y_g != 14)

                        x_e = expert[k][:, 0] * 38 - 4
                        y_e = expert[k][:, 1] * 23 + 14

                        non_zero_points_e = (x_e != -4) & (y_e != 14)

                        plt.scatter(x_g[non_zero_points_g], y_g[non_zero_points_g], s=10, zorder=100, c='none', marker='o',
                                    edgecolors='b')

                        plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                                    marker='o', edgecolors='k')

                        plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, alpha=0.5, zorder=99, c='none',
                                    marker='o', edgecolors='g')

            config = args_parser()

            map_path = glob.glob(config['path'] + '/*.osm')[0]
            if use_lanelet2_lib:
                projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                laneletmap = lanelet2.io.load(map_path, projector)
                map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
            else:
                map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)
            plt.savefig(root_figure + '\%s.png' % (
                        'scenario_' + scenario_test_west + 'model_' + str(ii) + '_left_vehicle'), dpi=300)
            plt.close()

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
                                    edgecolors='b')

                        plt.scatter(vehicles[k][0][0] * 38 - 4, vehicles[k][0][1] * 23 + 14, s=20, zorder=100, c='none',
                                    marker='o', edgecolors='k')

                        plt.scatter(x_e[non_zero_points_e], y_e[non_zero_points_e], s=30, alpha=0.5, zorder=99, c='none',
                                    marker='o', edgecolors='g')

            config = args_parser()

            map_path = glob.glob(config['path'] + '/*.osm')[0]
            if use_lanelet2_lib:
                projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
                laneletmap = lanelet2.io.load(map_path, projector)
                map_vis_lanelet2.draw_lanelet_map(laneletmap, plt.gca())
            else:
                map_vis_without_lanelet.draw_map_without_lanelet(map_path, plt.gca(), 0, 0)

            plt.savefig(root_figure + '\%s.png' % (
                        'scenario_' + scenario_test_west + 'model_' + str(ii) + '_straight_vehicle'), dpi=300)

            generate_acc_left_i = pd.DataFrame(acc_rate_gerente[0])
            generate_yaw_left_i = pd.DataFrame(yaw_rate_gerente[0])
            generate_acc_delta_angle = pd.concat([generate_acc_left_i, generate_yaw_left_i], axis=1)

            generate_acc_straight_i = pd.DataFrame(acc_rate_gerente[1])
            generate_yaw_straight_i = pd.DataFrame(yaw_rate_gerente[1])
            generate_acc_delta_angle = pd.concat([generate_acc_delta_angle, generate_acc_straight_i],axis=1)
            generate_acc_delta_angle = pd.concat([generate_acc_delta_angle, generate_yaw_straight_i],axis=1)

            expert_acc_left_i = pd.DataFrame(acc_rate[0])
            expert_yaw_left_i = pd.DataFrame(yaw_rate[0])
            expert_acc_delta_angle = pd.concat([expert_acc_left_i, expert_yaw_left_i], axis=1)
            expert_acc_straight_i = pd.DataFrame(acc_rate[1])
            expert_yaw_straight_i = pd.DataFrame(yaw_rate[1])
            expert_acc_delta_angle = pd.concat([expert_acc_delta_angle, expert_acc_straight_i], axis=1)
            expert_acc_delta_angle = pd.concat([expert_acc_delta_angle, expert_yaw_straight_i], axis=1)

            generate_expert_acc_yaw_one_model = pd.concat([generate_acc_delta_angle, expert_acc_delta_angle],axis=1)
            generate_expert_acc_yaw_one_model['model_id'] = ii

            root_action_figure = os.path.join(root, 'one_scenario_action_distribution_figure')
            os.makedirs(root_action_figure, exist_ok=True)
            fig2 = plt.figure(figsize=(6, 3 / 4 * 6))
            plt.hist(yaw_rate[0], alpha=0.5)
            plt.hist(yaw_rate_gerente[0], alpha=0.5)
            plt.savefig(root_action_figure + '\%s.png' % (scenario_test_west + '_左转车_yaw_rate_' + 'tess_model_' + str(ii)))

            fig3 = plt.figure(figsize=(6, 3 / 4 * 6))
            plt.hist(yaw_rate[1], alpha=0.5)
            plt.hist(yaw_rate_gerente[1], alpha=0.5)
            plt.savefig(root_action_figure + '\%s.png' % (scenario_test_west + '_直行车_yaw_rate_' + 'tess_model_' + str(ii)))

            fig4 = plt.figure(figsize=(6, 3 / 4 * 6))
            plt.hist(acc_rate[0], alpha=0.5)
            plt.hist(acc_rate_gerente[0], alpha=0.5)
            plt.savefig(root_action_figure + '\%s.png' % (scenario_test_west + '_左转车_acc_' + 'tess_model_' + str(ii)))

            fig5 = plt.figure(figsize=(6, 3 / 4 * 6))
            plt.hist(acc_rate[1], alpha=0.5)
            plt.hist(acc_rate_gerente[1], alpha=0.5)
            plt.savefig(root_action_figure + '\%s.png' % (scenario_test_west + '直行车_acc_' + 'tess_model_' + str(ii)))

        generate_expert_trj_all = pd.concat([generate_expert_trj_all, generate_expert_trj_one_model], axis=0)
        generate_expert_acc_yaw_all = pd.concat([generate_expert_acc_yaw_all, generate_expert_acc_yaw_one_model], axis=0)

    generate_expert_trj_all.columns = ['generate_x', 'generate_y', 'expert_x', 'expert_y', 'generate_acc', 'generate_yaw','generate_angle', 'expert_acc', 'expert_yaw', 'expert_angle','generate_v','expert_v','agent_id', 'model_id', 'direction']
    generate_expert_acc_yaw_all.columns = ['generate_acc_left', 'generate_yaw_left', 'generate_acc_straight','generate_yaw_straight', 'expert_acc_left', 'expert_yaw_left','expert_acc_straight', 'expert_yaw_straight', 'model_id']

    generate_expert_trj_all['length'] = 4.6
    generate_expert_trj_all['width'] = 1.8
    generate_expert_acc_yaw_all['length'] = 4.6
    generate_expert_acc_yaw_all['width'] = 1.8

    generate_expert_trj_all['generate_angle_now'] = generate_expert_trj_all['generate_yaw'] + generate_expert_trj_all['generate_angle']
    generate_expert_trj_all['expert_angle_now'] = generate_expert_trj_all['expert_yaw'] + generate_expert_trj_all['expert_angle']

    root_KL = os.path.join(root, 'DATA_usefor_js')
    os.makedirs(root_KL, exist_ok=True)
    root_accyaw = os.path.join(root, 'acc_yaw')
    os.makedirs(root_accyaw, exist_ok=True)

    generate_expert_trj_all.to_csv(f"{root_KL}/{scenario_test_west}_expert_generate_trj.csv")
    generate_expert_acc_yaw_all.to_csv(f"{root_accyaw}/{scenario_test_west}_expert_generate_acc_yaw.csv")

    # calculate rmse
    average_pos_rmse = [[], []]

    model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]
    for one_model in model_datass:
        left_trj_model = one_model[one_model['direction'] == 'left']
        straight_trj_model = one_model[one_model['direction'] == 'straight']
        a = 0
        trj_pos_rmse_left = float(0)
        trj_pos_rmse_left_2 = float(0)
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
                rmse_left_1.append(trj_pos_rmse_left)

        rmse_left_1_array = np.array(rmse_left_1)
        average_pos_rmse[0].append(np.mean(rmse_left_1_array[rmse_left_1_array >= 0]))

        b = 0
        trj_pos_rmse_straight = float(0)
        trj_pos_rmse_straight_2 = float(0)
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

                rmse_straight_1.append(trj_pos_rmse_straight)

        rmse_straight_1_array = np.array(rmse_straight_1)
        average_pos_rmse[1].append(np.mean(rmse_straight_1_array[rmse_straight_1_array >= 0]))

    average_pos_rmse_df = pd.DataFrame(average_pos_rmse)

    average_pos_rmse_df.columns = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5',
                                   'model_6', 'model_7', 'model_8', 'model_9', 'model_10', 'model_11',
                                   'model_12', 'model_13', 'model_14', 'model_15', 'model_16', 'model_17',
                                   'model_18', 'model_19', 'model_20', 'model_21', 'model_22', 'model_23',
                                   'model_24', 'model_25', 'model_26', 'model_27', 'model_28', 'model_29',
                                   'model_30', 'model_31', 'model_32', 'model_33', 'model_34', 'model_35',
                                   'model_36', 'model_37', 'model_38', 'model_39', 'model_40']

    root_pos_rmse = os.path.join(root, 'trj_pos_rmse')
    os.makedirs(root_pos_rmse, exist_ok=True)
    average_pos_rmse_df.to_csv(f"{root_pos_rmse}/{scenario_test_west}_expert_generate_pos_rmse.csv")

    average_acc_yaw = [[], [], [], []]
    model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]
    for one_model in model_datass:
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
    new_index = ['acc_left_rmse', 'yaw_left_rmse', 'acc_straight_rmse', 'yaw_straight_rmse']

    average_acc_yaw_df = average_acc_yaw_df.set_index(pd.Index(new_index))
    average_acc_yaw_df.columns = ['model_0', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5',
                                   'model_6', 'model_7', 'model_8', 'model_9', 'model_10', 'model_11',
                                   'model_12', 'model_13', 'model_14', 'model_15', 'model_16', 'model_17',
                                   'model_18', 'model_19', 'model_20', 'model_21', 'model_22', 'model_23',
                                   'model_24', 'model_25', 'model_26', 'model_27', 'model_28', 'model_29',
                                   'model_30', 'model_31', 'model_32', 'model_33', 'model_34', 'model_35',
                                   'model_36', 'model_37', 'model_38', 'model_39', 'model_40']

    root_action_rmse = os.path.join(root, 'action_rmse')
    os.makedirs(root_action_rmse, exist_ok=True)
    average_acc_yaw_df.to_csv(f"{root_action_rmse}/{scenario_test_west}_expert_generate_action_rmse.csv")

    def calculate_rectangle_vertices(center_x, center_y, length, width, angle):
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

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
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

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

            if o1 != o2 and o3 != o4:
                return True

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
                  Jiaohu_x, Jiaohu_y, Jiaohu_v, Jiaohu_angle_last):

        dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
        if dis_between_agent_jiaohu <= 15:
            agent_v = Agent_v
            neig_v = Jiaohu_v
            veh_length = 5
            veh_width = 2

            a_agent = math.tan(np.radians(Agent_angle_last))
            a_neig = math.tan(np.radians(Jiaohu_angle_last))

            b_agent = (Agent_y) - a_agent * (Agent_x)
            b_neig = (Jiaohu_y) - a_neig * (Jiaohu_x)

            vertices_agent_now = calculate_rectangle_vertices(Agent_x, Agent_y, veh_length, veh_width,
                                                              Agent_angle_last)
            vertices_neig_now = calculate_rectangle_vertices(Jiaohu_x, Jiaohu_y, veh_length, veh_width,
                                                             Jiaohu_angle_last)

            intersect_now = check_intersection(vertices_agent_now, vertices_neig_now)
            if intersect_now == True:
                GT_value = 0
            else:
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
                elif Agent_angle_last == 270:
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
                    if a_neig == a_agent:
                        GT_value = None
                    else:
                        jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)
                        jiaodiany = a_neig * jiaodianx + b_neig

                        agent_b = np.zeros(2)
                        if 0 <= Agent_angle_last < 90:  # tan>0
                            agent_b = np.array([1, math.tan(math.radians(Agent_angle_last))])
                        elif Agent_angle_last == 90:
                            agent_b = np.array([0, 2])
                        elif 90 < Agent_angle_last <= 180:  # tan<0
                            agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                        elif 180 < Agent_angle_last < 270:  # tan>0
                            agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                        elif Agent_angle_last == 270:
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
                        elif Jiaohu_angle_last == 270:
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

                            if (agent_first_dis / agent_v) < (neig_last_dis / neig_v):  # agent first
                                GT_value = abs(neig_last_dis / neig_v - agent_first_dis / agent_v)
                            elif (neig_first_dis / neig_v) < (agent_last_dis / agent_v):  # neig first
                                GT_value = abs(agent_last_dis / agent_v - neig_first_dis / neig_v)
                            else:
                                GT_value = min(abs(agent_last_dis / agent_v - neig_first_dis / neig_v),
                                               abs(neig_last_dis / neig_v - agent_first_dis / agent_v))

                        elif dot_product_agent >= 0 and dot_product_neig < 0:
                            GT_value = None

                        else:
                            GT_value = None
                else:
                    GT_value = None
        else:
            GT_value = None
        return GT_value

    def Cal_GT(time_trj,neig_trj,type):
        time_trj.index = range(len(time_trj))
        neig_trj.index = range(len(neig_trj))
        neig_x = neig_trj[type+'_x'][0]
        neig_y = neig_trj[type+'_y'][0]
        agent_x = time_trj[type + '_x'][0]
        agent_y = time_trj[type + '_y'][0]
        neig_heading_now = neig_trj[type + '_angle'][0]
        agent_heading_now = time_trj[type + '_angle'][0]
        neig_v = neig_trj[type + '_v'][0]
        agent_v = time_trj[type + '_v'][0]

        veh_length = time_trj['length'][0]
        veh_width = time_trj['width'][0]

        a_neig = math.tan(np.radians(neig_heading_now))
        a_agent = math.tan(np.radians(agent_heading_now))

        b_neig = (neig_y) - a_neig * (neig_x)
        b_agent = (agent_y) - a_agent * (agent_x)

        GT_value = None
        if a_neig == a_agent:
            GT_value = None
        else:
            jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)
            jiaodiany = a_neig * jiaodianx + b_neig

            agent_b = np.zeros(2)
            if 0 <= agent_heading_now < 90:  # tan>0
                agent_b = np.array([1, math.tan(math.radians(agent_heading_now))])
            elif agent_heading_now == 90:
                agent_b = np.array([0, 2])
            elif 90 < agent_heading_now <= 180:  # tan<0
                agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
            elif 180 < agent_heading_now < 270:  # tan>0
                agent_b = np.array([-1, -1 * math.tan(math.radians(agent_heading_now))])
            elif agent_heading_now == 270:
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
            elif neig_heading_now == 270:
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

    generate_expert_trj_all['GT_AVE'] = None

    model_datass = [name[1] for name in generate_expert_trj_all.groupby(['model_id'])]
    data_all_model_generate = pd.DataFrame()

    # generate
    for one_model in model_datass:
        trj_datass = [name[1] for name in one_model.groupby(['agent_id'])]
        for trj0 in trj_datass:
            trj0['time_ms'] = range(0, len(trj0) * 100, 100)
            trj0.index = range(len(trj0))
            trj = trj0[trj0['generate_x']!=-4]
            trj_max_time = trj['time_ms'].max()

            if len(trj)!=0:
                trj_id = trj['agent_id'][trj['time_ms'].idxmin()]
                if trj_id <= 2:
                    agent_direction = 'left'
                else:
                    agent_direction = 'straight'
                other_potential_trjs = one_model[one_model['agent_id']!=trj_id]
                other_trjs = [name[1] for name in other_potential_trjs.groupby(['agent_id'])]
                other_trjs_new_df = pd.DataFrame()
                for other_trj_id, other_trj in enumerate(other_trjs):
                    other_trj.index = range(len(other_trj))
                    other_trj['time_ms'] = range(0, len(other_trj) * 100, 100)
                    other_trj_use = other_trj[other_trj['generate_x']!=-4]
                    other_trj_use.sort_values(by='time_ms', inplace=True)
                    if len(other_trj_use)!=0:
                        other_trjs_new_df = pd.concat([other_trjs_new_df, other_trj_use], axis=0)
                    other_trj_new = other_trj[other_trj['time_ms'] <= trj_max_time]

                for i, row in trj.iterrows():
                    use_GT = []
                    agent_x = trj.loc[i,'generate_x']
                    agent_y = trj.loc[i,'generate_y']
                    agent_v = trj.loc[i,'generate_v']
                    angle = trj.loc[i,'generate_angle']
                    agent_angle_last = angle
                    time_agent = trj.loc[i,'time_ms']

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

                            use_GT.append(jiaohu_agent_GT_value)

                        use_GT_list_0 = [x for x in use_GT if x is not None]
                        use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                        rew_minGT_mapped = 0
                        if len(use_GT_list) != 0:
                            rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        else:
                            rew_aveGT = None

                        trj['GT_AVE'][i] = rew_aveGT

                data_all_model_generate = pd.concat([data_all_model_generate,trj],axis=0)

    data_all_model_generate['type'] = 'generate'

    data_all_model_expert = pd.DataFrame()

    for one_model in model_datass:
        trj_datass = [name[1] for name in one_model.groupby(['agent_id'])]
        for trj0 in trj_datass:
            trj0.index = range(len(trj0))
            trj0['time_ms'] = range(0, len(trj0) * 100, 100)
            trj = trj0[(trj0['expert_x'] != -4) & (trj0['expert_x'].notna())]

            trj_max_time = trj['time_ms'].max()
            if len(trj) != 0:
                trj_id = trj['agent_id'][trj['time_ms'].idxmin()]
                if trj_id <= 2:
                    agent_direction = 'left'
                else:
                    agent_direction = 'straight'
                other_potential_trjs = one_model[one_model['agent_id']!=trj_id]
                other_trjs = [name[1] for name in other_potential_trjs.groupby(['agent_id'])]
                other_trjs_new_df = pd.DataFrame()
                for other_trj_id, other_trj in enumerate(other_trjs):
                    other_trj.index = range(len(other_trj))
                    other_trj['time_ms'] = range(0, len(other_trj) * 100, 100)
                    other_trj_use = other_trj[other_trj['expert_x'] != -4]
                    other_trj_use.sort_values(by='time_ms', inplace=True)
                    if len(other_trj_use) != 0:
                        other_trjs_new_df = pd.concat([other_trjs_new_df, other_trj_use], axis=0)
                    other_trj_new = other_trj[other_trj['time_ms'] <= trj_max_time]

                for i in range(trj['time_ms'].idxmin(), trj['time_ms'].idxmax() + 1):
                    use_GT = []
                    agent_x = trj.loc[i, 'expert_x']
                    agent_y = trj.loc[i, 'expert_y']
                    agent_v = trj.loc[i, 'expert_v']
                    angle = trj.loc[i, 'expert_angle']
                    agent_angle_last = angle
                    time_agent = trj.loc[i, 'time_ms']

                    potential_interaction_data = other_trjs_new_df[abs(other_trjs_new_df['time_ms'] - time_agent) < 100]

                    potential_interaction_data.index = range(len(potential_interaction_data))

                    if len(potential_interaction_data) > 0:
                        for jj in range(len(potential_interaction_data)):
                            jiaohu_agent_x = potential_interaction_data['expert_x'][jj]
                            jiaohu_agent_y = potential_interaction_data['expert_y'][jj]
                            jiaohu_agent_v = potential_interaction_data['expert_v'][jj]
                            jiaohu_agent_angle_last = potential_interaction_data['expert_angle'][jj]

                            jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_v,
                                                              agent_angle_last, agent_direction,
                                                              jiaohu_agent_x, jiaohu_agent_y,
                                                              jiaohu_agent_v,
                                                              jiaohu_agent_angle_last)
                            use_GT.append(jiaohu_agent_GT_value)

                        use_GT_list_0 = [x for x in use_GT if x is not None]
                        use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                        rew_minGT_mapped = 0
                        if len(use_GT_list) != 0:
                            rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        else:
                            rew_aveGT = None

                        trj['GT_AVE'][i] = rew_aveGT

                data_all_model_expert = pd.concat([data_all_model_expert,trj],axis=0)

    data_all_model_expert['type'] = 'expert'

    data_all_model = pd.concat([data_all_model_generate, data_all_model_expert],axis=0)

    root_desried_acc = os.path.join(root, 'expert_generate_desried_acc')
    data_all_model.to_csv(f"{root_desried_acc}/{scenario_test_west}_expert_generate_gt.csv")



