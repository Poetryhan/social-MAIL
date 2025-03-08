import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from shapely import geometry
import math

import tensorflow as tf
import numpy as np
from shapely import geometry
from shapely.geometry import MultiPoint, MultiPolygon, box, Polygon
from shapely.geometry import Point
import pandas as pd
import pickle


n1 = list(range(0, 2))
n2 = list(range(2, 6))

class Scenario(BaseScenario):
    def __init__(self):
        self.scene_counter = 0

    def make_world(self, scenario_test, training_label):
        world = World()
        # set any world properties first
        world.dim_p = 2
        world.dim_c = 0

        num_left = 3
        num_straight = 5
        # num_ud = 4
        # num_du = 4
        num_agents = num_left + num_straight

        world.num_left = num_left
        world.num_straight = num_straight
        # world.num_ud = num_ud
        # world.num_du = num_du
        world.num_agents = num_agents

        num_landmarks = 0
        world.agents = [Agent() for i in range(num_agents)]

        for i in range(num_agents):
            world.agents[i].name = 'agent %d' % i
            world.agents[i].collide = False
            world.agents[i].silent = True
            world.agents[i].adversary = False

        self.reset_world(world, scenario_test, training_label)
        return world

    def reset_world(self, world, scenario_test, training_label):
        init_pointss = np.load(r'\Data\init_sinD.npy', allow_pickle=True)
        path_expert = r'\Data\sinD.pkl'  # path='/root/……/aus_openface.pkl'   pkl
        path_landmark = r'\Data\landmarks.pkl'

        f_expert = open(path_expert, 'rb')
        all_expert_trj = pickle.load(f_expert)
        f_landmark = open(path_landmark, 'rb')
        all_landmark_trj = pickle.load(f_landmark)

        if training_label:  # training
            # choices = [x for x in range(0, 91)]
            choices = [x for x in range(0, 49)] + [x for x in range(50, 69)] + [x for x in range(70, 91)]
            aa = np.random.choice(choices)
        else:  # testing
            aa = np.random.randint(scenario_test, scenario_test + 1)

        self.aa = aa

        init_points = init_pointss[aa]
        ini_steps = init_points[:, 57:58]
        world.landmarks = np.array(all_landmark_trj[aa]['ob'])

        for i in range(world.num_agents):
            world.agents[i].state.p_pos = np.zeros(2)
            world.agents[i].state.p_vel = np.zeros(2)

            world.agents[i].state.p_des = np.zeros(2)
            world.agents[i].state.p_dis = 0

            world.agents[i].state.p_ini_to_end_dis = 0
            world.agents[i].state.p_last_vx = 0
            world.agents[i].state.p_last_vy = 0
            world.agents[i].state.delta_angle_now = 0
            world.agents[i].state.delta_angle_last1 = 0
            world.agents[i].state.delta_angle_last2 = 0
            world.agents[i].state.heading_angle_last1 = 0
            world.agents[i].state.heading_angle_last2 = 0
            world.agents[i].state.acc_x = None
            world.agents[i].state.acc_y = None
            world.agents[i].state.delta_accx = 0
            world.agents[i].state.delta_accy = 0

            world.agents[i].collide = False
            world.agents[i].end_label = False
            world.agents[i].state.des_rew = 0
            world.agents[i].state.lane_rew = 0
            world.agents[i].state.heading_angle_rew = 0
            world.agents[i].state.delta_angle_rew = 0
            world.agents[i].state.heading_std_rew = 0
            world.agents[i].state.scenario_id = aa
            world.agents[i].state.reference_line = None
            world.agents[i].state.step = 0
            world.agents[i].state.min_distance = 0
            world.agents[i].state.delta_angle_last1 = 0

            world.agents[i].size = 2
            world.agents[i].id = i

        for i in range(world.num_agents):
            if init_points[i][57] == 0:
                if init_points[i][0] == 0 and init_points[i][1] == 0 and init_points[i][2] == 0 and init_points[i][3] == 0:
                    world.agents[i].state.p_pos = init_points[i][:2]
                    world.agents[i].state.p_vel = init_points[i][2:4]
                    world.agents[i].state.heading_angle_last1 = init_points[i][6]
                    world.agents[i].state.p_dis = init_points[i][7]
                    world.agents[i].state.p_ini_to_end_dis = init_points[i][7]
                    world.agents[i].state.p_last_vx = init_points[i][8]
                    world.agents[i].state.p_last_vy = init_points[i][9]
                    world.agents[i].collide = False
                    world.agents[i].end_label = True
                    world.agents[i].state.des_rew = 0
                    world.agents[i].state.lane_rew = 0
                    world.agents[i].state.reference_line = np.array(all_expert_trj[aa]['ob'][i][:, :2])
                    world.agents[i].state.step = 0
                    world.agents[i].state.min_distance = 0
                    world.agents[i].state.delta_angle_last1 = init_points[i][56]

                    p_pos_x = (world.agents[i].state.p_pos[0] * 38) - 4
                    p_pos_y = (world.agents[i].state.p_pos[1] * 23) + 14

                    dx = (init_points[i][4] * 59) - 37
                    dy = (init_points[i][5] * 27) - 4

                    des_x = dx + p_pos_x
                    des_y = dy + p_pos_y

                    world.agents[i].state.p_des[0] = (des_x + 4) / 38
                    world.agents[i].state.p_des[1] = (des_y - 14) / 23
                else:
                    world.agents[i].state.p_pos = init_points[i][:2]
                    world.agents[i].state.p_vel = init_points[i][2:4]
                    world.agents[i].state.heading_angle_last1 = init_points[i][6]
                    world.agents[i].state.p_dis = init_points[i][7]
                    world.agents[i].state.p_ini_to_end_dis = init_points[i][7]
                    world.agents[i].state.p_last_vx = init_points[i][8]
                    world.agents[i].state.p_last_vy = init_points[i][9]
                    world.agents[i].collide = False
                    world.agents[i].end_label = False
                    world.agents[i].state.des_rew = 0
                    world.agents[i].state.lane_rew = 0
                    world.agents[i].state.reference_line = np.array(all_expert_trj[aa]['ob'][i][:, :2])
                    world.agents[i].state.step = 0
                    world.agents[i].state.min_distance = 0
                    world.agents[i].state.delta_angle_last1 = init_points[i][56]

                    p_pos_x = (world.agents[i].state.p_pos[0] * 38) - 4
                    p_pos_y = (world.agents[i].state.p_pos[1] * 23) + 14

                    dx = (init_points[i][4] * 59) - 37
                    dy = (init_points[i][5] * 27) - 4

                    des_x = dx + p_pos_x
                    des_y = dy + p_pos_y

                    world.agents[i].state.p_des[0] = (des_x + 4) / 38
                    world.agents[i].state.p_des[1] = (des_y - 14) / 23
            else:
                world.agents[i].state.p_pos = np.zeros(2)
                world.agents[i].state.p_vel = np.zeros(2)

                world.agents[i].state.p_des = np.zeros(2)
                world.agents[i].state.p_dis = 0

                world.agents[i].state.p_ini_to_end_dis = 0
                world.agents[i].state.p_last_vx = 0
                world.agents[i].state.p_last_vy = 0
                world.agents[i].state.delta_angle_now = 0
                world.agents[i].state.delta_angle_last1 = 0
                world.agents[i].state.delta_angle_last2 = 0
                world.agents[i].state.heading_angle_last1 = 0
                world.agents[i].state.heading_angle_last2 = 0
                world.agents[i].state.acc_x = None
                world.agents[i].state.acc_y = None
                world.agents[i].state.delta_accx = 0
                world.agents[i].state.delta_accy = 0

                world.agents[i].collide = False
                world.agents[i].end_label = False
                world.agents[i].state.des_rew = 0
                world.agents[i].state.lane_rew = 0
                world.agents[i].state.heading_angle_rew = 0
                world.agents[i].state.delta_angle_rew = 0
                world.agents[i].state.heading_std_rew = 0
                world.agents[i].state.scenario_id = aa
                world.agents[i].state.reference_line = None
                world.agents[i].state.step = 0
                world.agents[i].state.min_distance = 0
                world.agents[i].state.delta_angle_last1 = 0
                world.agents[i].size = 2
                world.agents[i].id = i
        ini_obs = init_points[:, :57]
        ini_obs_lstm = np.zeros((8, 21, 57))
        for i in range(8):
            ini_obs_lstm[i, :20, :] = 0
            ini_obs_lstm[i, 20, :] = init_points[i, :57]
        return ini_obs, ini_steps, ini_obs_lstm

    def reward(self, agent, obs_use_for_reward, world):
        rew = 0
        return rew

    def observation(self, agent, action_n_agent, reset_infos, world):
        trj_go_step = action_n_agent[3]
        if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            t = world.time
            if reset_infos == [True]:
                if action_n_agent[2] == action_n_agent[4]:
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]
                else:
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]
            else:
                if action_n_agent[2] >= action_n_agent[4]:
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2])+1,:]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]
                else:
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0]!=0)]

            veh_same = []
            veh_left_agents = []
            veh_right_agents = []
            veh_left_landmark = []
            veh_right_landmark = []

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4))**2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14))**2) for ve in
                                        vehs_agent])

            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5))**2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3))**2)
                                           for ve_landmark in
                                           landmarks_veh_use])

            for ii in range(len(vehs_agent)):
                if vehs_agent[ii][0] > 0:
                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])
                    if agent.state.heading_angle_last1 * 191 - 1 < -90:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1
                    b = np.zeros(2)
                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:
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
                    cross = np.cross((b[0], b[1]), ((vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                                    (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if vehs_agent[ii][5] <= 2:
                        jiaohu_agent_direction = 'left'
                    else:
                        jiaohu_agent_direction = 'straight'

                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []
            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))

            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    for i in range(3):
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    top_3_veh = sorted_veh[:3, :]
                    veh_new = top_3_veh.copy()
                    for i in range(3):
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            for iii in range(len(landmarks_veh_use)):
                if landmarks_veh_use[iii][0] > 0:

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:
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
                    cross = np.cross((b[0], b[1]), ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                                    (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []

            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            obs = np.array([((agent_x - (veh_neig[0][0] * 38 - 4)) + 12) / 27 if veh_neig[0][0] != 0 else 0,
                            ((agent_y - (veh_neig[0][1] * 23 + 14)) + 14) / 18 if veh_neig[0][1] != 0 else 0,
                            ((agent_vx - (veh_neig[0][2] * 21 - 14)) + 7) / 14 if veh_neig[0][2] != 0 else 0,
                            ((agent_vy - (veh_neig[0][3] * 12 - 2)) + 5) / 6 if veh_neig[0][3] != 0 else 0,

                            ((agent_x - (veh_neig[1][0] * 38 - 4)) + 15) / 30 if veh_neig[1][0] != 0 else 0,
                            ((agent_y - (veh_neig[1][1] * 23 + 14)) + 7) / 14 if veh_neig[1][1] != 0 else 0,
                            ((agent_vx - (veh_neig[1][2] * 21 - 14)) + 18) / 36 if veh_neig[1][2] != 0 else 0,
                            ((agent_vy - (veh_neig[1][3] * 12 - 2)) + 5) / 7 if veh_neig[1][3] != 0 else 0,

                            ((agent_x - (veh_neig[2][0] * 38 - 4)) + 15) / 30 if veh_neig[2][0] != 0 else 0,
                            ((agent_y - (veh_neig[2][1] * 23 + 14)) + 7) / 14 if veh_neig[2][1] != 0 else 0,
                            ((agent_vx - (veh_neig[2][2] * 21 - 14)) + 14) / 25 if veh_neig[2][2] != 0 else 0,
                            ((agent_vy - (veh_neig[2][3] * 12 - 2)) + 1) / 2 if veh_neig[2][3] != 0 else 0,

                            ((agent_x - (veh_neig[3][0] * 38 - 4)) + 14) / 5 if veh_neig[3][0] != 0 else 0,
                            ((agent_y - (veh_neig[3][1] * 23 + 14)) + 6) / 4 if veh_neig[3][1] != 0 else 0,
                            ((agent_vx - (veh_neig[3][2] * 21 - 14)) - 4) / 7 if veh_neig[3][2] != 0 else 0,
                            ((agent_vy - (veh_neig[3][3] * 12 - 2)) - 0) / 1 if veh_neig[3][3] != 0 else 0,

                            ((agent_x - (veh_neig[4][0] * 38 - 4)) + 15) / 30 if veh_neig[4][0] != 0 else 0,
                            ((agent_y - (veh_neig[4][1] * 23 + 14)) + 15) / 24 if veh_neig[4][1] != 0 else 0,
                            ((agent_vx - (veh_neig[4][2] * 21 - 14)) + 13) / 26 if veh_neig[4][2] != 0 else 0,
                            ((agent_vy - (veh_neig[4][3] * 12 - 2)) + 9) / 15 if veh_neig[4][3] != 0 else 0,

                            ((agent_x - (veh_neig[5][0] * 38 - 4)) + 15) / 30 if veh_neig[5][0] != 0 else 0,
                            ((agent_y - (veh_neig[5][1] * 23 + 14)) + 15) / 24 if veh_neig[5][1] != 0 else 0,
                            ((agent_vx - (veh_neig[5][2] * 21 - 14)) + 13) / 26 if veh_neig[5][2] != 0 else 0,
                            ((agent_vy - (veh_neig[5][3] * 12 - 2)) + 8) / 14 if veh_neig[5][3] != 0 else 0,

                            ((agent_x - (veh_neig[6][0] * 38 - 4)) + 15) / 4 if veh_neig[6][0] != 0 else 0,
                            ((agent_y - (veh_neig[6][1] * 23 + 14)) - 2) / 4 if veh_neig[6][1] != 0 else 0,
                            ((agent_vx - (veh_neig[6][2] * 21 - 14)) - 4) / 2 if veh_neig[6][2] != 0 else 0,
                            ((agent_vy - (veh_neig[6][3] * 12 - 2)) - 2) / 1 if veh_neig[6][3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[0][0] * 39 - 5)) + 14) / 29 if veh_neig_landmark[0][0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[0][1] * 38 - 3)) + 15) / 30 if veh_neig_landmark[0][1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[0][2] * 31 - 16)) + 21) / 35 if veh_neig_landmark[0][2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[0][3] * 21 - 10)) + 5) / 16 if veh_neig_landmark[0][3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[1][0] * 39 - 5)) + 15) / 30 if veh_neig_landmark[1][0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[1][1] * 38 - 3)) + 15) / 29 if veh_neig_landmark[1][1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[1][2] * 31 - 16)) + 14) / 25 if veh_neig_landmark[1][2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[1][3] * 21 - 10)) + 7) / 17 if veh_neig_landmark[1][3] != 0 else 0,
                            veh_neig[0][4] if veh_neig[0][4] != 0 else 0,
                            veh_neig[1][4] if veh_neig[1][4] != 0 else 0,
                            veh_neig[2][4] if veh_neig[2][4] != 0 else 0,
                            veh_neig[3][4] if veh_neig[3][4] != 0 else 0,
                            veh_neig[4][4] if veh_neig[4][4] != 0 else 0,
                            veh_neig[5][4] if veh_neig[5][4] != 0 else 0,
                            veh_neig[6][4] if veh_neig[6][4] != 0 else 0,
                            veh_neig_landmark[0][4] if veh_neig_landmark[0][4] != 0 else 0,
                            veh_neig_landmark[1][4] if veh_neig_landmark[1][4] != 0 else 0,
                            agent.state.min_distance, agent.state.delta_angle_last1]).reshape([1, -1])

            obs_usefor_reward = np.array(
                [veh_neig[0][0] * 38 - 4, veh_neig[0][1] * 23 + 14,
                 veh_neig[0][2] * 21 - 14, veh_neig[0][3] * 12 - 2, veh_neig[0][4] * 191 - 1,
                 veh_neig[1][0] * 38 - 4, veh_neig[1][1] * 23 + 14,
                 veh_neig[1][2] * 21 - 14, veh_neig[1][3] * 12 - 2, veh_neig[1][4] * 191 - 1,
                 veh_neig[2][0] * 38 - 4, veh_neig[2][1] * 23 + 14,
                 veh_neig[2][2] * 21 - 14, veh_neig[2][3] * 12 - 2, veh_neig[2][4] * 191 - 1,
                 veh_neig[3][0] * 38 - 4, veh_neig[3][1] * 23 + 14,
                 veh_neig[3][2] * 21 - 14, veh_neig[3][3] * 12 - 2, veh_neig[3][4] * 191 - 1,
                 veh_neig[4][0] * 38 - 4, veh_neig[4][1] * 23 + 14,
                 veh_neig[4][2] * 21 - 14, veh_neig[4][3] * 12 - 2, veh_neig[4][4] * 191 - 1,
                 veh_neig[5][0] * 38 - 4, veh_neig[5][1] * 23 + 14,
                 veh_neig[5][2] * 21 - 14, veh_neig[5][3] * 12 - 2, veh_neig[5][4] * 191 - 1,
                 veh_neig[6][0] * 38 - 4, veh_neig[6][1] * 23 + 14,
                 veh_neig[6][2] * 21 - 14, veh_neig[6][3] * 12 - 2, veh_neig[6][4] * 191 - 1,
                 veh_neig_landmark[0][0] * 39 - 5, veh_neig_landmark[0][1] * 38 - 3,
                 veh_neig_landmark[0][2] * 31 - 16, veh_neig_landmark[0][3] * 21 - 10, veh_neig_landmark[0][4] * 360 - 90,
                 veh_neig_landmark[1][0] * 39 - 5, veh_neig_landmark[1][1] * 38 - 3,
                 veh_neig_landmark[1][2] * 31 - 16, veh_neig_landmark[1][3] * 21 - 10, veh_neig_landmark[1][4] * 360 - 90,
                 ]).reshape([1, -1])

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new+37) / 59, (d_y_new+4) / 27])

            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]),np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))
            ini_steps = np.array([agent.state.ini_step])
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
                current_agent_step = int(trj_go_step)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
                current_agent_step = int(trj_go_step) + 1

            obs_data = []

            for i in range(0, 21):
                current_step = current_agent_step - i
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))
            obs_data_lstm = np.vstack(obs_data)
        else:
            a = np.zeros([57])
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
            obs_data = []

            for i in range(0, 21):
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            obs_data_lstm = np.vstack(obs_data)
            obs_usefor_reward = np.zeros([45]).reshape([1, -1])
        return obs_data_lstm, obs_usefor_reward, a

    def observation_nowstep(self, agent, action_n_agent, world):
        trj_go_step = action_n_agent[3]
        if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            if action_n_agent[2] >= action_n_agent[4]:
                landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]
                landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]
            else:
                landmarks_veh = [np.zeros([7]) for _ in range(30)]
                landmarks_veh_use = [other[:5] for other in landmarks_veh]

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0]!=0)]

            veh_same = []
            veh_left_agents = []
            veh_right_agents = []
            veh_left_landmark = []
            veh_right_landmark = []

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4)) ** 2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14)) ** 2) for ve in
                                        vehs_agent])

            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5)) ** 2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3)) ** 2)
                                           for ve_landmark in
                                           landmarks_veh_use])


            for ii in range(len(vehs_agent)):
                if vehs_agent[ii][0] > 0:

                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:
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
                    cross = np.cross((b[0], b[1]), ((vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                                    (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if vehs_agent[ii][5] <= 2:
                        jiaohu_agent_direction = 'left'
                    else:
                        jiaohu_agent_direction = 'straight'


                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []

            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))

            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    for i in range(3):
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    top_3_veh = sorted_veh[:3, :]
                    veh_new = top_3_veh.copy()
                    for i in range(3):
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            for iii in range(len(landmarks_veh_use)):
                if landmarks_veh_use[iii][0] > 0:

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:
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
                    cross = np.cross((b[0], b[1]), ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                                    (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []

            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            obs = np.array([((agent_x - (veh_neig[0][0] * 38 - 4)) + 12) / 27 if veh_neig[0][0] != 0 else 0,
                            ((agent_y - (veh_neig[0][1] * 23 + 14)) + 14) / 18 if veh_neig[0][1] != 0 else 0,
                            ((agent_vx - (veh_neig[0][2] * 21 - 14)) + 7) / 14 if veh_neig[0][2] != 0 else 0,
                            ((agent_vy - (veh_neig[0][3] * 12 - 2)) + 5) / 6 if veh_neig[0][3] != 0 else 0,

                            ((agent_x - (veh_neig[1][0] * 38 - 4)) + 15) / 30 if veh_neig[1][0] != 0 else 0,
                            ((agent_y - (veh_neig[1][1] * 23 + 14)) + 7) / 14 if veh_neig[1][1] != 0 else 0,
                            ((agent_vx - (veh_neig[1][2] * 21 - 14)) + 18) / 36 if veh_neig[1][2] != 0 else 0,
                            ((agent_vy - (veh_neig[1][3] * 12 - 2)) + 5) / 7 if veh_neig[1][3] != 0 else 0,

                            ((agent_x - (veh_neig[2][0] * 38 - 4)) + 15) / 30 if veh_neig[2][0] != 0 else 0,
                            ((agent_y - (veh_neig[2][1] * 23 + 14)) + 7) / 14 if veh_neig[2][1] != 0 else 0,
                            ((agent_vx - (veh_neig[2][2] * 21 - 14)) + 14) / 25 if veh_neig[2][2] != 0 else 0,
                            ((agent_vy - (veh_neig[2][3] * 12 - 2)) + 1) / 2 if veh_neig[2][3] != 0 else 0,

                            ((agent_x - (veh_neig[3][0] * 38 - 4)) + 14) / 5 if veh_neig[3][0] != 0 else 0,
                            ((agent_y - (veh_neig[3][1] * 23 + 14)) + 6) / 4 if veh_neig[3][1] != 0 else 0,
                            ((agent_vx - (veh_neig[3][2] * 21 - 14)) - 4) / 7 if veh_neig[3][2] != 0 else 0,
                            ((agent_vy - (veh_neig[3][3] * 12 - 2)) - 0) / 1 if veh_neig[3][3] != 0 else 0,

                            ((agent_x - (veh_neig[4][0] * 38 - 4)) + 15) / 30 if veh_neig[4][0] != 0 else 0,
                            ((agent_y - (veh_neig[4][1] * 23 + 14)) + 15) / 24 if veh_neig[4][1] != 0 else 0,
                            ((agent_vx - (veh_neig[4][2] * 21 - 14)) + 13) / 26 if veh_neig[4][2] != 0 else 0,
                            ((agent_vy - (veh_neig[4][3] * 12 - 2)) + 9) / 15 if veh_neig[4][3] != 0 else 0,

                            ((agent_x - (veh_neig[5][0] * 38 - 4)) + 15) / 30 if veh_neig[5][0] != 0 else 0,
                            ((agent_y - (veh_neig[5][1] * 23 + 14)) + 15) / 24 if veh_neig[5][1] != 0 else 0,
                            ((agent_vx - (veh_neig[5][2] * 21 - 14)) + 13) / 26 if veh_neig[5][2] != 0 else 0,
                            ((agent_vy - (veh_neig[5][3] * 12 - 2)) + 8) / 14 if veh_neig[5][3] != 0 else 0,

                            ((agent_x - (veh_neig[6][0] * 38 - 4)) + 15) / 4 if veh_neig[6][0] != 0 else 0,
                            ((agent_y - (veh_neig[6][1] * 23 + 14)) - 2) / 4 if veh_neig[6][1] != 0 else 0,
                            ((agent_vx - (veh_neig[6][2] * 21 - 14)) - 4) / 2 if veh_neig[6][2] != 0 else 0,
                            ((agent_vy - (veh_neig[6][3] * 12 - 2)) - 2) / 1 if veh_neig[6][3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[0][0] * 39 - 5)) + 14) / 29 if veh_neig_landmark[0][0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[0][1] * 38 - 3)) + 15) / 30 if veh_neig_landmark[0][1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[0][2] * 31 - 16)) + 21) / 35 if veh_neig_landmark[0][2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[0][3] * 21 - 10)) + 5) / 16 if veh_neig_landmark[0][3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[1][0] * 39 - 5)) + 15) / 30 if veh_neig_landmark[1][0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[1][1] * 38 - 3)) + 15) / 29 if veh_neig_landmark[1][1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[1][2] * 31 - 16)) + 14) / 25 if veh_neig_landmark[1][2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[1][3] * 21 - 10)) + 7) / 17 if veh_neig_landmark[1][3] != 0 else 0,
                            veh_neig[0][4] if veh_neig[0][4] != 0 else 0,
                            veh_neig[1][4] if veh_neig[1][4] != 0 else 0,
                            veh_neig[2][4] if veh_neig[2][4] != 0 else 0,
                            veh_neig[3][4] if veh_neig[3][4] != 0 else 0,
                            veh_neig[4][4] if veh_neig[4][4] != 0 else 0,
                            veh_neig[5][4] if veh_neig[5][4] != 0 else 0,
                            veh_neig[6][4] if veh_neig[6][4] != 0 else 0,
                            veh_neig_landmark[0][4] if veh_neig_landmark[0][4] != 0 else 0,
                            veh_neig_landmark[1][4] if veh_neig_landmark[1][4] != 0 else 0,
                            agent.state.min_distance, agent.state.delta_angle_last1]).reshape([1, -1])

            obs_usefor_reward = np.array(
                [veh_neig[0][0] * 38 - 4, veh_neig[0][1] * 23 + 14,
                 veh_neig[0][2] * 21 - 14, veh_neig[0][3] * 12 - 2, veh_neig[0][4] * 191 - 1,
                 veh_neig[1][0] * 38 - 4, veh_neig[1][1] * 23 + 14,
                 veh_neig[1][2] * 21 - 14, veh_neig[1][3] * 12 - 2, veh_neig[1][4] * 191 - 1,
                 veh_neig[2][0] * 38 - 4, veh_neig[2][1] * 23 + 14,
                 veh_neig[2][2] * 21 - 14, veh_neig[2][3] * 12 - 2, veh_neig[2][4] * 191 - 1,
                 veh_neig[3][0] * 38 - 4, veh_neig[3][1] * 23 + 14,
                 veh_neig[3][2] * 21 - 14, veh_neig[3][3] * 12 - 2, veh_neig[3][4] * 191 - 1,
                 veh_neig[4][0] * 38 - 4, veh_neig[4][1] * 23 + 14,
                 veh_neig[4][2] * 21 - 14, veh_neig[4][3] * 12 - 2, veh_neig[4][4] * 191 - 1,
                 veh_neig[5][0] * 38 - 4, veh_neig[5][1] * 23 + 14,
                 veh_neig[5][2] * 21 - 14, veh_neig[5][3] * 12 - 2, veh_neig[5][4] * 191 - 1,
                 veh_neig[6][0] * 38 - 4, veh_neig[6][1] * 23 + 14,
                 veh_neig[6][2] * 21 - 14, veh_neig[6][3] * 12 - 2, veh_neig[6][4] * 191 - 1,
                 veh_neig_landmark[0][0] * 39 - 5, veh_neig_landmark[0][1] * 38 - 3,
                 veh_neig_landmark[0][2] * 31 - 16, veh_neig_landmark[0][3] * 21 - 10, veh_neig_landmark[0][4] * 360 - 90,
                 veh_neig_landmark[1][0] * 39 - 5, veh_neig_landmark[1][1] * 38 - 3,
                 veh_neig_landmark[1][2] * 31 - 16, veh_neig_landmark[1][3] * 21 - 10, veh_neig_landmark[1][4] * 360 - 90,
                 ]).reshape([1, -1])

            des = np.zeros(2)

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new+37) / 59, (d_y_new+4) / 27])

            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]),np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))
            ini_steps = np.array([agent.state.ini_step])

            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            current_agent_step = int(trj_go_step)

            obs_data = []

            for i in range(0, 21):
                current_step = current_agent_step - i
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))

            obs_data_lstm = np.vstack(obs_data)

        else:
            a = np.zeros([57])
            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            obs_data = []

            for i in range(0, 21):
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            obs_data_lstm = np.vstack(obs_data)
            obs_usefor_reward = np.zeros([45]).reshape([1, -1])

        return obs_data_lstm, obs_usefor_reward, a

    def done(self, agent, world):
        if world.time >= 185:
            return True
        elif agent.collide == True:
            return True
        elif agent.end_label == True:
            return True
        elif agent.state.step >= 185:  # 185 for training 230 for testing
            return True
        else:
            return False
