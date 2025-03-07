import gym
import click
import multiagent
import time
import tensorflow as tf
import make_env
import numpy as np
from rl.common.misc_util import set_global_seeds
from sandbox.mack.acktr_disc import onehot
from irl.mack.airl_con_ac import Model
from irl.mack.kfac_discriminator_airl import Discriminator
# from irl.mack.gail_con_ac import Model
from sandbox.mack.policies import MaskedCategoricalPolicy, GaussianPolicy, MASKATTGaussianPolicy
from rl import bench
# import imageio
import pickle as pkl
import copy
import math


# dependency = np.load('dependency_dd4.npy', allow_pickle = True) ####

# @click.command()
# @click.option('--env', type=click.STRING)
# @click.option('--image', is_flag=True, flag_value=True)

def create_env(env_id, scenario_test, training_label):
    env = make_env.make_env(env_id, scenario_test, training_label)
    env.seed(10)
    # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
    set_global_seeds(10)
    return env

def makeModel(env_id, scenario_test, training_label):
    tf.reset_default_graph()
    env = create_env(env_id, scenario_test, training_label)
    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space

    identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]

    make_model = lambda: Model(
        MASKATTGaussianPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=10, nsteps=500,
        nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.1, max_grad_norm=0.5, kfac_clip=0.001,
        lrschedule='linear', identical=identical)

    model = make_model()
    return model

def get_dis(path, model, env_id, mid, scenario_test, training_label):
    print("load model from", path)
    env = create_env(env_id, scenario_test, training_label)

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    discriminator = [
        Discriminator(model.sess, ob_space, ac_space,
                      state_only=True, discount=0.99, nstack=1, index=k, disc_type='decentralized',
                      scope="Discriminator_%d" % k,  # gp_coef=gp_coef,
                      total_steps=1e7 // (10 * 500),
                      lr_rate=0.1, l2_loss_ratio=0.1) for k in range(n_agents)
    ]
    model.sess.run(tf.global_variables_initializer())
    for n_v in range(8):
        did = str(n_v) + '_0' + str(mid)
        path2 = path + 'd_' + did
        discriminator[n_v].load(path2)
    return discriminator


def render(path, model, env_id, mid, scenario_test, training_label):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id, scenario_test, training_label)

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space

    images = []
    sample_trajs = []
    num_trajs = 1
    max_steps = 230
    avg_ret = [[] for _ in range(n_agents)]

    for i in range(num_trajs):
        all_attention_weight_spatial, all_attention_weight_temporal, all_ob_lstm, all_ob, all_agent_ob, all_ac, all_rew, all_social, ep_ret = [], [], [], [], [], [], [], [], [0 for k in range(n_agents)]
        for k in range(n_agents):
            all_ob_lstm.append([])
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
            all_attention_weight_spatial.append([])
            all_attention_weight_temporal.append([])
            all_social.append([])
        obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm = env.reset()
        action = [np.zeros(2) for _ in range(n_agents)]
        step = 0
        done_eva = False
        obs_lstm = [copy.deepcopy(ob_lstm[None, :]) for ob_lstm in obs_lstm]
        obs = [copy.deepcopy(ob[None, :]) for ob in obs]
        ini_obs = [copy.deepcopy(ini[None, :]) for ini in ini_obs]
        action = [copy.deepcopy(ac[None, :]) for ac in action]
        ini_step_n = [copy.deepcopy(ini_step[None, :]) for ini_step in ini_step_n]
        ini_obs_lstm = [copy.deepcopy(ini_ob_lstm[None, :]) for ini_ob_lstm in ini_obs_lstm]
        trj_GO_STEP = [np.array([0 for _ in range(n_agents)])]

        while not done_eva:
            ini_update_inf = [np.array([0 for _ in range(8)]) for k in range(1)]
            for k in range(n_agents):
                ini_step_k = ini_step_n[k][0][0]
                if step == ini_step_k:
                    obs_lstm[k][0] = ini_obs_lstm[k][0]
                    obs[k][0] = ini_obs[k][0]
                    action[k][0] = np.zeros(2)
                    ini_update_inf[0][k] = True
                elif step < ini_step_k:
                    obs[k][0] = np.zeros(57)
                    action[k][0] = np.zeros(2)
                    obs_lstm[k][0][:] = 0
                    ini_update_inf[0][k] = False

            ini_obs_old_list = []
            for i in range(1):
                ini_obs_old_list.append(
                    [np.concatenate((action[k_][0], np.array([step]),
                                            np.array([trj_GO_STEP[0][k_]]),
                                            np.array([ini_step_n[k_][0][0]]), ini_obs[k_][0],
                                            np.array([ini_update_inf[i][k_]]))) for k_ in
                     range(8)])
            obs_lstm_nowstep, obs_nowstep = env.ini_obs_update(
                ini_obs_old_list[0])

            for i in range(0):
                for k in range(8):
                    if ini_update_inf[i][k] == True:
                        for j in range(len(obs_lstm)):
                            obs_lstm[j][i] = obs_lstm_nowstep[j]
                            obs[j][i] = obs_nowstep[j]
                        action[k][i] = np.zeros(2)

            action, _, _, atten_weights_spatial, atten_weights_temporal = model.step(obs_lstm, obs, action)

            for k_ in range(n_agents):
                all_ob[k_].append(obs[k_][0])
                all_ob_lstm[k_].append(obs_lstm[k_][0])
                all_attention_weight_spatial[k_].append(atten_weights_spatial[k_])
                all_attention_weight_temporal[k_].append(atten_weights_temporal[k_])

            action2 = []

            env_go_step = step

            for k__ in range(n_agents):
                trj_go_step = step - ini_step_n[k__][0][0]
                if trj_go_step > 0:
                    trj_GO_STEP[0][k__] = trj_go_step
                else:
                    trj_GO_STEP[0][k__] = trj_go_step

                if k__ <= 2:
                    acc = min(max(action[k__][0][0], -1), 1)
                    delta_theta = min(max(action[k__][0][1], -1.3), 1.2)
                    action[k__][0][0] = acc
                    action[k__][0][1] = delta_theta
                else:
                    acc = min(max(action[k__][0][0], -1), 1)
                    delta_theta = min(max(action[k__][0][1], -1), 1)
                    action[k__][0][0] = acc
                    action[k__][0][1] = delta_theta

            action2.append([np.concatenate((action[k][0], np.array([env_go_step]),
                                            np.array([trj_GO_STEP[0][k]]),
                                            np.array([ini_step_n[k][0][0]]), ini_obs[k][0])) for k in
                            range(n_agents)])
            all_agent_ob.append(np.concatenate(obs, axis=1))
            obs_lstm, obs, rew, dones, _, ini_step_n, ini_ob, action_new = env.step(action2[0])
            action_new = [action1_new[None, :] for action1_new in action_new]
            action = action_new

            for k in range(n_agents):
                if dones[k]:
                    obs[k] = obs[k] * 0.0
                    action[k] = action[k] * 0.0
                    obs_lstm[k] = obs_lstm[k] * 0.0

            for k_ in range(n_agents):
                all_ac[k_].append(action[k_][0])

            for k in range(n_agents):
                all_rew[k].append(rew[k])
                ep_ret[k] += rew[k]

            obs_lstm = [ob_lstm[None, :] for ob_lstm in obs_lstm]
            obs = [ob[None, :] for ob in obs]
            ini_step_n = [[[ini_step[0]]] for ini_step in ini_step_n]
            step += 1

            if step == max_steps:
                done_eva = True
                step = 0
            else:
                done_eva = False

        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])
            all_ob_lstm[k] = np.squeeze(all_ob_lstm[k])
            all_attention_weight_spatial[k] = np.squeeze(all_attention_weight_spatial[k])
            all_attention_weight_temporal[k] = np.squeeze(all_attention_weight_temporal[k])

        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob,'all_attention_weight_spatial':all_attention_weight_spatial,
            'all_attention_weight_temporal': all_attention_weight_temporal
        }

        sample_trajs.append(traj_data)

        for k in range(n_agents):
            avg_ret[k].append(ep_ret[k])

    return sample_trajs


def render_discrimination_nogenerate(path, path_d, model, env_id, mid, generate_trj, scenario_test, training_label):

    model.load(path)
    env = create_env(env_id, scenario_test, training_label)
    discriminator = get_dis(path_d, model, env_id, mid, scenario_test, training_label)
    print('discriminator', discriminator, np.shape(discriminator))

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space

    images = []
    sample_trajs = []
    num_trajs = 1
    max_steps = 230
    avg_ret = [[] for _ in range(n_agents)]

    all_ac, obs_lstm_all_agents, all_ob, all_rew, all_social, all_attention_weight= [], [], [], [],[], []
    for k in range(n_agents):
        obs_lstm_all_agents.append([])
        all_ob.append([])
        all_rew.append([])
        all_social.append([])
        all_ac.append([])
        all_attention_weight.append([])

    step = 0
    done_eva = False
    obs_all_agents = generate_trj['ob']
    acs_all_agents = generate_trj['ac']
    attention_weight_all_agents = generate_trj['all_attention_weight']

    for k in range(n_agents):
        agent_ob = obs_all_agents[k][:, :57]
        obs_k_t = []
        for t in range(max_steps):
            obs_t = np.zeros((21, 57))
            if t >= 20:
                obs_t = agent_ob[t - 20:t + 1]
            else:
                obs_t[20 - t:] = agent_ob[:t + 1]
            obs_k_t.append(obs_t)

        obs_lstm_all_agents[k].append(obs_k_t)

    while not done_eva:

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

        def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                      Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                      Jiaohu_direction):

            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:

                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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

        rewards_panbieqi = []
        social_panbieqi = []
        path_prob = np.zeros(n_agents)
        for k in range(n_agents):
            if k <= 2:
                direction_agent = 'left'
            else:
                direction_agent = 'straight'

            re_obs_lstm = obs_lstm_all_agents[k][0][step]

            obs_lstm_next = re_obs_lstm
            batch_num = np.shape(obs_lstm_all_agents)[1]

            rew_input_fuyuan = re_obs_lstm

            rew_social_allbatch = []

            for i_batch in range(1):
                if rew_input_fuyuan[20][0] != 0:
                    use_GT = []
                    pianyi_distance = rew_input_fuyuan[20][-2]
                    agent_x = rew_input_fuyuan[20][0] * 38 - 4
                    agent_y = rew_input_fuyuan[20][1] * 23 + 14
                    agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                    agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                    agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1

                    agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                    for agent_k_ in range(n_agents):
                        if agent_k_ != k:
                            if agent_k_ <= 2:
                                direction_jiaohu = 'left'
                            else:
                                direction_jiaohu = 'straight'

                            rew_input_fuyuan_agent_k_ = obs_lstm_all_agents[agent_k_][0][step]

                            if rew_input_fuyuan_agent_k_[20][10] != 0:
                                jiaohu_agent_x = rew_input_fuyuan[20][0] * 38 - 4
                                jiaohu_agent_y = rew_input_fuyuan[20][1] * 23 + 14
                                jiaohu_agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                                jiaohu_agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                                jiaohu_agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1

                                jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  jiaohu_agent_x, jiaohu_agent_y,
                                                                  jiaohu_agent_vx, jiaohu_agent_vy,
                                                                  jiaohu_agent_angle_last,
                                                                  direction_jiaohu)
                                use_GT.append(jiaohu_agent_GT_value)
                            else:
                                jiaohu_agent_x = -4
                                jiaohu_agent_y = 14
                                jiaohu_agentk_vx = -14
                                jiaohu_agent_vy = -2
                                jiaohu_agent_angle_last = -1
                                jiaohu_agent_GT_value = None
                                # dis_min = 100000
                                use_GT.append(jiaohu_agent_GT_value)

                    # landmark in left
                    if rew_input_fuyuan[20][38] != 0:
                        delta_left_jiaohu_landmark_x = rew_input_fuyuan[20][38] * 29 - 14
                        delta_left_jiaohu_landmark_y = rew_input_fuyuan[20][39] * 30 - 15
                        delta_left_jiaohu_landmark_vx = rew_input_fuyuan[20][40] * 35 - 21
                        delta_left_jiaohu_landmark_vy = rew_input_fuyuan[20][41] * 16 - 5
                        left_jiaohu_landmark_angle_last = rew_input_fuyuan[20][53] * 360 - 90
                        left_jiaohu_landmark_x = agent_x - delta_left_jiaohu_landmark_x
                        left_jiaohu_landmark_y = agent_y - delta_left_jiaohu_landmark_y
                        left_jiaohu_landmark_vx = agent_vx - delta_left_jiaohu_landmark_vx
                        left_jiaohu_landmark_vy = agent_vy - delta_left_jiaohu_landmark_vy
                        left_jiaohu_landmark_angle_last = left_jiaohu_landmark_angle_last
                    else:
                        left_jiaohu_landmark_x = -5
                        left_jiaohu_landmark_y = -3
                        left_jiaohu_landmark_vx = -16
                        left_jiaohu_landmark_vy = -10
                        left_jiaohu_landmark_angle_last = -90

                    # landmark in right
                    if rew_input_fuyuan[20][42] != 0:
                        delta_right_jiaohu_landmark_x = rew_input_fuyuan[20][42] * 35 - 15
                        delta_right_jiaohu_landmark_y = rew_input_fuyuan[20][43] * 29 - 15
                        delta_right_jiaohu_landmark_vx = rew_input_fuyuan[20][44] * 25 - 14
                        delta_right_jiaohu_landmark_vy = rew_input_fuyuan[20][45] * 17 - 7
                        right_jiaohu_landmark_angle_last = rew_input_fuyuan[20][54] * 360 - 90
                        right_jiaohu_landmark_x = agent_x - delta_right_jiaohu_landmark_x
                        right_jiaohu_landmark_y = agent_y - delta_right_jiaohu_landmark_y
                        right_jiaohu_landmark_vx = agent_vx - delta_right_jiaohu_landmark_vx
                        right_jiaohu_landmark_vy = agent_vy - delta_right_jiaohu_landmark_vy
                        right_jiaohu_landmark_angle_last = right_jiaohu_landmark_angle_last
                    else:
                        right_jiaohu_landmark_x = -5
                        right_jiaohu_landmark_y = -3
                        right_jiaohu_landmark_vx = -16
                        right_jiaohu_landmark_vy = -10
                        right_jiaohu_landmark_angle_last = -90

                    if left_jiaohu_landmark_x != -5:
                        direction_landmark = 'landmark'
                        left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  left_jiaohu_landmark_x,
                                                                  left_jiaohu_landmark_y,
                                                                  left_jiaohu_landmark_vx,
                                                                  left_jiaohu_landmark_vy,
                                                                  left_jiaohu_landmark_angle_last,
                                                                  direction_landmark)

                        use_GT.append(left_jiaohu_landmark_GT_value)
                    else:
                        left_jiaohu_landmark_GT_value = None
                        # dis_min = 100000
                        use_GT.append(left_jiaohu_landmark_GT_value)

                    if right_jiaohu_landmark_x != -5:
                        direction_landmark = 'landmark'
                        right_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                   agent_angle_last, direction_agent,
                                                                   right_jiaohu_landmark_x,
                                                                   right_jiaohu_landmark_y,
                                                                   right_jiaohu_landmark_vx,
                                                                   right_jiaohu_landmark_vy,
                                                                   right_jiaohu_landmark_angle_last,
                                                                   direction_landmark)

                        use_GT.append(right_jiaohu_landmark_GT_value)

                    else:
                        right_jiaohu_landmark_GT_value = None
                        # dis_min = 10000
                        use_GT.append(right_jiaohu_landmark_GT_value)

                    penalty = 1
                    delta_angle_last1 = rew_input_fuyuan[20][56]
                    # comfort
                    comfort_adj = 0
                    if direction_agent == 'left':
                        left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                        left_delta_angle_last1_realmean = 1.085
                        left_delta_angle_last1_realstd = 0.702
                        if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                            comfort_adj = 0

                        else:
                            dis_left_delta_angle_last1 = abs(
                                left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                            if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                        dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                    else:
                        right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                        right_delta_angle_last1_realmean = 0.001
                        right_delta_angle_last1_realstd = 0.076
                        if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                            comfort_adj = 0

                        else:
                            dis_right_delta_angle_last1 = abs(
                                right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                            if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                            dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty

                    # efficiency
                    rew_avespeed = agent_v / 6.8  # 85th expert speed
                    # lane offset
                    rew_lane_pianyi = pianyi_distance
                    # safe
                    use_GT_list = [x for x in use_GT if x is not None]
                    if len(use_GT_list) != 0:
                        rew_minGT = sum(use_GT_list) / len(use_GT_list)
                        if rew_minGT <= 1.5:
                            normalized_data = (rew_minGT - 0) / (1.5 - 0)
                            rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                        elif 1.5 < rew_minGT < 3:
                            normalized_data = (rew_minGT - 1.5) / (3 - 1.5)
                            rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                        elif 3 <= rew_minGT <= 4:
                            normalized_data = (rew_minGT - 3) / (4 - 3)
                            rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                        elif rew_minGT > 4:
                            normalized_data = np.exp(-(1 / (rew_minGT - 4)))
                            rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                    else:
                        rew_minGT_mapped = 0
                        social_pre_ibatch = 0

                    rew_social_allbatch.append(
                        [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                else:
                    rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

            canshu_social_allbatch_array = np.array(rew_social_allbatch)

            score, pre = discriminator[k].get_reward(re_obs_lstm,
                                                     np.array([0, 0]),
                                                     obs_lstm_next,
                                                     path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)

            rewards_panbieqi.append(np.squeeze(score))
            social_panbieqi.append(np.squeeze(pre))

        for k in range(n_agents):
            all_rew[k].append(rewards_panbieqi[k])
            all_social[k].append(social_panbieqi[k])

        step += 1
        if step == max_steps:
            done_eva = True
            step = 0
        else:
            done_eva = False

    for k in range(n_agents):
        all_ob[k] = obs_all_agents[k]
        all_ac[k] = acs_all_agents[k]
        all_attention_weight[k] = attention_weight_all_agents[k]

    traj_data = {
        "ob": all_ob, "rew": all_rew, 'all_social': all_social, "ac": all_ac, 'all_attention_weight':all_attention_weight
    }

    sample_trajs.append(traj_data)

    return sample_trajs


def render_discrimination_expert(path, path_d, model, env_id, mid, expert_trj, scenario_test, training_label):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id, scenario_test, training_label)

    discriminator = get_dis(path_d, model, env_id, mid, scenario_test, training_label)  # 获得判别器
    print('discriminator', discriminator, np.shape(discriminator))

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space

    images = []
    sample_trajs = []
    num_trajs = 1
    max_steps = 185
    avg_ret = [[] for _ in range(n_agents)]

    all_ac, obs_lstm_all_agents, all_ob, all_rew, all_social, all_attention_weight= [], [], [], [],[], []
    for k in range(n_agents):
        obs_lstm_all_agents.append([])
        all_ob.append([])
        all_rew.append([])
        all_social.append([])
        all_ac.append([])
        all_attention_weight.append([])

    step = 0
    done_eva = False
    obs_all_agents = expert_trj['ob']
    acs_all_agents = expert_trj['ac']

    for k in range(n_agents):
        agent_ob = obs_all_agents[k][:, :57]

        obs_k_t = []
        for t in range(max_steps):

            obs_t = np.zeros((21, 57))
            if t >= 20:
                obs_t = agent_ob[t - 20:t + 1]
            else:
                obs_t[20 - t:] = agent_ob[:t + 1]
            obs_k_t.append(obs_t)

        obs_lstm_all_agents[k].append(obs_k_t)

    while not done_eva:

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


        def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                      Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                      Jiaohu_direction):

            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:
                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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

        rewards_panbieqi = []
        social_panbieqi = []
        path_prob = np.zeros(n_agents)
        for k in range(n_agents):
            if k <= 2:
                direction_agent = 'left'
            else:
                direction_agent = 'straight'

            re_obs_lstm = obs_lstm_all_agents[k][0][step]

            obs_lstm_next = re_obs_lstm
            batch_num = np.shape(obs_lstm_all_agents)[1]

            rew_input_fuyuan = re_obs_lstm

            rew_social_allbatch = []

            for i_batch in range(1):
                if rew_input_fuyuan[20][0] != 0:
                    use_GT = []
                    pianyi_distance = rew_input_fuyuan[20][-2]
                    agent_x = rew_input_fuyuan[20][0] * 38 - 4
                    agent_y = rew_input_fuyuan[20][1] * 23 + 14
                    agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                    agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                    agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1

                    agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                    for agent_k_ in range(n_agents):
                        if agent_k_ != k:
                            if agent_k_ <= 2:
                                direction_jiaohu = 'left'
                            else:
                                direction_jiaohu = 'straight'

                            rew_input_fuyuan_agent_k_ = obs_lstm_all_agents[agent_k_][0][step]

                            if rew_input_fuyuan_agent_k_[20][10] != 0:
                                jiaohu_agent_x = rew_input_fuyuan[20][0] * 38 - 4
                                jiaohu_agent_y = rew_input_fuyuan[20][1] * 23 + 14
                                jiaohu_agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                                jiaohu_agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                                jiaohu_agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1

                                jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  jiaohu_agent_x, jiaohu_agent_y,
                                                                  jiaohu_agent_vx, jiaohu_agent_vy,
                                                                  jiaohu_agent_angle_last,
                                                                  direction_jiaohu)
                                use_GT.append(jiaohu_agent_GT_value)
                            else:
                                jiaohu_agent_x = -4
                                jiaohu_agent_y = 14
                                jiaohu_agentk_vx = -14
                                jiaohu_agent_vy = -2
                                jiaohu_agent_angle_last = -1
                                jiaohu_agent_GT_value = None
                                # dis_min = 100000
                                use_GT.append(jiaohu_agent_GT_value)

                    # landmark in left
                    if rew_input_fuyuan[20][38] != 0:
                        delta_left_jiaohu_landmark_x = rew_input_fuyuan[20][38] * 29 - 14
                        delta_left_jiaohu_landmark_y = rew_input_fuyuan[20][39] * 30 - 15
                        delta_left_jiaohu_landmark_vx = rew_input_fuyuan[20][40] * 35 - 21
                        delta_left_jiaohu_landmark_vy = rew_input_fuyuan[20][41] * 16 - 5
                        left_jiaohu_landmark_angle_last = rew_input_fuyuan[20][53] * 360 - 90
                        left_jiaohu_landmark_x = agent_x - delta_left_jiaohu_landmark_x
                        left_jiaohu_landmark_y = agent_y - delta_left_jiaohu_landmark_y
                        left_jiaohu_landmark_vx = agent_vx - delta_left_jiaohu_landmark_vx
                        left_jiaohu_landmark_vy = agent_vy - delta_left_jiaohu_landmark_vy
                        left_jiaohu_landmark_angle_last = left_jiaohu_landmark_angle_last
                    else:
                        left_jiaohu_landmark_x = -5
                        left_jiaohu_landmark_y = -3
                        left_jiaohu_landmark_vx = -16
                        left_jiaohu_landmark_vy = -10
                        left_jiaohu_landmark_angle_last = -90

                    # landmark in right
                    if rew_input_fuyuan[20][42] != 0:
                        delta_right_jiaohu_landmark_x = rew_input_fuyuan[20][42] * 35 - 15
                        delta_right_jiaohu_landmark_y = rew_input_fuyuan[20][43] * 29 - 15
                        delta_right_jiaohu_landmark_vx = rew_input_fuyuan[20][44] * 25 - 14
                        delta_right_jiaohu_landmark_vy = rew_input_fuyuan[20][45] * 17 - 7
                        right_jiaohu_landmark_angle_last = rew_input_fuyuan[20][54] * 360 - 90
                        right_jiaohu_landmark_x = agent_x - delta_right_jiaohu_landmark_x
                        right_jiaohu_landmark_y = agent_y - delta_right_jiaohu_landmark_y
                        right_jiaohu_landmark_vx = agent_vx - delta_right_jiaohu_landmark_vx
                        right_jiaohu_landmark_vy = agent_vy - delta_right_jiaohu_landmark_vy
                        right_jiaohu_landmark_angle_last = right_jiaohu_landmark_angle_last
                    else:
                        right_jiaohu_landmark_x = -5
                        right_jiaohu_landmark_y = -3
                        right_jiaohu_landmark_vx = -16
                        right_jiaohu_landmark_vy = -10
                        right_jiaohu_landmark_angle_last = -90

                    if left_jiaohu_landmark_x != -5:
                        direction_landmark = 'landmark'
                        left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  left_jiaohu_landmark_x,
                                                                  left_jiaohu_landmark_y,
                                                                  left_jiaohu_landmark_vx,
                                                                  left_jiaohu_landmark_vy,
                                                                  left_jiaohu_landmark_angle_last,
                                                                  direction_landmark)

                        use_GT.append(left_jiaohu_landmark_GT_value)
                    else:
                        left_jiaohu_landmark_GT_value = None
                        # dis_min = 100000
                        use_GT.append(left_jiaohu_landmark_GT_value)

                    if right_jiaohu_landmark_x != -5:
                        direction_landmark = 'landmark'
                        right_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                   agent_angle_last, direction_agent,
                                                                   right_jiaohu_landmark_x,
                                                                   right_jiaohu_landmark_y,
                                                                   right_jiaohu_landmark_vx,
                                                                   right_jiaohu_landmark_vy,
                                                                   right_jiaohu_landmark_angle_last,
                                                                   direction_landmark)

                        use_GT.append(right_jiaohu_landmark_GT_value)

                    else:
                        right_jiaohu_landmark_GT_value = None
                        # dis_min = 10000
                        use_GT.append(right_jiaohu_landmark_GT_value)

                    penalty = 1
                    delta_angle_last1 = rew_input_fuyuan[20][56]
                    # comfort
                    comfort_adj = 0
                    if direction_agent == 'left':
                        left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                        left_delta_angle_last1_realmean = 1.085
                        left_delta_angle_last1_realstd = 0.702
                        if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                            comfort_adj = 0

                        else:
                            dis_left_delta_angle_last1 = abs(
                                left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                            if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                        dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                    else:
                        right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                        right_delta_angle_last1_realmean = 0.001
                        right_delta_angle_last1_realstd = 0.076
                        if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                            comfort_adj = 0

                        else:
                            dis_right_delta_angle_last1 = abs(
                                right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                            if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                            dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty

                    # efficiency
                    rew_avespeed = agent_v / 6.8  # 85th expert speed

                    # lane offset
                    rew_lane_pianyi = pianyi_distance

                    # safe
                    use_GT_list = [x for x in use_GT if x is not None]
                    if len(use_GT_list) != 0:
                        rew_minGT = sum(use_GT_list) / len(use_GT_list)
                        if rew_minGT <= 1.5:
                            normalized_data = (rew_minGT - 0) / (1.5 - 0)
                            rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                        elif 1.5 < rew_minGT < 3:
                            normalized_data = (rew_minGT - 1.5) / (3 - 1.5)
                            rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                        elif 3 <= rew_minGT <= 4:
                            normalized_data = (rew_minGT - 3) / (4 - 3)
                            rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                        elif rew_minGT > 4:
                            normalized_data = np.exp(-(1 / (rew_minGT - 4)))
                            rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                    else:
                        rew_minGT_mapped = 0

                    rew_social_allbatch.append(
                        [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                else:
                    rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

            canshu_social_allbatch_array = np.array(rew_social_allbatch)

            score, pre = discriminator[k].get_reward(re_obs_lstm,
                                                     np.array([0, 0]),
                                                     obs_lstm_next,
                                                     path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)

            rewards_panbieqi.append(np.squeeze(score))
            social_panbieqi.append(np.squeeze(pre))

        for k in range(n_agents):
            all_rew[k].append(rewards_panbieqi[k])
            all_social[k].append(social_panbieqi[k])

        step += 1
        if step == max_steps:
            done_eva = True
            step = 0
        else:
            done_eva = False

    for k in range(n_agents):
        all_ob[k] = obs_all_agents[k]
        all_ac[k] = acs_all_agents[k]

    traj_data = {
        "ob": all_ob, "rew": all_rew, 'all_social': all_social, "ac": all_ac
    }

    sample_trajs.append(traj_data)

    return sample_trajs

def mimic(path, model, env):
    print("load model from", path)
    model.load(path)
    expert_path = 'multi-agent-trj/expert_trjs/dronedata_veh143_8.pkl'
    with open(expert_path, "rb") as f:
        base_traj = pkl.load(f)
    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    n_actions = [action.n for action in ac_space]
    num_links = 143

    images = []
    sample_trajs = []
    num_trajs = 20  ####
    max_steps = 80  ####
    avg_ret = [[] for _ in range(n_agents)]

    for i in range(num_trajs):
        all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0 for k in range(n_agents)]
        for k in range(n_agents):
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
        obss = base_traj[i]['all_ob']
        step = 0
        # obs = env.reset()
        obs_link = [np.array([i + 1, 1, 1]).reshape(1, -1, ) for i in range(num_links)]
        obs = [np.array([ob, 0, 0]).reshape(1, -1, ) for ob in obss[step]]
        obs = obs_link + obs
        # obs = [ob[None,:] for ob in obs]
        action = [np.zeros([1]) for _ in range(n_agents)]
        step = 0
        done = False
        while not done:
            action, _, _ = model.step(obs, action)
            actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
            # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]

            for k in range(n_agents):
                all_ob[k].append(obs[k])
                all_ac[k].append(actions_list[k])
            all_agent_ob.append(np.concatenate(obs, axis=1))
            # obs, rew, done, _ = env.step(actions_list)
            for k in range(num_links):
                if action[k][0] == 1:
                    obs[k][0][2] += 1
                elif action[k][0] == 2:
                    obs[k][0][2] += 2
                elif action[k][0] == 3:
                    obs[k][0][2] -= 1
                elif action[k][0] == 3:
                    obs[k][0][2] -= 2

            for k in range(num_links, n_agents):
                if obs[k][0][0] > 0:
                    if action[k][0] > 4 and action[k][0] <= 8:
                        obs[k][0][0] = dependency[int(obs[k][0][0])][int(action[k][0]) - 1]
                else:
                    if obss[step + 1][k - num_links] > 0:
                        action[k][0] = 12
                        obs[k][0][0] = obss[step + 1][k - num_links]
                if obs[k][0][0] > 0 and obs[k][0][0] < 144:
                    aa = int(obs[k][0][0])
                    obs[k][0][2] = obs[aa - 1][0][2]

                # all_rew[k].append(rew[k])
                # ep_ret[k] += rew[k]
            # obs = [ob[None, :] for ob in obs]
            step += 1

            # if image:
            #     img = env.render(mode='rgb_array')
            #     images.append(img[0])
            #     time.sleep(0.02)
            if step == max_steps:
                done = True
                step = 0
            else:
                done = False

        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])

        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob
        }

        sample_trajs.append(traj_data)
        # print('traj_num', i, 'expected_return', ep_ret)

        for k in range(n_agents):
            avg_ret[k].append(ep_ret[k])

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
    return sample_trajs