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
from sandbox.mack.policies import MaskedCategoricalPolicy, GaussianPolicy, LSTMGaussianPolicy, MASKATTGaussianPolicy
from rl import bench
# import imageio
import pickle as pkl
import copy
import math


# dependency = np.load('dependency_dd4.npy', allow_pickle = True) ####

# @click.command()
# @click.option('--env', type=click.STRING)
# @click.option('--image', is_flag=True, flag_value=True)

def create_env(env_id, scenario_test):
    env = make_env.make_env(env_id, scenario_test)
    env.seed(10)
    # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
    set_global_seeds(10)
    return env


def makeModel(env_id, scenario_test):
    tf.reset_default_graph()

    # env_id = 'trj_network'

    # def create_env():
    #     env = make_env.make_env(env_id)
    #     env.seed(10)
    #     # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
    #     set_global_seeds(10)
    #     return env

    env = create_env(env_id, scenario_test)
    # path = 'multi-agent-trj/logger/airl/trj_network/decentralized/s-200/l-0.1-b-500-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-2/m_0'+mid
    # path = 'multi-agent-trj/logger/gail/trj_network/decentralized/s-200/l-0.01-b-1000-d-0.1-c-500/seed-1/m_0' + mid

    # print(path)
    n_agents = len(env.action_space)

    ob_space = env.observation_space
    ac_space = env.action_space

    # if env_id == 'trj_network':
    #     identical=[False] + [True for _ in range(11)] + [False] + [True for _ in range(49)]
    # if env_id == 'trj_network_drone':
    #     identical = [False] + [True for _ in range(156)] + [False] + [True for _ in range(199)]
    # if env_id == 'trj_intersection':
    # identical=[False] + [True for _ in range(8)] + [False] + [True for _ in range(12)] + [False] + [True for _ in range(3)] + [False] + [True for _ in range(1)]
    # identical=[False] + [True for _ in range(6)] + [False] + [True for _ in range(2)] + [False] + [True for _ in range(3)] + [False] + [True for _ in range(3)]
    identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]

    print('observation space')
    print(ob_space)
    print('action space')
    print(ac_space)

    # n_actions = [action.n for action in ac_space]

    make_model = lambda: Model(
        MASKATTGaussianPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=10, nsteps=500,
        nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.1, max_grad_norm=0.5, kfac_clip=0.001,
        lrschedule='linear', identical=identical)

    model = make_model()
    return model
def get_dis(path, model, env_id, mid, scenario_test):
    print("load model from", path)
    # model.load(path+'m_00001')
    env = create_env(env_id, scenario_test)

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    # n_actions = [action.n for action in ac_space]
    # tf.reset_default_graph()
    # 创建新的会话
    # with tf.Session() as sess:
    discriminator = [
        Discriminator(model.sess, ob_space, ac_space,
                      state_only=True, discount=0.99, nstack=1, index=k, disc_type='decentralized',
                      scope="Discriminator_%d" % k,  # gp_coef=gp_coef,
                      total_steps=1e7 // (10 * 500),
                      lr_rate=0.1, l2_loss_ratio=0.1) for k in range(n_agents)
    ]
    model.sess.run(tf.global_variables_initializer())
    print('render里的discriminator：', discriminator)
    # 加载Discriminator模型参数
    for n_v in range(8):
        did = str(n_v) + '_0' + str(mid)
        path2 = path + 'd_' + did
        # tf.reset_default_graph()
        discriminator[n_v].load(path2)
    # did='0_00001'
    # path2 = path +'d_'+did
    # path = 'multi-agent-trj/logger/airl/trj_network_drone/decentralized/s-15/l-0.1-b-50-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-21/d_'+did

    return discriminator

def render(path, model, env_id, mid, scenario_test):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id, scenario_test)

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    # n_actions = [action.n for action in ac_space]

    images = []
    sample_trajs = []
    num_trajs = 1  #### 是不是跑两次？取两次数据
    max_steps = 230  # 484 # 179    #### 每个agent的步长可以限制185 但是场景的步长不要限制了
    avg_ret = [[] for _ in range(n_agents)]

    for i in range(num_trajs):
        all_attention_weight_spatial, all_attention_weight_temporal, all_ob_lstm, all_ob, all_agent_ob, all_ac, \
        all_rew, all_social, ep_ret, all_rew_n_social_generate, all_collide_situation = [], [], [], [], [], [], [], [], [0 for k in range(n_agents)], [], []
        for k in range(n_agents):
            all_ob_lstm.append([])
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
            all_attention_weight_spatial.append([])
            all_attention_weight_temporal.append([])
            all_social.append([])  # 存储社会倾向
            all_rew_n_social_generate.append([])
            all_collide_situation.append([])

            # ep_ret.append([])
        # obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm
        obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, rew_n_social_generate, collide_situation = env.reset(scenario_test)  # 初始观测值输入  # , reset_info
        print('env.reset(scenario_test) iniobs的x和y:',ini_obs[3][:2]) # (8, 18) (8, 18)
        print('collide_situation:',np.shape(collide_situation))  # (8,)
        action = [np.zeros(2) for _ in range(n_agents)]
        step = 0
        done_eva = False
        # 在循环之前创建深拷贝
        obs_lstm = [copy.deepcopy(ob_lstm[None, :]) for ob_lstm in obs_lstm]
        obs = [copy.deepcopy(ob[None, :]) for ob in obs]
        ini_obs = [copy.deepcopy(ini[None, :]) for ini in ini_obs]
        action = [copy.deepcopy(ac[None, :]) for ac in action]
        ini_step_n = [copy.deepcopy(ini_step[None, :]) for ini_step in ini_step_n]
        ini_obs_lstm = [copy.deepcopy(ini_ob_lstm[None, :]) for ini_ob_lstm in ini_obs_lstm]
        # print('ini_step_n:', np.shape(ini_step_n), ini_step_n[0][0][0])  # (8, 1, 1) 0.0
        # print('ini_obs为：', np.shape(ini_obs), ini_obs[0][0])  # (8, 1, 18)
        # print('action为：', np.shape(action)) # (8, 1, 2)
        # print('obs为：', np.shape(obs)) # (8, 1, 18)
        trj_GO_STEP = [np.array([0 for _ in range(n_agents)])]  # (1, 8)


        while not done_eva:
            tstart = time.time()  # 开始时间
            tstart1 = time.time()  # 开始时间
            ini_update_inf = [np.array([0 for _ in range(8)]) for k in range(1)]
            for k in range(n_agents):  # 循环遍历每个智能体，将当前时间步骤的观察、动作、值函数估计、完成状态等数据添加到相应的数据列表中。
                trj_go_step = step - ini_step_n[k][0][0]
                trj_GO_STEP[0][k] = trj_go_step
                ini_step_k = ini_step_n[k][0][0]  # ini_steps的shape是 (18, 1, 1)
                # print('step:',step,'agent2的inistep:',ini_step_n[2][0][0]) # agent2的inistep: 45.0
                if step == ini_step_k:  # ini_steps的shape是【8，10，1】 当前这个时刻的步长就是agent开始的时刻
                    # print('ini_obs:',len(ini_obs[k][0]))
                    obs_lstm[k][0] = ini_obs_lstm[k][0]
                    obs[k][0] = ini_obs[k][0]  # 将obs设置为初始的obs
                    action[k][0] = np.zeros(2)
                    ini_update_inf[0][k] = True
                    # print('初始化obs，开始仿真！', obs[k][0], ini_obs[k][0], k, step, ini_step_k, ini_obs[2][0])
                elif step < ini_step_k:  # ini_steps的shape是【16，10，1】 当前这个时刻的步长就是agent开始的时刻
                    # print('step<ini_step_k agent:', k, 'ini_step:', ini_step_k, 'step', step, 'agent1:', ini_obs[2][0])
                    obs[k][0] = np.zeros(57)  # 将obs设置为0
                    action[k][0] = np.zeros(2)  # 将action设置为初始的0
                    obs_lstm[k][0][:] = 0  # 将obs_lstm设置为0
                    ini_update_inf[0][k] = False
                    # print('step<ini_step_k agentfuzhiobs:', k, 'ini_step:', ini_step_k, 'step', step, 'agent1:', ini_obs[2][0], obs[2][0])
                # print('agent:', k, 'ini_step:', ini_step_k, 'step', step, 'agent3:', obs[3][0])

            # 对这一步，每一个环境下的每一个agent的状态都更新，到时候根据具体的情况判断是否更新每个cpu
            ini_obs_old_list = []
            # print('ini_update_inf:', np.shape(ini_update_inf))
            # print('ini_obs_old_list中agent3的iniobs:',ini_obs[3][0][:2])
            for i in range(1):
                ini_obs_old_list.append(
                    [np.concatenate((action[k_][0], np.array([step]),
                                            np.array([trj_GO_STEP[0][k_]]),
                                            np.array([ini_step_n[k_][0][0]]), ini_obs[k_][0],
                                            np.array([ini_update_inf[i][k_]]))) for k_ in
                     range(8)])
            # print('ini_obs_old_list:', np.shape(ini_obs_old_list))  # (1，8，63)
            obs_lstm_nowstep, obs_nowstep, collide_situation_nowstep = env.ini_obs_update(
                ini_obs_old_list[0])  # 初始观测值输入  # , reset_info
            tend1 = time.time()  # 开始时间
            print('ini_obs_update花费的时间：',tend1-tstart1)

            # print('obs_lstm_nowstep:', np.shape(obs_lstm_nowstep), type(obs_lstm_nowstep),
            #       'obs_nowstep:', np.shape(obs_nowstep),
            #       type(obs_nowstep), np.shape(collide_situation_nowstep))
            # obs_lstm_nowstep: (8, 21, 57) <class 'list'> obs_nowstep: (8, 57) <class 'list'>
            # print('scenario_test:',scenario_test, 'agent3,step:', step, 'obs_nowstep的x和y:',obs_nowstep[3][:2])

            # print('self.obs_lstm:', type(obs_lstm), np.shape(obs_lstm))   # self.obs_lstm: <class 'list'> (8, 1, 21, 57)
            # print('self.obs:', type(obs), np.shape(obs))  # (8, 1, 57)
            tstart2 = time.time()  # 开始时间
            for i in range(0):
                for k in range(8):
                    if ini_update_inf[i][k] == True:
                        for j in range(len(obs_lstm)):
                            obs_lstm[j][i] = obs_lstm_nowstep[j]  # 对于这个cpu环境，每个agent的状态都得更新
                            obs[j][i] = obs_nowstep[j]  # 对于这个cpu环境，每个agent的状态都得更新
                        action[k][i] = np.zeros(2)  # 对于这个cpu环境，只需要刚开始进入交叉口的agent的动作进行更新

            # ob_lstm, ob, av

            action, _, _, atten_weights_spatial, atten_weights_temporal = model.step(obs_lstm, obs, action)
            # print('循环的时候model.step之后ini_obs为：', np.shape(ini_obs), obs[3][0])  # (8, 1, 18)
            # print('step:', step, '运行完model.step之后：',np.shape(obs_lstm), np.shape(obs), np.shape(action), np.shape(ini_step_n),
            #       np.shape(ini_obs), np.shape(atten_weights_spatial), np.shape(atten_weights_temporal))
            # (8, 1, 21, 57) (8, 1, 57) (8, 1, 2) (8, 1, 1) (8, 1, 57) (8, 21, 1, 10, 10) (8,1,21,21)
            tend2 = time.time()  # 开始时间
            print('获得动作model.step花费的时间：', tend2 - tstart2)
            for k_ in range(n_agents):
                all_ob[k_].append(obs[k_][0])
                # all_ac[k_].append(action[k_][0]) # 到后面再更新
                all_ob_lstm[k_].append(obs_lstm[k_][0])
                all_attention_weight_spatial[k_].append(atten_weights_spatial[k_])
                all_attention_weight_temporal[k_].append(atten_weights_temporal[k_])
                all_collide_situation[k_].append(collide_situation_nowstep[k_])
            # print('step:',step,'agent3ini_step:',ini_step_n[3][0][0],'agent3action:',action[3], all_ac[3])

            action2 = []

            tstart3 = time.time()  # 开始时间
            env_go_step = step
            # env_GO_STEP[0] = env_go_step  # self.env_GO_STEP (10,1) 环境累计前进了多少步
            # for k__ in range(n_agents):
            #     trj_go_step = step - ini_step_n[k__][0][0]
            #     if trj_go_step > 0:
            #         trj_GO_STEP[0][k__] = trj_go_step  # self.trj_GO_STEP (10,18) 环境中的轨迹前进了多少步
            #     else: # if trj_go_step <= 0:
            #         trj_GO_STEP[0][k__] = trj_go_step
            #
            #     if k__ <= 2:
            #         # 左转车
            #         acc = min(max(action[k__][0][0], -1), 1)
            #         delta_theta = min(max(action[k__][0][1], -1.3), 1.2)  # 改为最大是0.5（对应真实的是2.5）
            #         action[k__][0][0] = acc
            #         action[k__][0][1] = delta_theta
            #     else:
            #         # 直行车
            #         acc = min(max(action[k__][0][0], -1), 1)
            #         delta_theta = min(max(action[k__][0][1], -1), 1)
            #         action[k__][0][0] = acc
            #         action[k__][0][1] = delta_theta

            # for kkk in range(8):
            #     if kkk <= 2:
            #         if round(action[kkk][0][1], 1) > 1.2 or round(action[kkk][0][1],1) < -1.3:
            #             print('reder左转车的转向角为：', kkk, action[kkk][0][1],
            #                   ((2.8 * (action[kkk][0][1] + 1)) / 2) - 0.3)
            #     else:
            #         if action[kkk][0][1] > 1 or action[kkk][0][1] < -1:
            #             print('render直行车的转向角为：', kkk, action[kkk][0][1],
            #                   ((2.4 * (action[kkk][0][1] + 1)) / 2) - 1.2)

            action2.append([np.concatenate((action[k][0], np.array([env_go_step]),
                                            np.array([trj_GO_STEP[0][k]]),
                                            np.array([ini_step_n[k][0][0]]), ini_obs[k][0])) for k in
                            range(n_agents)])
            # print('action2:', np.shape(action2))  # (1, 8, 27)
            all_agent_ob.append(np.concatenate(obs, axis=1))
            # ob, reward, done, info, ini_step, ini_ob
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs

            rewards_panbieqi = []  # rewards = [] 和 report_rewards = []：创建用于存储判别器奖励的空列表。
            social_panbieqi = []  # 存放社会倾向 -π/2~π/2
            path_prob = np.zeros(n_agents)
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs, actions_new_n
            obs_lstm, obs, rew, dones, _, ini_step_n, ini_ob, action_new, rew_social_generate, collide_situation = env.step(action2[0])
            tend3 = time.time()  # 开始时间
            print('获得位置env.step花费的时间：', tend3 - tstart3)
            tend = time.time()  # 开始时间
            print('得到下一时刻所有车辆的obs花费的时间：',tend-tstart)
            # print('循环的时候env.step之后ini_obs为：', np.shape(ini_obs), obs[3])  # (8, 1, 56)
            # print('运行完step:', np.shape(obs), np.shape(rew), np.shape(dones), np.shape(ini_step_n),
            #       np.shape(ini_obs), np.shape(action_new), np.shape(rew_social_generate), np.shape(collide_situation))
            # (8, 57) (8,) (8,) (8, 1) (8, 1, 57) (8, 2) (8, 4) (8,)
            action_new = [action1_new[None, :] for action1_new in action_new]
            # print('运行完stepaction_new:', np.shape(action_new))  # (8, 1, 2)
            action = action_new



            for k in range(n_agents):
                if dones[k]:  # dones的shape为(8, 10)
                    # 第ni线程的第k个agent的done为True，说明这个agent在执行完动作之后新的时刻会驶出交叉口大边界了，或者上一时刻的点和终点之间的距离已经小于0.01m了
                    obs[k] = obs[k] * 0.0  # 处理完成状态，如果一个智能体完成了，将其观察数据置为零。
                    action[k] = action[k] * 0.0
                    obs_lstm[k] = obs_lstm[k] * 0.0
            # print('social_panbieqi:',np.shape(social_panbieqi),np.sin(social_panbieqi[0]),np.cos(social_panbieqi[0]))
            # print('rewards_panbieqi:',np.shape(rewards_panbieqi))
            # for kkkk in range(8):
            #     if kkkk <= 2:
            #         if round(action[kkkk][0][1], 1) > 1.2 or round(action[kkkk][0][1],1) < -1.3:
            #             print('reder处理完之后左转车的转向角为：', kkkk, action[kkkk][0][1],
            #                   ((2.8 * (action[kkkk][0][1] + 1)) / 2) - 0.3)
            #     else:
            #         if action[kkkk][0][1] > 1 or action[kkkk][0][1] < -1:
            #             print('render处理完之后直行车的转向角为：', kkkk, action[kkkk][0][1],
            #                   ((2.4 * (action[kkkk][0][1] + 1)) / 2) - 1.2)

            # print('rew:',np.shape(rew))  # rew: (8,)
            # print('ep_ret:',np.shape(ep_ret))  # ep_ret: (8,)

            for k_ in range(n_agents):
                all_ac[k_].append(action[k_][0])

            for k in range(n_agents):
                all_rew[k].append(rew[k])  # +rewards_panbieqi[k]
                ep_ret[k] += rew[k]
                # all_social[k].append(social_panbieqi[k])

            obs_lstm = [ob_lstm[None, :] for ob_lstm in obs_lstm]
            obs = [ob[None, :] for ob in obs]
            ini_step_n = [[[ini_step[0]]] for ini_step in ini_step_n]
            step += 1

            # print('运行完step并处理之后:', np.shape(all_ob), np.shape(obs), np.shape(rew), np.shape(dones),
            #       np.shape(ini_step_n), np.shape(ini_obs), np.shape(obs_lstm), np.shape(all_ob_lstm), dones)
            # (8, step, 57) (8, 1, 57) (8,) (8,) (8, 1, 1) (8, 1, 57) (8, 1, 21, 57) (8, step, 21, 57)
            # [False, False, True, False, False, False, True, True]
            # all_ob:(8, step, 56) obs:(8, 1, 56) rew:(8,) dones:(8,) ini_step_n:(8, 1, 1)
            # ini_obs:(8, 1, 56) obs_lstm:(8, 1, 21, 56) all_ob_lstm:(8, step, 21, 56)

            #if dones == [True,True,True,True,True,True,True,True]:
            #    done_eva = True
            #   step = 0
            if step == max_steps:
                done_eva = True
                step = 0
            else:
                done_eva = False

        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])
            all_ob_lstm[k] = np.squeeze(all_ob_lstm[k])
            all_attention_weight_spatial[k] = np.squeeze(all_attention_weight_spatial[k])  # 是对 all_attention_weight_spatial[k] 进行操作，将其维度为1的轴去除
            all_attention_weight_temporal[k] = np.squeeze(all_attention_weight_temporal[k])
            all_collide_situation[k] = np.squeeze(all_collide_situation[k])

        print('all_ob:', np.shape(all_ob))
        print('all_ob_lstm:', np.shape(all_ob_lstm))
        print('all_attention_weight_spatial:', np.shape(all_attention_weight_spatial))  # (8, 230, 21, 10, 10)
        print('all_attention_weight_temporal:', np.shape(all_attention_weight_temporal))  # (8, 230, 21, 21)
        print('all_collide_situation:', np.shape(all_collide_situation))  # (8, 230)
        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob,'all_attention_weight_spatial': all_attention_weight_spatial,
            'all_attention_weight_temporal': all_attention_weight_temporal,
            'all_collide_situation': all_collide_situation
        }

        sample_trajs.append(traj_data)
        # print('traj_num', i, 'expected_return', ep_ret)

        for k in range(n_agents):
            avg_ret[k].append(ep_ret[k])

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]),np.mean(all_rew[k]))
    return sample_trajs

    # # actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
    # # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]
    # # action2 = []
    # # action2.append([np.concatenate((action[k][0], np.array([step]))) for k in range(n_agents)])
    # action2 = [np.squeeze(ac) for ac in action]
    # # print('评估时的action:',action2,'第几步:',step)
    # for k in range(n_agents):
    #     all_ob[k].append(obs[k])
    #     all_ac[k].append(action[k])
    #         all_agent_ob.append(np.concatenate(obs, axis=1))
    #         obs, rew, done, _ = env.step(action2)
    #         for k in range(n_agents):
    #             all_rew[k].append(rew[k])
    #             ep_ret[k] += rew[k]
    #         obs = [ob[None, :] for ob in obs]
    #         step += 1
    #
    #         # if image:
    #         #     img = env.render(mode='rgb_array')
    #         #     images.append(img[0])
    #         #     time.sleep(0.02)
    #         # if step == max_steps or True in done:
    #         if step == max_steps:
    #             done_eva = True
    #             step = 0
    #         else:
    #             done_eva = False
    #
    #     for k in range(n_agents):
    #         all_ob[k] = np.squeeze(all_ob[k])
    #
    #     all_agent_ob = np.squeeze(all_agent_ob)
    #     traj_data = {
    #         "ob": all_ob, "ac": all_ac, "rew": all_rew,
    #         "ep_ret": ep_ret, "all_ob": all_agent_ob
    #     }
    #
    #     sample_trajs.append(traj_data)
    #     # print('traj_num', i, 'expected_return', ep_ret)
    #
    #     for k in range(n_agents):
    #         avg_ret[k].append(ep_ret[k])
    #
    # print(path)
    # for k in range(n_agents):
    #     print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
    # return sample_trajs

    # images = np.array(images)
    # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
    # if image:
    #     print(images.shape)
    #     imageio.mimsave(path + '.mp4', images, fps=25)
    # return sample_trajs

def render_discrimination(path, model, env_id, mid):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id)

    # 读取discrimination
    path_d = r'E:/wsh-科研/nvn_xuguan_sind/sinD_nvn_xuguan_9jiaohu_ATT-GPU-规则在内-social/MA_Intersection_straight' \
         r'/multi-agent-irl/irl/mack/multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
         r'/v9/l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0/seed-13/'

    discriminator = get_dis(path_d, model, env_id, mid)  # 获得判别器
    print('discriminator', discriminator, np.shape(discriminator))

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    # n_actions = [action.n for action in ac_space]

    images = []
    sample_trajs = []
    num_trajs = 1  #### 是不是跑两次？取两次数据
    max_steps = 230  # 484 # 179    #### 每个agent的步长可以限制185 但是场景的步长不要限制了
    avg_ret = [[] for _ in range(n_agents)]

    for i in range(num_trajs):
        all_attention_weight, all_ob_lstm, all_ob, all_agent_ob, all_ac, all_rew, all_social, ep_ret = [], [], [], [], [], [], [], [0 for k in range(n_agents)]
        for k in range(n_agents):
            all_ob_lstm.append([])
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
            all_attention_weight.append([])
            all_social.append([])  # 存储社会倾向
            # ep_ret.append([])
        # obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm
        obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm = env.reset()  # 初始观测值输入  # , reset_info
        # print('第几次:',i,'评估时的初始值输入:',np.shape(obs),np.shape(ini_obs)) # (8, 18) (8, 18)
        action = [np.zeros(2) for _ in range(n_agents)]
        step = 0
        done_eva = False
        # 在循环之前创建深拷贝
        obs_lstm = [copy.deepcopy(ob_lstm[None, :]) for ob_lstm in obs_lstm]
        obs = [copy.deepcopy(ob[None, :]) for ob in obs]
        ini_obs = [copy.deepcopy(ini[None, :]) for ini in ini_obs]
        action = [copy.deepcopy(ac[None, :]) for ac in action]
        ini_step_n = [copy.deepcopy(ini_step[None, :]) for ini_step in ini_step_n]
        ini_obs_lstm = [copy.deepcopy(ini_ob_lstm[None, :]) for ini_ob_lstm in ini_obs_lstm]
        # print('ini_step_n:', np.shape(ini_step_n), ini_step_n[0][0][0])  # (8, 1, 1) 0.0
        # print('ini_obs为：', np.shape(ini_obs), ini_obs[0][0])  # (8, 1, 18)
        # print('action为：', np.shape(action)) # (8, 1, 2)
        # print('obs为：', np.shape(obs)) # (8, 1, 18)
        trj_GO_STEP = [np.array([0 for _ in range(n_agents)])]  # (1, 8)

        while not done_eva:

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

            def Cal_GT_crash(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                             Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                             Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

                # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
                dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
                if dis_between_agent_jiaohu <= 15:
                    # 计算和这个车辆的GT
                    agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                    neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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
                        GT_value = -2  # 非常不安全
                        dis_min = 0
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
                                GT_value = 1  # 安全
                                dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
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
                                    # 2.1 AGENT 会把交互对象看做 有冲突的对象;交互对象也会把agent看做 有冲突的对象
                                    # 判断当agent到交点的时候，neig在哪，如果撞了，GT_value=-1,否则None
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录快车走到交点的路程中，两车之间轨迹点的距离，步长0.5s（不考虑车宽，因为是否碰撞上已经考虑了）
                                    if t_agent < t_neig:
                                        # agent先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_agent > 23:
                                            t_agent = 23
                                        else:
                                            t_agent = t_agent
                                        # try:
                                        time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_agent:
                                            time = np.arange(0, t_agent + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_agent, 0.5)
                                            time = np.append(time, t_agent)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离
                                    else:
                                        # neig先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_neig > 23:
                                            t_neig = 23
                                        else:
                                            t_neig = t_neig
                                        # try:
                                        time_n_0 = int(t_neig / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_neig:
                                            time = np.arange(0, t_neig + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_neig, 0.5)
                                            time = np.append(time, t_neig)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                elif dot_product_agent >= 0 and dot_product_neig < 0:
                                    # 2.2 agent把neig看做冲突对象，但是neig不把agent看做冲突对象，仍然需要判断在agent到冲突点的路程中，是否会发生碰撞，以及距离的大小
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录agent车走到交点的路程中，两车之间轨迹点的距离（不考虑车宽，因为是否碰撞上已经考虑了）
                                    # agent会到达冲突点，neig不会到达冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞

                                    if t_agent > 23:
                                        t_agent = 23
                                    else:
                                        t_agent = t_agent
                                    # try:
                                    time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_agent:
                                        time = np.arange(0, t_agent + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_agent, 0.5)
                                        time = np.append(time, t_agent)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t, Agent_y_t,
                                                                                                veh_length, veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t, Jiaohu_y_t,
                                                                                               veh_length, veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                else:
                                    # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                                    # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                                    GT_value = 0
                                    dis_min = 100000



                        else:
                            # neig不是前车 2.4.2
                            GT_value = 0  # 不看做交互，因为当前时刻已经没有碰撞了
                            dis_min = 100000


                else:
                    GT_value = 0  # 不交互
                    dis_min = 100000

                return GT_value, dis_min

            def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                          Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                          Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

                # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
                dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
                if dis_between_agent_jiaohu <= 15:
                    # 计算和这个车辆的GT
                    agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                    neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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

            for k in range(n_agents):  # 循环遍历每个智能体，将当前时间步骤的观察、动作、值函数估计、完成状态等数据添加到相应的数据列表中。
                ini_step_k = ini_step_n[k][0][0]  # ini_steps的shape是 (18, 1, 1)
                # print('step:',step,'agent2的inistep:',ini_step_n[2][0][0]) # agent2的inistep: 45.0
                if step == ini_step_k:  # ini_steps的shape是【8，10，1】 当前这个时刻的步长就是agent开始的时刻
                    # print('ini_obs:',len(ini_obs[k][0]))
                    obs_lstm[k][0] = ini_obs_lstm[k][0]
                    obs[k][0] = ini_obs[k][0]  # 将obs设置为初始的obs
                    action[k][0] = np.zeros(2)
                    # print('初始化obs，开始仿真！', obs[k][0], ini_obs[k][0], k, step, ini_step_k, ini_obs[2][0])
                elif step < ini_step_k:  # ini_steps的shape是【16，10，1】 当前这个时刻的步长就是agent开始的时刻
                    # print('step<ini_step_k agent:', k, 'ini_step:', ini_step_k, 'step', step, 'agent1:', ini_obs[2][0])
                    obs[k][0] = np.zeros(57)  # 将obs设置为0
                    action[k][0] = np.zeros(2)  # 将action设置为初始的0
                    obs_lstm[k][0][:] = 0  # 将obs_lstm设置为0
                    # print('step<ini_step_k agentfuzhiobs:', k, 'ini_step:', ini_step_k, 'step', step, 'agent1:', ini_obs[2][0], obs[2][0])
                # print('agent:', k, 'ini_step:', ini_step_k, 'step', step, 'agent3:', obs[3][0])

            # ob_lstm, ob, av
            action, _, _, atten_weights = model.step(obs_lstm, obs, action)
            # print('循环的时候model.step之后ini_obs为：', np.shape(ini_obs), obs[3][0])  # (8, 1, 18)
            print('step:', step, '运行完model.step之后：',np.shape(obs_lstm), np.shape(obs), np.shape(action), np.shape(ini_step_n),
                  np.shape(ini_obs), np.shape(atten_weights))
            # (8, 1, 21, 57) (8, 1, 57) (8, 1, 2) (8, 1, 1) (8, 1, 57) (8, 21, 1, 10, 10)

            for k in range(n_agents):
                all_ob[k].append(obs[k][0])
                # all_ac[k].append(action[k][0])
                all_ob_lstm[k].append(obs_lstm[k][0])
                all_attention_weight[k].append(atten_weights[k])

            # print('step:',step,'agent3ini_step:',ini_step_n[3][0][0],'agent3action:',action[3], all_ac[3])

            action2 = []

            env_go_step = step
            # env_GO_STEP[0] = env_go_step  # self.env_GO_STEP (10,1) 环境累计前进了多少步
            for k in range(n_agents):
                trj_go_step = step - ini_step_n[k][0][0]
                if trj_go_step > 0:
                    trj_GO_STEP[0][k] = trj_go_step  # self.trj_GO_STEP (10,18) 环境中的轨迹前进了多少步
                else:  # if trj_go_step <= 0:
                    trj_GO_STEP[0][k] = trj_go_step

                if k <= 2:
                    # 左转车
                    acc = min(max(action[k][0][0], -1), 1)
                    delta_theta = min(max(action[k][0][1], -1.3), 1.2)  # 改为最大是0.5（对应真实的是2.5）
                    action[k][0][0] = acc
                    action[k][0][1] = delta_theta
                else:
                    # 直行车
                    acc = min(max(action[k][0][0], -1), 1)
                    delta_theta = min(max(action[k][0][1], -1), 1)
                    action[k][0][0] = acc
                    action[k][0][1] = delta_theta

            for k in range(8):
                if k <= 2:
                    if action[k][0][1] > 1.2 or action[k][0][1] < -1.3:
                        print('reder左转车的转向角为：', k, action[k][0][1],
                              ((2.8 * (action[k][0][1] + 1)) / 2) - 0.3)
                else:
                    if action[k][0][1] > 1 or action[k][0][1] < -1:
                        print('render直行车的转向角为：', k, action[k][0][1],
                              ((2.4 * (action[k][0][1] + 1)) / 2) - 1.2)

            action2.append([np.concatenate((action[k][0], np.array([env_go_step]),
                                            np.array([trj_GO_STEP[0][k]]),
                                            np.array([ini_step_n[k][0][0]]), ini_obs[k][0])) for k in
                            range(n_agents)])
            # print('action2:', np.shape(action2))  # (1, 8, 27)
            all_agent_ob.append(np.concatenate(obs, axis=1))
            # ob, reward, done, info, ini_step, ini_ob
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs
            re_obs_lstm = obs_lstm

            obs_lstm, obs, rew, dones, _, ini_step_n, ini_ob, action_new = env.step(action2[0])
            # print('循环的时候env.step之后ini_obs为：', np.shape(ini_obs), obs[3])  # (8, 1, 56)
            print('运行完step:', np.shape(obs), np.shape(rew), np.shape(dones), np.shape(ini_step_n), np.shape(ini_obs), np.shape(action_new))

            action_new = [action1_new[None, :] for action1_new in action_new]
            action = action_new

            for k in range(n_agents):
                if dones[k]:  # dones的shape为(8, 10)
                    # 第ni线程的第k个agent的done为True，说明这个agent在执行完动作之后新的时刻会驶出交叉口大边界了，或者上一时刻的点和终点之间的距离已经小于0.01m了
                    obs[k] = obs[k] * 0.0  # 处理完成状态，如果一个智能体完成了，将其观察数据置为零。
                    action[k] = action[k] * 0.0
                    obs_lstm[k] = obs_lstm[k] * 0.0
            # print('social_panbieqi:',np.shape(social_panbieqi),np.sin(social_panbieqi[0]),np.cos(social_panbieqi[0]))
            # print('rewards_panbieqi:',np.shape(rewards_panbieqi))

            for k in range(n_agents):
                all_ac[k].append(action[k][0])

            obs_lstm = [ob_lstm[None, :] for ob_lstm in obs_lstm]
            obs = [ob[None, :] for ob in obs]
            ini_step_n = [[[ini_step[0]]] for ini_step in ini_step_n]
            step += 1

            rewards_panbieqi = []  # rewards = [] 和 report_rewards = []：创建用于存储判别器奖励的空列表。
            social_panbieqi = []  # 存放社会倾向 -π/2~π/2
            path_prob = np.zeros(n_agents)
            for k in range(n_agents):
                if k <= 2:
                    direction_agent = 'left'
                else:
                    direction_agent = 'straight'

                # 调用判别器的 get_reward 方法计算判别器奖励，将其添加到 rewards 列表中。这个奖励通常用于更新策略。
                # 没整明白 !
                # print('判别器的输入格式：', 're_obs_lstm:', np.shape(re_obs_lstm[k]), type(re_obs_lstm[k]))  # (10, 21, 46)
                # print('判别器的输入格式：', 're_actions:', np.shape(re_actions[k]), type(re_actions[k]))  # (10, 2)
                # print('判别器的输入格式：', 're_obs_next:', np.shape(re_obs_next_lstm[k]), type(re_obs_next_lstm[k]))  # (10, 21, 46)
                # print('判别器的输入格式：', 're_path_prob:', np.shape(re_path_prob[k]), type(re_path_prob[k]))  # 0.0
                batch_num = np.shape(re_obs_lstm[k])[0]
                # print('判别器的输入batch_num:',batch_num)
                # 计算利己和利他所需要的参数
                # 计算利己奖励和利他奖励，然后利用网络学习参数φ，cos(φ)=利己倾向，sin(φ)利他倾向
                rew_input_fuyuan = re_obs_lstm[k]

                rew_social_allbatch = []  # 存放这一个agent 所有batch的参数

                # 利己性参数-速度, 针对每一个batch来计算
                for i_batch in range(batch_num):
                    # 改成当前时刻应该更好，因为是当前时刻的奖励，过去的已经无法改变了。过去的状态可以看做是影响社交倾向的因素
                    # 如果是考虑历史数据的话，对于一些当前时刻无效，但历史时刻有效的数据来说，奖励就没有实际含义了
                    # 其实也可以有实际含义。再想想。还是不考虑了
                    if rew_input_fuyuan[i_batch][20][0] != 0:
                        use_GT = []  # 存放这个ibatch的主要交互对象的GT
                        # speed = np.sqrt(rew_input_fuyuan[i_batch][20][2] ** 2 + rew_input_fuyuan[i_batch][20][3] ** 2)
                        pianyi_distance = rew_input_fuyuan[i_batch][20][-2]
                        # 计算和主要交互对象的GT
                        # 提取代理的状态和终点坐标
                        agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                        agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                        agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                        agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                        agent_angle_last = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # 上一个点的前进方向

                        agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                        # 避免碰撞
                        # 计算agent和周围最密切的三个交互对象的GT
                        # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
                        # 安全奖励
                        rew_GT = 0
                        # use_GT = []  # 存储和主要交互对象的GT  会有7+2个值，包括除主agent之外所有的agent和交互的landmark，即使没有这个对象，也会赋值为0
                        # 计算和除主车之外所有agent以及交互的landmark车辆的GT
                        # 把所有的agent都考虑
                        for agent_k_ in range(n_agents):
                            if agent_k_ != k:
                                # 这个agent不是我们正在计算的k
                                if agent_k_ <= 2:
                                    direction_jiaohu = 'left'
                                else:
                                    direction_jiaohu = 'straight'

                                rew_input_fuyuan_agent_k_ = re_obs_lstm[agent_k_]

                                if rew_input_fuyuan_agent_k_[i_batch][20][10] != 0:
                                    jiaohu_agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                                    jiaohu_agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                                    jiaohu_agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                                    jiaohu_agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                                    jiaohu_agent_angle_last = rew_input_fuyuan[i_batch][20][
                                                                  6] * 191 - 1  # 上一个点的前进方向

                                    jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                      agent_angle_last, direction_agent,
                                                                      jiaohu_agent_x, jiaohu_agent_y,
                                                                      jiaohu_agent_vx, jiaohu_agent_vy,
                                                                      jiaohu_agent_angle_last,
                                                                      direction_jiaohu)
                                    # if same_jiaohu_agent_GT_value is not None:
                                    use_GT.append(jiaohu_agent_GT_value)
                                else:  # 没有这个agent
                                    jiaohu_agent_x = -4
                                    jiaohu_agent_y = 14
                                    jiaohu_agentk_vx = -14
                                    jiaohu_agent_vy = -2
                                    jiaohu_agent_angle_last = -1
                                    jiaohu_agent_GT_value = None
                                    # dis_min = 100000
                                    use_GT.append(jiaohu_agent_GT_value)

                        # 左侧视野的landmark
                        if rew_input_fuyuan[i_batch][20][38] != 0:
                            delta_left_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][38] * 29 - 14
                            delta_left_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][39] * 30 - 15
                            delta_left_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][40] * 35 - 21
                            delta_left_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][41] * 16 - 5
                            left_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][53] * 360 - 90
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

                        # 右侧视野的landmark
                        if rew_input_fuyuan[i_batch][20][42] != 0:
                            delta_right_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][42] * 35 - 15
                            delta_right_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][43] * 29 - 15
                            delta_right_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][44] * 25 - 14
                            delta_right_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][45] * 17 - 7
                            right_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][54] * 360 - 90
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

                        # 计算角度波动
                        # 计算一些rew
                        # # 计算上个时刻，上上个时刻，上上上时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                        # if rew_input_fuyuan[i_batch][18][0] == 0:
                        #     # 没有上上上时刻的角度，所以判断上上个时刻的角度（一开始无论左转还是直行几乎都是直行的角度）
                        #     if rew_input_fuyuan[i_batch][19][0] == 0:
                        #         # 也没有上上时刻的角度
                        #         heading_angle_last3_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                        #         heading_angle_last2_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                        #     else:
                        #         # 有上上时刻的角度，上上上时刻的角度也用上上时刻的角度来代替
                        #         heading_angle_last3_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                        #         heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                        # else:
                        #     # 有上上上时刻的角度，所以也有上上个时刻的角度
                        #     heading_angle_last3_real = rew_input_fuyuan[i_batch][18][6] * 191 - 1  # [0,1]
                        #     heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                        #
                        # heading_angle_last1_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                        #
                        # # 计算平均值
                        # mean_value = np.mean(
                        #     [heading_angle_last1_real, heading_angle_last2_real, heading_angle_last3_real])
                        # # 计算每个数据与平均值的差的平方
                        # squared_differences = [(x - mean_value) ** 2 for x in
                        #                        [heading_angle_last1_real, heading_angle_last2_real,
                        #                         heading_angle_last3_real]]
                        # # 计算平方差的平均值
                        # mean_squared_difference = np.mean(squared_differences)
                        # # 计算标准差
                        # std_dev = np.sqrt(mean_squared_difference)
                        # if std_dev > 3:
                        #     rew_heading_std_bodong = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                        # else:
                        #     rew_heading_std_bodong = 1

                        # 计算steering angle正负来回变化带来的惩罚
                        # 上一时刻的转角，若超过均值的1个标准差，则给予惩罚之类的
                        penalty = 1  # 惩罚系数
                        delta_angle_last1 = rew_input_fuyuan[i_batch][20][56]
                        comfort_adj = 0  # 初始化转向角过大惩罚
                        if direction_agent == 'left':
                            left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                            left_delta_angle_last1_realmean = 1.085
                            left_delta_angle_last1_realstd = 0.702
                            if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                                comfort_adj = 0  # 不做惩罚

                            else:
                                dis_left_delta_angle_last1 = abs(
                                    left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                                if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                    comfort_adj = -1 * penalty
                                else:
                                    comfort_adj = -(
                                            dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                                    # 越靠近left_delta_angle_last1_realstd，惩罚越接近-1
                        else:
                            right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                            right_delta_angle_last1_realmean = 0.001
                            right_delta_angle_last1_realstd = 0.076
                            if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                                comfort_adj = 0  # 不做惩罚

                            else:
                                dis_right_delta_angle_last1 = abs(
                                    right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                                if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                    comfort_adj = -1 * penalty
                                else:
                                    comfort_adj = -(
                                                dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                                    # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1

                        # 利己-效率
                        rew_avespeed = agent_v / 6.8  # 除以85分位速度
                        # 利己-车道偏移
                        rew_lane_pianyi = pianyi_distance

                        # 利他-GT
                        # print("use_GT：",use_GT)  # 9个元素的list，例如[None, None, None, None, None, None, None, 0.32294667405015254, None]
                        use_GT_list = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                        print('use_GT_list:', use_GT_list)
                        if len(use_GT_list) != 0:
                            # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                            rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                            # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                            if rew_minGT <= 1.5:
                                # 归一化
                                normalized_data = (rew_minGT - 0) / (1.5 - 0)
                                # 映射到目标范围
                                rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                            elif 1.5 < rew_minGT < 3:
                                # 归一化
                                normalized_data = (rew_minGT - 1.5) / (3 - 1.5)

                                # 映射到目标范围
                                rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                            elif 3 <= rew_minGT <= 4:
                                # 归一化
                                normalized_data = (rew_minGT - 3) / (4 - 3)

                                # 映射到目标范围
                                rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                            elif rew_minGT > 4:
                                # 归一化
                                normalized_data = np.exp(-(1 / (rew_minGT - 4)))

                                # 映射到目标范围
                                rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                        else:
                            rew_minGT_mapped = 0
                            social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0

                        # 下面的代码是用Cal_GT_crash计算的利他策略方式。用了碰撞，但是经常导致利他倾向是0.可能是因为这种方法计算出来的很多rew_aveGT_mapped都是0
                        # print('判别器训练use_GT:', use_GT)  # 应该是一个包含9个数字的list，1代表安全，-1代表未来可能碰撞，0代表无交互，-2代表当前碰撞
                        # count_very_danger = sum(1 for item in use_GT if item[0] == -2)  # 统计 -2 的个数
                        # count_danger = sum(1 for item in use_GT if item[0] == -1)  # 统计 -1 的个数
                        # count_safe = sum(1 for item in use_GT if item[0] == 1)  # 统计 1 的个数
                        # count_nojiaohu = sum(1 for item in use_GT if item[0] == 0)  # 统计 0 的个数
                        #
                        # if count_safe == 0 and count_danger == 0 and count_very_danger == 0:
                        #     rew_aveGT_mapped = 0  # 无交互对象
                        # else:
                        #     if count_safe != 0:
                        #         # 找到第一个值为1的元素并统计第二个值
                        #         selected_items = [item[1] for item in use_GT if item[0] == 1]
                        #         # 计算第二个值的平均值
                        #         average_min_disvalue = sum(selected_items) / len(selected_items)
                        #         lita_cof = count_safe / (
                        #                     count_very_danger + count_safe + count_danger)  # 在有冲突的对象中安全交互的比例
                        #         rew_aveGT = lita_cof * average_min_disvalue
                        #         rew_aveGT_mapped = 1 - np.exp(-(rew_aveGT / 3))  # 除以3的目的是，尽可能的减缓较小距离就有较大奖励的情况
                        #         # 消解冲突的奖励归一化 0-1, 1是最考虑所有人的安全，都不撞，并且最小距离比较远
                        #
                        #         # social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0
                        #     else:
                        #         rew_aveGT_mapped = -1  # 一点也不合作

                        print('生成器 rew_avespeed:', rew_avespeed, 10 * rew_avespeed,
                              'rew_lane_pianyi:', rew_lane_pianyi, -10 * rew_lane_pianyi,
                              'comfort_adj:', comfort_adj, 5 * comfort_adj,
                              'rew_aveGT_mapped:', rew_minGT_mapped, 10 * rew_minGT_mapped)
                        rew_social_allbatch.append(
                            [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                        # print('生成器 rew_social_allbatch:', rew_social_allbatch)
                    else:
                        # 此时刻是无效数据，历史时刻都已经考虑过了
                        rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

                canshu_social_allbatch_array = np.array(rew_social_allbatch)

                # print('canshu_social_allbatch_array:',np.shape(canshu_social_allbatch_array),canshu_social_allbatch_array)  # (batch,4)
                # print('re_obs_lstm[k]:',np.shape(re_obs_lstm[k]),type(re_obs_lstm[k]))  # (batch,4)

                score, pre = discriminator[k].get_reward(re_obs_lstm[k],
                                                         action[k],  ##################
                                                         obs_lstm[k],
                                                         path_prob[k], canshu_social_allbatch_array,
                                                         discrim_score=False)
                # print('判别器的输出格式：','score:', np.shape(score), score)
                print('判别器的输出格式：','pre:', np.shape(pre), pre[0])

                rewards_panbieqi.append(np.squeeze(
                    score))  # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                social_panbieqi.append(np.squeeze(pre))


            print('rew:',np.shape(rew))
            print('ep_ret:',np.shape(ep_ret))



            for k in range(n_agents):
                all_rew[k].append(rew[k]+rewards_panbieqi[k])  #
                ep_ret[k] += rew[k]
                all_social[k].append(social_panbieqi[k])



            # print('运行完step并处理之后:', np.shape(all_ob), np.shape(obs), np.shape(rew), np.shape(dones),
            #       np.shape(ini_step_n), np.shape(ini_obs), np.shape(obs_lstm), np.shape(all_ob_lstm))
            # all_ob:(8, step, 56) obs:(8, 1, 56) rew:(8,) dones:(8,) ini_step_n:(8, 1, 1)
            # ini_obs:(8, 1, 56) obs_lstm:(8, 1, 21, 56) all_ob_lstm:(8, step, 21, 56)

            # if dones == [True,True,True,True,True,True,True,True]:
            #     done_eva = True
            #     step = 0
            if step == max_steps:
                done_eva = True
                step = 0
            else:
                done_eva = False

        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])
            all_ob_lstm[k] = np.squeeze(all_ob_lstm[k])
            all_attention_weight[k] = np.squeeze(all_attention_weight[k])
            # np.squeeze 是 NumPy 库中的一个函数，用于移除数组中形状为 1 的轴。举例来说，如果一个数组的形状是 (1, 10, 1, 5)，使用 np.squeeze 后，形状会变成 (10, 5)。



        print('all_ob:', np.shape(all_ob))
        print('all_ob_lstm:', np.shape(all_ob_lstm))
        print('all_attention_weight:', np.shape(all_attention_weight))
        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,'all_social': all_social,
            "ep_ret": ep_ret, "all_ob": all_agent_ob,'all_attention_weight':all_attention_weight
        }

        sample_trajs.append(traj_data)
        # print('traj_num', i, 'expected_return', ep_ret)

        for k in range(n_agents):
            avg_ret[k].append(ep_ret[k])

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]),np.mean(all_rew[k]))
    return sample_trajs

    # # actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
    # # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]
    # # action2 = []
    # # action2.append([np.concatenate((action[k][0], np.array([step]))) for k in range(n_agents)])
    # action2 = [np.squeeze(ac) for ac in action]
    # # print('评估时的action:',action2,'第几步:',step)
    # for k in range(n_agents):
    #     all_ob[k].append(obs[k])
    #     all_ac[k].append(action[k])
    #         all_agent_ob.append(np.concatenate(obs, axis=1))
    #         obs, rew, done, _ = env.step(action2)
    #         for k in range(n_agents):
    #             all_rew[k].append(rew[k])
    #             ep_ret[k] += rew[k]
    #         obs = [ob[None, :] for ob in obs]
    #         step += 1
    #
    #         # if image:
    #         #     img = env.render(mode='rgb_array')
    #         #     images.append(img[0])
    #         #     time.sleep(0.02)
    #         # if step == max_steps or True in done:
    #         if step == max_steps:
    #             done_eva = True
    #             step = 0
    #         else:
    #             done_eva = False
    #
    #     for k in range(n_agents):
    #         all_ob[k] = np.squeeze(all_ob[k])
    #
    #     all_agent_ob = np.squeeze(all_agent_ob)
    #     traj_data = {
    #         "ob": all_ob, "ac": all_ac, "rew": all_rew,
    #         "ep_ret": ep_ret, "all_ob": all_agent_ob
    #     }
    #
    #     sample_trajs.append(traj_data)
    #     # print('traj_num', i, 'expected_return', ep_ret)
    #
    #     for k in range(n_agents):
    #         avg_ret[k].append(ep_ret[k])
    #
    # print(path)
    # for k in range(n_agents):
    #     print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
    # return sample_trajs

    # images = np.array(images)
    # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
    # if image:
    #     print(images.shape)
    #     imageio.mimsave(path + '.mp4', images, fps=25)
    # return sample_trajs


def render_discrimination_nogenerate(path, path_d, model, env_id, mid, generate_trj, scenario_test):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id,scenario_test)

    # 读取discrimination
    # path_d = r'E:/wsh-科研/nvn_xuguan_sind/sinD_nvn_xuguan_9jiaohu_ATT-GPU-规则在内-social/MA_Intersection_straight' \
    #      r'/multi-agent-irl/irl/mack/multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
    #      r'/v9/l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0/seed-13/'

    discriminator = get_dis(path_d, model, env_id, mid, scenario_test)  # 获得判别器
    print('discriminator', discriminator, np.shape(discriminator))

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    # n_actions = [action.n for action in ac_space]

    images = []
    sample_trajs = []
    num_trajs = 1  #### 是不是跑两次？取两次数据
    max_steps = 230  # 484 # 179    #### 每个agent的步长可以限制185 但是场景的步长不要限制了
    avg_ret = [[] for _ in range(n_agents)]

    all_ac, obs_lstm_all_agents, all_ob, all_rew, all_social, all_attention_weight_spatial, all_attention_weight_temporal = [], [], [], [],[], [], []
    for k in range(n_agents):
        obs_lstm_all_agents.append([])
        all_ob.append([])
        all_rew.append([])
        all_social.append([])  # 存储社会倾向
        all_ac.append([])
        all_attention_weight_spatial.append([])
        all_attention_weight_temporal.append([])
        # ep_ret.append([])
    # obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm

    step = 0
    done_eva = False
    # 在循环之前创建深拷贝
    obs_all_agents = generate_trj['ob']  # (8,230,57)
    acs_all_agents = generate_trj['ac']  # (8,230,2)
    attention_weight_spatial_all_agents = generate_trj['all_attention_weight_spatial']  # (8,230,1)
    attention_weight_temporal_all_agents = generate_trj['all_attention_weight_temporal']

    for k in range(n_agents):
        # 把obs填充为有历史数据的 每一个时刻的obs为【21,46】
        agent_ob = obs_all_agents[k][:, :57]  # 获取当前 agent 的观察数据，形状为 [185, 46]
        # 遍历每个时刻
        obs_k_t = []
        for t in range(max_steps):
            # 提取当前时刻和前 20 个时刻的数据，不够 20 个时刻的部分用零填充
            obs_t = np.zeros((21, 57))
            if t >= 20:
                obs_t = agent_ob[t - 20:t + 1]
            else:
                obs_t[20 - t:] = agent_ob[:t + 1]
            obs_k_t.append(obs_t)

        obs_lstm_all_agents[k].append(obs_k_t)
    print('obs_lstm_all_agents:',np.shape(obs_lstm_all_agents))  # (8, 1, 230, 21, 57)

    while not done_eva:

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

        def Cal_GT_crash(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                         Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                         Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

            # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:
                # 计算和这个车辆的GT
                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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
                    GT_value = -2  # 非常不安全
                    dis_min = 0
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
                            GT_value = 1  # 安全
                            dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
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
                                # 2.1 AGENT 会把交互对象看做 有冲突的对象;交互对象也会把agent看做 有冲突的对象
                                # 判断当agent到交点的时候，neig在哪，如果撞了，GT_value=-1,否则None
                                agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                t_agent = agent_dis / agent_v

                                neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                t_neig = neig_dis / neig_v

                                dis_time = []  # 记录快车走到交点的路程中，两车之间轨迹点的距离，步长0.5s（不考虑车宽，因为是否碰撞上已经考虑了）
                                if t_agent < t_neig:
                                    # agent先到冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                    if t_agent > 23:
                                        t_agent = 23
                                    else:
                                        t_agent = t_agent
                                    # try:
                                    time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_agent:
                                        time = np.arange(0, t_agent + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_agent, 0.5)
                                        time = np.append(time, t_agent)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                Agent_y_t,
                                                                                                veh_length,
                                                                                                veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                               Jiaohu_y_t,
                                                                                               veh_length,
                                                                                               veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离
                                else:
                                    # neig先到冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                    if t_neig > 23:
                                        t_neig = 23
                                    else:
                                        t_neig = t_neig
                                    # try:
                                    time_n_0 = int(t_neig / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_neig:
                                        time = np.arange(0, t_neig + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_neig, 0.5)
                                        time = np.append(time, t_neig)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                Agent_y_t,
                                                                                                veh_length,
                                                                                                veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                               Jiaohu_y_t,
                                                                                               veh_length,
                                                                                               veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                            elif dot_product_agent >= 0 and dot_product_neig < 0:
                                # 2.2 agent把neig看做冲突对象，但是neig不把agent看做冲突对象，仍然需要判断在agent到冲突点的路程中，是否会发生碰撞，以及距离的大小
                                agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                t_agent = agent_dis / agent_v

                                neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                t_neig = neig_dis / neig_v

                                dis_time = []  # 记录agent车走到交点的路程中，两车之间轨迹点的距离（不考虑车宽，因为是否碰撞上已经考虑了）
                                # agent会到达冲突点，neig不会到达冲突点
                                crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞

                                if t_agent > 23:
                                    t_agent = 23
                                else:
                                    t_agent = t_agent
                                # try:
                                time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                if time_n_0 * 0.5 == t_agent:
                                    time = np.arange(0, t_agent + 0.01, 0.5)
                                else:
                                    time = np.arange(0, t_agent, 0.5)
                                    time = np.append(time, t_agent)  # 添加 t_agent 到
                                # except:
                                #     time_n_0 = 36
                                #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                for time_futurenow in time:
                                    # 走多少秒
                                    agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                    Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                        np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                    Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                        np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                    neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                    Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                        np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                    Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                        np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                    # 统计此时距离
                                    dis_time_futurenow = np.sqrt(
                                        (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                    dis_time.append(dis_time_futurenow)

                                    # 判断当前时刻辆车是否会相撞
                                    # 绘制两个矩形
                                    # 计算矩形的四个顶点坐标
                                    vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t, Agent_y_t,
                                                                                            veh_length, veh_width,
                                                                                            Agent_angle_last)
                                    vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t, Jiaohu_y_t,
                                                                                           veh_length, veh_width,
                                                                                           Jiaohu_angle_last)
                                    # 判断两个矩阵是否有交集
                                    intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                             vertices_neig_futurenow)
                                    if intersect_futurenow == True:
                                        # 说明agent和交互对象在未来交点相撞了
                                        GT_value = -1  # 不安全
                                        dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                        crash_futurenow = True
                                        break  # 退出这个循环，已经检测到未来会相撞
                                    else:
                                        GT_value = 1  # 安全
                                        crash_futurenow = False
                                if crash_futurenow == False:
                                    dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                            else:
                                # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                                # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                                GT_value = 0
                                dis_min = 100000



                    else:
                        # neig不是前车 2.4.2
                        GT_value = 0  # 不看做交互，因为当前时刻已经没有碰撞了
                        dis_min = 100000


            else:
                GT_value = 0  # 不交互
                dis_min = 100000

            return GT_value, dis_min

        def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                      Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                      Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

            # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:
                # 计算和这个车辆的GT
                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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



        rewards_panbieqi = []  # rewards = [] 和 report_rewards = []：创建用于存储判别器奖励的空列表。
        social_panbieqi = []  # 存放社会倾向 -π/2~π/2
        path_prob = np.zeros(n_agents)
        for k in range(n_agents):
            if k <= 2:
                direction_agent = 'left'
            else:
                direction_agent = 'straight'

            # 调用判别器的 get_reward 方法计算判别器奖励，将其添加到 rewards 列表中。这个奖励通常用于更新策略。
            # 没整明白 !
            # print('判别器的输入格式：', 're_obs_lstm:', np.shape(re_obs_lstm[k]), type(re_obs_lstm[k]))  # (10, 21, 46)
            # print('判别器的输入格式：', 're_actions:', np.shape(re_actions[k]), type(re_actions[k]))  # (10, 2)
            # print('判别器的输入格式：', 're_obs_next:', np.shape(re_obs_next_lstm[k]), type(re_obs_next_lstm[k]))  # (10, 21, 46)
            # print('判别器的输入格式：', 're_path_prob:', np.shape(re_path_prob[k]), type(re_path_prob[k]))  # 0.0

            re_obs_lstm = obs_lstm_all_agents[k][0][step]
            if step <= 228:
                obs_lstm_next = obs_lstm_all_agents[k][0][step+1]
            else:
                obs_lstm_next = obs_lstm_all_agents[k][0][step]
            batch_num = np.shape(obs_lstm_all_agents)[1]
            print('判别器的输入re_obs_lstm:', np.shape(re_obs_lstm))  # (21, 57)
            print('判别器的输入batch_num:',batch_num)
            # 计算利己和利他所需要的参数
            # 计算利己奖励和利他奖励，然后利用网络学习参数φ，cos(φ)=利己倾向，sin(φ)利他倾向
            rew_input_fuyuan = obs_lstm_next

            rew_social_allbatch = []  # 存放这一个agent 所有batch的参数

            # 利己性参数-速度, 针对每一个batch来计算
            for i_batch in range(1):
                # 改成当前时刻应该更好，因为是当前时刻的奖励，过去的已经无法改变了。过去的状态可以看做是影响社交倾向的因素
                # 如果是考虑历史数据的话，对于一些当前时刻无效，但历史时刻有效的数据来说，奖励就没有实际含义了
                # 其实也可以有实际含义。再想想。还是不考虑了
                if rew_input_fuyuan[20][0] != 0:
                    use_GT = []  # 存放这个ibatch的主要交互对象的GT
                    # speed = np.sqrt(rew_input_fuyuan[i_batch][20][2] ** 2 + rew_input_fuyuan[i_batch][20][3] ** 2)
                    pianyi_distance = rew_input_fuyuan[20][-2]
                    # 计算和主要交互对象的GT
                    # 提取代理的状态和终点坐标
                    agent_x = rew_input_fuyuan[20][0] * 38 - 4
                    agent_y = rew_input_fuyuan[20][1] * 23 + 14
                    agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                    agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                    agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1  # 上一个点的前进方向

                    agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                    # 避免碰撞
                    # 计算agent和周围最密切的三个交互对象的GT
                    # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
                    # 安全奖励
                    rew_GT = 0
                    # use_GT = []  # 存储和主要交互对象的GT  会有7+2个值，包括除主agent之外所有的agent和交互的landmark，即使没有这个对象，也会赋值为0
                    # 计算和除主车之外所有agent以及交互的landmark车辆的GT
                    # 把所有的agent都考虑
                    for agent_k_ in range(n_agents):
                        if agent_k_ != k:
                            # 这个agent不是我们正在计算的k
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
                                jiaohu_agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1  # 上一个点的前进方向

                                jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  jiaohu_agent_x, jiaohu_agent_y,
                                                                  jiaohu_agent_vx, jiaohu_agent_vy,
                                                                  jiaohu_agent_angle_last,
                                                                  direction_jiaohu)
                                # if same_jiaohu_agent_GT_value is not None:
                                use_GT.append(jiaohu_agent_GT_value)
                            else:  # 没有这个agent
                                jiaohu_agent_x = -4
                                jiaohu_agent_y = 14
                                jiaohu_agentk_vx = -14
                                jiaohu_agent_vy = -2
                                jiaohu_agent_angle_last = -1
                                jiaohu_agent_GT_value = None
                                # dis_min = 100000
                                use_GT.append(jiaohu_agent_GT_value)

                    # 左侧视野的landmark
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

                    # 右侧视野的landmark
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

                    # 计算角度波动
                    # 计算一些rew
                    # # 计算上个时刻，上上个时刻，上上上时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                    # if rew_input_fuyuan[i_batch][18][0] == 0:
                    #     # 没有上上上时刻的角度，所以判断上上个时刻的角度（一开始无论左转还是直行几乎都是直行的角度）
                    #     if rew_input_fuyuan[i_batch][19][0] == 0:
                    #         # 也没有上上时刻的角度
                    #         heading_angle_last3_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #         heading_angle_last2_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #     else:
                    #         # 有上上时刻的角度，上上上时刻的角度也用上上时刻的角度来代替
                    #         heading_angle_last3_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #         heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    # else:
                    #     # 有上上上时刻的角度，所以也有上上个时刻的角度
                    #     heading_angle_last3_real = rew_input_fuyuan[i_batch][18][6] * 191 - 1  # [0,1]
                    #     heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #
                    # heading_angle_last1_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #
                    # # 计算平均值
                    # mean_value = np.mean(
                    #     [heading_angle_last1_real, heading_angle_last2_real, heading_angle_last3_real])
                    # # 计算每个数据与平均值的差的平方
                    # squared_differences = [(x - mean_value) ** 2 for x in
                    #                        [heading_angle_last1_real, heading_angle_last2_real,
                    #                         heading_angle_last3_real]]
                    # # 计算平方差的平均值
                    # mean_squared_difference = np.mean(squared_differences)
                    # # 计算标准差
                    # std_dev = np.sqrt(mean_squared_difference)
                    # if std_dev > 3:
                    #     rew_heading_std_bodong = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                    # else:
                    #     rew_heading_std_bodong = 1

                    # 计算steering angle正负来回变化带来的惩罚
                    # 上一时刻的转角，若超过均值的1个标准差，则给予惩罚之类的
                    penalty = 1  # 惩罚系数
                    delta_angle_last1 = rew_input_fuyuan[20][56]
                    comfort_adj = 0  # 初始化转向角过大惩罚
                    if direction_agent == 'left':
                        left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                        left_delta_angle_last1_realmean = 1.085
                        left_delta_angle_last1_realstd = 0.702
                        if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                            comfort_adj = 0  # 不做惩罚

                        else:
                            dis_left_delta_angle_last1 = abs(
                                left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                            if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                        dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                                # 越靠近left_delta_angle_last1_realstd，惩罚越接近-1
                    else:
                        right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                        right_delta_angle_last1_realmean = 0.001
                        right_delta_angle_last1_realstd = 0.076
                        if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                            comfort_adj = 0  # 不做惩罚

                        else:
                            dis_right_delta_angle_last1 = abs(
                                right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                            if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                            dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                                # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1

                    # 利己-效率
                    rew_avespeed = agent_v / 6.8  # 除以85分位速度
                    # 利己-车道偏移
                    rew_lane_pianyi = pianyi_distance

                    # 利他-GT
                    # print("use_GT：",use_GT)  # 9个元素的list，例如[None, None, None, None, None, None, None, 0.32294667405015254, None]
                    use_GT_list = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                    print('use_GT_list:', use_GT_list)
                    if len(use_GT_list) != 0:
                        # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                        # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                        if rew_minGT <= 1.5:
                            # 归一化
                            normalized_data = (rew_minGT - 0) / (1.5 - 0)
                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                        elif 1.5 < rew_minGT < 3:
                            # 归一化
                            normalized_data = (rew_minGT - 1.5) / (3 - 1.5)

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                        elif 3 <= rew_minGT <= 4:
                            # 归一化
                            normalized_data = (rew_minGT - 3) / (4 - 3)

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                        elif rew_minGT > 4:
                            # 归一化
                            normalized_data = np.exp(-(1 / (rew_minGT - 4)))

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                    else:
                        rew_minGT_mapped = 0
                        social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0

                    print('生成器 rew_avespeed:', rew_avespeed, 10 * rew_avespeed,
                          'rew_lane_pianyi:', rew_lane_pianyi, -10 * rew_lane_pianyi,
                          'comfort_adj:', comfort_adj, 5 * comfort_adj,
                          'rew_aveGT_mapped:', rew_minGT_mapped, 10 * rew_minGT_mapped)
                    rew_social_allbatch.append(
                        [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                    # print('生成器 rew_social_allbatch:', rew_social_allbatch)
                else:
                    # 此时刻是无效数据，历史时刻都已经考虑过了
                    rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

            canshu_social_allbatch_array = np.array(rew_social_allbatch)

            # print('canshu_social_allbatch_array:',np.shape(canshu_social_allbatch_array),canshu_social_allbatch_array)  # (batch,4)
            # print('re_obs_lstm[k]:',np.shape(re_obs_lstm[k]),type(re_obs_lstm[k]))  # (batch,4)

            score, pre = discriminator[k].get_reward(re_obs_lstm,
                                                     np.array([0, 0]),
                                                     obs_lstm_next,
                                                     path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)
            # print('判别器的输出格式：','score:', np.shape(score), score)
            print('判别器的输出格式：','pre:', np.shape(pre), pre[0])

            rewards_panbieqi.append(np.squeeze(score))  # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
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
        all_attention_weight_spatial[k] = attention_weight_spatial_all_agents[k]
        all_attention_weight_temporal[k] = attention_weight_temporal_all_agents[k]

    print('all_ob:', np.shape(all_ob))
    print('all_ac:', np.shape(all_ac))
    print('all_attention_weight_spatial:', np.shape(all_attention_weight_spatial))
    print('all_attention_weight_temporal:', np.shape(all_attention_weight_temporal))

    traj_data = {
        "ob": all_ob, "rew": all_rew, 'all_social': all_social, "ac": all_ac,
        'all_attention_weight_spatial':all_attention_weight_spatial,
        'all_attention_weight_temporal': all_attention_weight_temporal,
    }

    sample_trajs.append(traj_data)
    # print('traj_num', i, 'expected_return', ep_ret)

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]),np.mean(all_rew[k]))
    return sample_trajs

    # # actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
    # # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]
    # # action2 = []
    # # action2.append([np.concatenate((action[k][0], np.array([step]))) for k in range(n_agents)])
    # action2 = [np.squeeze(ac) for ac in action]
    # # print('评估时的action:',action2,'第几步:',step)
    # for k in range(n_agents):
    #     all_ob[k].append(obs[k])
    #     all_ac[k].append(action[k])
    #         all_agent_ob.append(np.concatenate(obs, axis=1))
    #         obs, rew, done, _ = env.step(action2)
    #         for k in range(n_agents):
    #             all_rew[k].append(rew[k])
    #             ep_ret[k] += rew[k]
    #         obs = [ob[None, :] for ob in obs]
    #         step += 1
    #
    #         # if image:
    #         #     img = env.render(mode='rgb_array')
    #         #     images.append(img[0])
    #         #     time.sleep(0.02)
    #         # if step == max_steps or True in done:
    #         if step == max_steps:
    #             done_eva = True
    #             step = 0
    #         else:
    #             done_eva = False
    #
    #     for k in range(n_agents):
    #         all_ob[k] = np.squeeze(all_ob[k])
    #
    #     all_agent_ob = np.squeeze(all_agent_ob)
    #     traj_data = {
    #         "ob": all_ob, "ac": all_ac, "rew": all_rew,
    #         "ep_ret": ep_ret, "all_ob": all_agent_ob
    #     }
    #
    #     sample_trajs.append(traj_data)
    #     # print('traj_num', i, 'expected_return', ep_ret)
    #
    #     for k in range(n_agents):
    #         avg_ret[k].append(ep_ret[k])
    #
    # print(path)
    # for k in range(n_agents):
    #     print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
    # return sample_trajs

    # images = np.array(images)
    # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
    # if image:
    #     print(images.shape)
    #     imageio.mimsave(path + '.mp4', images, fps=25)
    # return sample_trajs


def render_discrimination_expert(path, path_d, model, env_id, mid, expert_trj):
    print("load model from", path)
    model.load(path)

    env = create_env(env_id)

    # 读取discrimination
    # path_d = r'E:/wsh-科研/nvn_xuguan_sind/sinD_nvn_xuguan_9jiaohu_ATT-GPU-规则在内-social/MA_Intersection_straight' \
    #      r'/multi-agent-irl/irl/mack/multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
    #      r'/v9/l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0/seed-13/'

    discriminator = get_dis(path_d, model, env_id, mid)  # 获得判别器
    print('discriminator', discriminator, np.shape(discriminator))

    n_agents = len(env.action_space)
    ob_space = env.observation_space
    ac_space = env.action_space
    # n_actions = [action.n for action in ac_space]

    images = []
    sample_trajs = []
    num_trajs = 1  #### 是不是跑两次？取两次数据
    max_steps = 185  # 484 # 179    #### 每个agent的步长可以限制185 但是场景的步长不要限制了
    avg_ret = [[] for _ in range(n_agents)]

    all_ac, obs_lstm_all_agents, all_ob, all_rew, all_social, all_attention_weight= [], [], [], [],[], []
    for k in range(n_agents):
        obs_lstm_all_agents.append([])
        all_ob.append([])
        all_rew.append([])
        all_social.append([])  # 存储社会倾向
        all_ac.append([])
        all_attention_weight.append([])
        # ep_ret.append([])
    # obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm

    step = 0
    done_eva = False
    # 在循环之前创建深拷贝
    obs_all_agents = expert_trj['ob']  # (8,230,57)
    acs_all_agents = expert_trj['ac']  # (8,230,2)
    # attention_weight_all_agents = generate_trj['all_attention_weight']  # (8,230,1)

    for k in range(n_agents):
        # 把obs填充为有历史数据的 每一个时刻的obs为【21,46】
        agent_ob = obs_all_agents[k][:, :57]  # 获取当前 agent 的观察数据，形状为 [185, 46]
        # 遍历每个时刻
        obs_k_t = []
        for t in range(max_steps):
            # 提取当前时刻和前 20 个时刻的数据，不够 20 个时刻的部分用零填充
            obs_t = np.zeros((21, 57))
            if t >= 20:
                obs_t = agent_ob[t - 20:t + 1]
            else:
                obs_t[20 - t:] = agent_ob[:t + 1]
            obs_k_t.append(obs_t)

        obs_lstm_all_agents[k].append(obs_k_t)
    print('obs_lstm_all_agents:',np.shape(obs_lstm_all_agents))  # (8, 1, 230, 21, 57)

    while not done_eva:

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

        def Cal_GT_crash(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                         Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                         Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

            # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:
                # 计算和这个车辆的GT
                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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
                    GT_value = -2  # 非常不安全
                    dis_min = 0
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
                            GT_value = 1  # 安全
                            dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
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
                                # 2.1 AGENT 会把交互对象看做 有冲突的对象;交互对象也会把agent看做 有冲突的对象
                                # 判断当agent到交点的时候，neig在哪，如果撞了，GT_value=-1,否则None
                                agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                t_agent = agent_dis / agent_v

                                neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                t_neig = neig_dis / neig_v

                                dis_time = []  # 记录快车走到交点的路程中，两车之间轨迹点的距离，步长0.5s（不考虑车宽，因为是否碰撞上已经考虑了）
                                if t_agent < t_neig:
                                    # agent先到冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                    if t_agent > 23:
                                        t_agent = 23
                                    else:
                                        t_agent = t_agent
                                    # try:
                                    time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_agent:
                                        time = np.arange(0, t_agent + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_agent, 0.5)
                                        time = np.append(time, t_agent)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                Agent_y_t,
                                                                                                veh_length,
                                                                                                veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                               Jiaohu_y_t,
                                                                                               veh_length,
                                                                                               veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离
                                else:
                                    # neig先到冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                    if t_neig > 23:
                                        t_neig = 23
                                    else:
                                        t_neig = t_neig
                                    # try:
                                    time_n_0 = int(t_neig / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_neig:
                                        time = np.arange(0, t_neig + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_neig, 0.5)
                                        time = np.append(time, t_neig)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                Agent_y_t,
                                                                                                veh_length,
                                                                                                veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                               Jiaohu_y_t,
                                                                                               veh_length,
                                                                                               veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                            elif dot_product_agent >= 0 and dot_product_neig < 0:
                                # 2.2 agent把neig看做冲突对象，但是neig不把agent看做冲突对象，仍然需要判断在agent到冲突点的路程中，是否会发生碰撞，以及距离的大小
                                agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                t_agent = agent_dis / agent_v

                                neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                t_neig = neig_dis / neig_v

                                dis_time = []  # 记录agent车走到交点的路程中，两车之间轨迹点的距离（不考虑车宽，因为是否碰撞上已经考虑了）
                                # agent会到达冲突点，neig不会到达冲突点
                                crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞

                                if t_agent > 23:
                                    t_agent = 23
                                else:
                                    t_agent = t_agent
                                # try:
                                time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                if time_n_0 * 0.5 == t_agent:
                                    time = np.arange(0, t_agent + 0.01, 0.5)
                                else:
                                    time = np.arange(0, t_agent, 0.5)
                                    time = np.append(time, t_agent)  # 添加 t_agent 到
                                # except:
                                #     time_n_0 = 36
                                #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                for time_futurenow in time:
                                    # 走多少秒
                                    agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                    Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                        np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                    Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                        np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                    neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                    Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                        np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                    Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                        np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                    # 统计此时距离
                                    dis_time_futurenow = np.sqrt(
                                        (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                    dis_time.append(dis_time_futurenow)

                                    # 判断当前时刻辆车是否会相撞
                                    # 绘制两个矩形
                                    # 计算矩形的四个顶点坐标
                                    vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t, Agent_y_t,
                                                                                            veh_length, veh_width,
                                                                                            Agent_angle_last)
                                    vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t, Jiaohu_y_t,
                                                                                           veh_length, veh_width,
                                                                                           Jiaohu_angle_last)
                                    # 判断两个矩阵是否有交集
                                    intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                             vertices_neig_futurenow)
                                    if intersect_futurenow == True:
                                        # 说明agent和交互对象在未来交点相撞了
                                        GT_value = -1  # 不安全
                                        dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                        crash_futurenow = True
                                        break  # 退出这个循环，已经检测到未来会相撞
                                    else:
                                        GT_value = 1  # 安全
                                        crash_futurenow = False
                                if crash_futurenow == False:
                                    dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                            else:
                                # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                                # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                                GT_value = 0
                                dis_min = 100000



                    else:
                        # neig不是前车 2.4.2
                        GT_value = 0  # 不看做交互，因为当前时刻已经没有碰撞了
                        dis_min = 100000


            else:
                GT_value = 0  # 不交互
                dis_min = 100000

            return GT_value, dis_min

        def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                      Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                      Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

            # 先计算两车的距离，如果在15m之内，则看做可能交互的对象。否则直接不交互。
            dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
            if dis_between_agent_jiaohu <= 15:
                # 计算和这个车辆的GT
                agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

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



        rewards_panbieqi = []  # rewards = [] 和 report_rewards = []：创建用于存储判别器奖励的空列表。
        social_panbieqi = []  # 存放社会倾向 -π/2~π/2
        path_prob = np.zeros(n_agents)
        for k in range(n_agents):
            if k <= 2:
                direction_agent = 'left'
            else:
                direction_agent = 'straight'

            # 调用判别器的 get_reward 方法计算判别器奖励，将其添加到 rewards 列表中。这个奖励通常用于更新策略。
            # 没整明白 !
            # print('判别器的输入格式：', 're_obs_lstm:', np.shape(re_obs_lstm[k]), type(re_obs_lstm[k]))  # (10, 21, 46)
            # print('判别器的输入格式：', 're_actions:', np.shape(re_actions[k]), type(re_actions[k]))  # (10, 2)
            # print('判别器的输入格式：', 're_obs_next:', np.shape(re_obs_next_lstm[k]), type(re_obs_next_lstm[k]))  # (10, 21, 46)
            # print('判别器的输入格式：', 're_path_prob:', np.shape(re_path_prob[k]), type(re_path_prob[k]))  # 0.0

            re_obs_lstm = obs_lstm_all_agents[k][0][step]

            obs_lstm_next = re_obs_lstm
            batch_num = np.shape(obs_lstm_all_agents)[1]
            print('判别器的输入re_obs_lstm:', np.shape(re_obs_lstm))  # (21, 57)
            print('判别器的输入batch_num:',batch_num)
            # 计算利己和利他所需要的参数
            # 计算利己奖励和利他奖励，然后利用网络学习参数φ，cos(φ)=利己倾向，sin(φ)利他倾向
            rew_input_fuyuan = re_obs_lstm

            rew_social_allbatch = []  # 存放这一个agent 所有batch的参数

            # 利己性参数-速度, 针对每一个batch来计算
            for i_batch in range(1):
                # 改成当前时刻应该更好，因为是当前时刻的奖励，过去的已经无法改变了。过去的状态可以看做是影响社交倾向的因素
                # 如果是考虑历史数据的话，对于一些当前时刻无效，但历史时刻有效的数据来说，奖励就没有实际含义了
                # 其实也可以有实际含义。再想想。还是不考虑了
                if rew_input_fuyuan[20][0] != 0:
                    use_GT = []  # 存放这个ibatch的主要交互对象的GT
                    # speed = np.sqrt(rew_input_fuyuan[i_batch][20][2] ** 2 + rew_input_fuyuan[i_batch][20][3] ** 2)
                    pianyi_distance = rew_input_fuyuan[20][-2]
                    # 计算和主要交互对象的GT
                    # 提取代理的状态和终点坐标
                    agent_x = rew_input_fuyuan[20][0] * 38 - 4
                    agent_y = rew_input_fuyuan[20][1] * 23 + 14
                    agent_vx = rew_input_fuyuan[20][2] * 21 - 14
                    agent_vy = rew_input_fuyuan[20][3] * 12 - 2
                    agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1  # 上一个点的前进方向

                    agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                    # 避免碰撞
                    # 计算agent和周围最密切的三个交互对象的GT
                    # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
                    # 安全奖励
                    rew_GT = 0
                    # use_GT = []  # 存储和主要交互对象的GT  会有7+2个值，包括除主agent之外所有的agent和交互的landmark，即使没有这个对象，也会赋值为0
                    # 计算和除主车之外所有agent以及交互的landmark车辆的GT
                    # 把所有的agent都考虑
                    for agent_k_ in range(n_agents):
                        if agent_k_ != k:
                            # 这个agent不是我们正在计算的k
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
                                jiaohu_agent_angle_last = rew_input_fuyuan[20][6] * 191 - 1  # 上一个点的前进方向

                                jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                  agent_angle_last, direction_agent,
                                                                  jiaohu_agent_x, jiaohu_agent_y,
                                                                  jiaohu_agent_vx, jiaohu_agent_vy,
                                                                  jiaohu_agent_angle_last,
                                                                  direction_jiaohu)
                                # if same_jiaohu_agent_GT_value is not None:
                                use_GT.append(jiaohu_agent_GT_value)
                            else:  # 没有这个agent
                                jiaohu_agent_x = -4
                                jiaohu_agent_y = 14
                                jiaohu_agentk_vx = -14
                                jiaohu_agent_vy = -2
                                jiaohu_agent_angle_last = -1
                                jiaohu_agent_GT_value = None
                                # dis_min = 100000
                                use_GT.append(jiaohu_agent_GT_value)

                    # 左侧视野的landmark
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

                    # 右侧视野的landmark
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

                    # 计算角度波动
                    # 计算一些rew
                    # # 计算上个时刻，上上个时刻，上上上时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                    # if rew_input_fuyuan[i_batch][18][0] == 0:
                    #     # 没有上上上时刻的角度，所以判断上上个时刻的角度（一开始无论左转还是直行几乎都是直行的角度）
                    #     if rew_input_fuyuan[i_batch][19][0] == 0:
                    #         # 也没有上上时刻的角度
                    #         heading_angle_last3_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #         heading_angle_last2_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #     else:
                    #         # 有上上时刻的角度，上上上时刻的角度也用上上时刻的角度来代替
                    #         heading_angle_last3_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #         heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    # else:
                    #     # 有上上上时刻的角度，所以也有上上个时刻的角度
                    #     heading_angle_last3_real = rew_input_fuyuan[i_batch][18][6] * 191 - 1  # [0,1]
                    #     heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #
                    # heading_angle_last1_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #
                    # # 计算平均值
                    # mean_value = np.mean(
                    #     [heading_angle_last1_real, heading_angle_last2_real, heading_angle_last3_real])
                    # # 计算每个数据与平均值的差的平方
                    # squared_differences = [(x - mean_value) ** 2 for x in
                    #                        [heading_angle_last1_real, heading_angle_last2_real,
                    #                         heading_angle_last3_real]]
                    # # 计算平方差的平均值
                    # mean_squared_difference = np.mean(squared_differences)
                    # # 计算标准差
                    # std_dev = np.sqrt(mean_squared_difference)
                    # if std_dev > 3:
                    #     rew_heading_std_bodong = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                    # else:
                    #     rew_heading_std_bodong = 1

                    # 计算steering angle正负来回变化带来的惩罚
                    # 上一时刻的转角，若超过均值的1个标准差，则给予惩罚之类的
                    penalty = 1  # 惩罚系数
                    delta_angle_last1 = rew_input_fuyuan[20][56]
                    comfort_adj = 0  # 初始化转向角过大惩罚
                    if direction_agent == 'left':
                        left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                        left_delta_angle_last1_realmean = 1.085
                        left_delta_angle_last1_realstd = 0.702
                        if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                            comfort_adj = 0  # 不做惩罚

                        else:
                            dis_left_delta_angle_last1 = abs(
                                left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                            if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                        dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                                # 越靠近left_delta_angle_last1_realstd，惩罚越接近-1
                    else:
                        right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                        right_delta_angle_last1_realmean = 0.001
                        right_delta_angle_last1_realstd = 0.076
                        if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                            comfort_adj = 0  # 不做惩罚

                        else:
                            dis_right_delta_angle_last1 = abs(
                                right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                            if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                comfort_adj = -1 * penalty
                            else:
                                comfort_adj = -(
                                            dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                                # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1

                    # 利己-效率
                    rew_avespeed = agent_v / 6.8  # 除以85分位速度
                    # 利己-车道偏移
                    rew_lane_pianyi = pianyi_distance

                    # 利他-GT
                    # print("use_GT：",use_GT)  # 9个元素的list，例如[None, None, None, None, None, None, None, 0.32294667405015254, None]
                    use_GT_list = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                    print('use_GT_list:', use_GT_list)
                    if len(use_GT_list) != 0:
                        # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                        rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                        # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                        if rew_minGT <= 1.5:
                            # 归一化
                            normalized_data = (rew_minGT - 0) / (1.5 - 0)
                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                        elif 1.5 < rew_minGT < 3:
                            # 归一化
                            normalized_data = (rew_minGT - 1.5) / (3 - 1.5)

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                        elif 3 <= rew_minGT <= 4:
                            # 归一化
                            normalized_data = (rew_minGT - 3) / (4 - 3)

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                        elif rew_minGT > 4:
                            # 归一化
                            normalized_data = np.exp(-(1 / (rew_minGT - 4)))

                            # 映射到目标范围
                            rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                    else:
                        rew_minGT_mapped = 0
                        social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0

                    print('生成器 rew_avespeed:', rew_avespeed, 10 * rew_avespeed,
                          'rew_lane_pianyi:', rew_lane_pianyi, -10 * rew_lane_pianyi,
                          'comfort_adj:', comfort_adj, 5 * comfort_adj,
                          'rew_aveGT_mapped:', rew_minGT_mapped, 10 * rew_minGT_mapped)
                    rew_social_allbatch.append(
                        [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                    # print('生成器 rew_social_allbatch:', rew_social_allbatch)
                else:
                    # 此时刻是无效数据，历史时刻都已经考虑过了
                    rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

            canshu_social_allbatch_array = np.array(rew_social_allbatch)

            # print('canshu_social_allbatch_array:',np.shape(canshu_social_allbatch_array),canshu_social_allbatch_array)  # (batch,4)
            # print('re_obs_lstm[k]:',np.shape(re_obs_lstm[k]),type(re_obs_lstm[k]))  # (batch,4)

            score, pre = discriminator[k].get_reward(re_obs_lstm,
                                                     np.array([0, 0]),
                                                     obs_lstm_next,
                                                     path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)
            # print('判别器的输出格式：','score:', np.shape(score), score)
            print('判别器的输出格式：','pre:', np.shape(pre), pre[0])

            rewards_panbieqi.append(np.squeeze(score))  # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
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
        # all_attention_weight[k] = attention_weight_all_agents[k]

    print('all_ob:', np.shape(all_ob))
    print('all_ac:', np.shape(all_ac))
    # print('all_attention_weight:', np.shape(all_attention_weight))

    traj_data = {
        "ob": all_ob, "rew": all_rew, 'all_social': all_social, "ac": all_ac
    }

    sample_trajs.append(traj_data)
    # print('traj_num', i, 'expected_return', ep_ret)

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]),np.mean(all_rew[k]))
    return sample_trajs

    # # actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
    # # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]
    # # action2 = []
    # # action2.append([np.concatenate((action[k][0], np.array([step]))) for k in range(n_agents)])
    # action2 = [np.squeeze(ac) for ac in action]
    # # print('评估时的action:',action2,'第几步:',step)
    # for k in range(n_agents):
    #     all_ob[k].append(obs[k])
    #     all_ac[k].append(action[k])
    #         all_agent_ob.append(np.concatenate(obs, axis=1))
    #         obs, rew, done, _ = env.step(action2)
    #         for k in range(n_agents):
    #             all_rew[k].append(rew[k])
    #             ep_ret[k] += rew[k]
    #         obs = [ob[None, :] for ob in obs]
    #         step += 1
    #
    #         # if image:
    #         #     img = env.render(mode='rgb_array')
    #         #     images.append(img[0])
    #         #     time.sleep(0.02)
    #         # if step == max_steps or True in done:
    #         if step == max_steps:
    #             done_eva = True
    #             step = 0
    #         else:
    #             done_eva = False
    #
    #     for k in range(n_agents):
    #         all_ob[k] = np.squeeze(all_ob[k])
    #
    #     all_agent_ob = np.squeeze(all_agent_ob)
    #     traj_data = {
    #         "ob": all_ob, "ac": all_ac, "rew": all_rew,
    #         "ep_ret": ep_ret, "all_ob": all_agent_ob
    #     }
    #
    #     sample_trajs.append(traj_data)
    #     # print('traj_num', i, 'expected_return', ep_ret)
    #
    #     for k in range(n_agents):
    #         avg_ret[k].append(ep_ret[k])
    #
    # print(path)
    # for k in range(n_agents):
    #     print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))
    # return sample_trajs

    # images = np.array(images)
    # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
    # if image:
    #     print(images.shape)
    #     imageio.mimsave(path + '.mp4', images, fps=25)
    # return sample_trajs


# if __name__ == '__main__':
#     render()



# discriminator[0].load(path2)

#     discriminator[0].get_reward(np.array([1,0]),
#                                 np.array([0,0]),
#                                 np.array([1,0]),
#                                 np.array([1,0]),
#                                 discrim_score=False)


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