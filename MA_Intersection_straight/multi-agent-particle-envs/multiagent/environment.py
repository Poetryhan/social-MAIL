import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from scipy.special import comb, perm
import pickle


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, observation_callback_AVTEST_nowstep=None,
                 observation_callback_AVTEST_nextstep=None, observation_callback_nowstep=None):
        print('运行的是nvn的MultiAgentEnv')
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.observation_callback_AVTEST_nowstep = observation_callback_AVTEST_nowstep
        self.observation_callback_AVTEST_nextstep = observation_callback_AVTEST_nextstep
        self.observation_callback_nowstep = observation_callback_nowstep
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        '''在这行代码中，self.shared_reward 的值取决于 world.collaborative 是否存在。如果 world.collaborative 存在且为真（即不为 None 且为真值），则 self.shared_reward 将设置为 True，表示智能体共享奖励。如果 world.collaborative 不存在（即未设置），那么 self.shared_reward 将被设置为 False，表示智能体不共享奖励。
这种设置允许在创建环境时选择是否启用智能体之间的协作，如果 world.collaborative 被设置为 True，则表示任务是协作的，智能体共享奖励。如果未设置 world.collaborative，则默认为非协作任务，智能体不共享奖励。这种灵活性允许根据具体的任务需求来确定是否启用协作。
        '''
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []

        # # 加速度范围
        # acceleration_low = -14.0
        # acceleration_high = 6
        #
        # # 角度范围
        # angle_low = -5
        # angle_high = 180.0

        for agent in self.agents:
            total_action_space = []
            # physical action space

            if self.discrete_action_space:  # self.discrete_action_space = False
                u_action_space = spaces.Discrete(8)  ####
            else:
                angle_low = -1
                angle_high = 1
                acceleration_low = -1
                acceleration_high = 1

                # 创建动作空间
                u_action_space = spaces.Box(
                    low=np.array([acceleration_low, angle_low]),
                    high=np.array([acceleration_high, angle_high]),
                    dtype=np.float32
                )

            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:  # self.discrete_action_space = False
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=-1, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:  # world.agents[i].silent = True
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            # obs_dim = len(observation_callback(agent, self.world))
            # self.observation_space.append(spaces.Box(np.array([0, 0, 0]),np.array([+12, +3, +3]), dtype=np.float16)) ####
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(57,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering 这段代码涉及到环境的可视化渲染部分，主要用于在屏幕上显示智能体和环境的可视化信息。
        self.shared_viewer = shared_viewer  # shared_viewer=True
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n  # 其中 self.n 是环境中智能体的数量。这表示每个智能体都有自己的可视化窗口，每个窗口独立渲染对应智能体的信息。
        self._reset_render()

    def step(self, action_n):  # step函数用于执行一个时间步的模拟环境交互 ，参数action_n是一个场景内上一个步长所有智能体动作的列表。这个函数会返回一个元组，包括每个智能体的新观察值、奖励、完成标志和信息。(10,67,2)
        # print('是不是运行了这个?????????????????????????????????????????????????????????,world的time是多少???????????',np.shape(action_n))  # (8, 62)
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        ini_step_n = [[item[4]] for item in action_n]
        ini_obs = [item[5:62] for item in action_n]  # 这个场景的初始观测值
        obs_n_lstm = []  # 包含20个历史时刻信息的obs，加上当前时刻的总共21个时刻
        actions_new_n = np.zeros((8, 2))  # 存放n个agent的动作,因为经过step之后action可能会改变 （8,2）
        rew_n_social_generate = []
        collide_situation = []  # 存储碰撞情况

        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # 调用 _set_action 方法，将动作 action_n[i] 应用于智能体 agent。这个方法用于设置智能体的动作，根据智能体的属性，可能需要将连续动作转换为适当的形式。
            self._set_action(action_n[i], agent, self.action_space[i], 'first')
            # (self.actions[k][i], np.array([env_go_step]),
            #  np.array([self.trj_GO_STEP[i][k]]),
            #  np.array([self.ini_steps[k][i][0]]),
            #  self.ini_obs[k][i])
        # advance world state
        #     print('step中的action_n[i]是：', action_n[i])
        self.world.step('first')  # 使环境中的物理世界前进一步，这将更新每个智能体的状态，包括位置和速度。

        # record observation for each agent  安全模块，避免碰撞
        # def calculate_rectangle_vertices(center_x, center_y, length, width, angle):
        #     # 计算矩形的四个顶点相对于中心点的坐标（逆时针方向）
        #     angle_rad = np.radians(angle)  # 将角度转换为弧度
        #     cos_angle = np.cos(angle_rad)
        #     sin_angle = np.sin(angle_rad)
        #
        #     # 计算矩形的四个顶点相对于中心点的坐标
        #     x_offset = 0.5 * length
        #     y_offset = 0.5 * width
        #     vertices = [
        #         (center_x - x_offset * cos_angle + y_offset * sin_angle,
        #          center_y - x_offset * sin_angle - y_offset * cos_angle),
        #         (center_x + x_offset * cos_angle + y_offset * sin_angle,
        #          center_y + x_offset * sin_angle - y_offset * cos_angle),
        #         (center_x + x_offset * cos_angle - y_offset * sin_angle,
        #          center_y + x_offset * sin_angle + y_offset * cos_angle),
        #         (center_x - x_offset * cos_angle - y_offset * sin_angle,
        #          center_y - x_offset * sin_angle + y_offset * cos_angle)
        #     ]
        #
        #     return vertices
        #
        # def check_intersection(rect1_vertices, rect2_vertices):
        #     # 检查两个矩形是否相交
        #     def orientation(p, q, r):
        #         val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        #         if val == 0:
        #             return 0  # 线段 pqr 共线
        #         return 1 if val > 0 else 2  # 顺时针或逆时针方向
        #
        #     def on_segment(p, q, r):
        #         if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        #                 q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        #             return True
        #         return False
        #
        #     def do_intersect(p1, q1, p2, q2):
        #         o1 = orientation(p1, q1, p2)
        #         o2 = orientation(p1, q1, q2)
        #         o3 = orientation(p2, q2, p1)
        #         o4 = orientation(p2, q2, q1)
        #
        #         # 一般情况下
        #         if o1 != o2 and o3 != o4:
        #             return True
        #
        #         # 特殊情况
        #         if (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, q2, q1)) or \
        #                 (o3 == 0 and on_segment(p2, p1, q2)) or (o4 == 0 and on_segment(p2, q1, q2)):
        #             return True
        #
        #         return False
        #
        #     for i in range(4):
        #         for j in range(4):
        #             if do_intersect(rect1_vertices[i], rect1_vertices[(i + 1) % 4], rect2_vertices[j],
        #                             rect2_vertices[(j + 1) % 4]):
        #                 return True
        #
        #     return False
        #
        # landmark_crash = [np.zeros((2)) for _ in range(8)] # 记录每一个agent是否和landmark_crash相撞，如果撞了，就是1，没撞就是0
        # agent_crash = [np.zeros((8)) for _ in range(8)]   # 记录每一个agent是否和其他agent相撞，如果撞了，就是1，没撞就是0，要检查所有其他agent
        # agent_inf_real = [np.zeros((5)) for _ in range(8)]  # 记录每一个agent的真实坐标和角度,vx,vy
        # dec_agent_inf = [np.zeros((1)) for _ in range(8)]  # 记录每一个agent是否已经有减速经历，有则是1，无则是0
        # length_agent = 5
        # width_agent = 2
        # length_landmark = 5
        # width_landmark = 2
        # for agent in self.agents:
        #     agent_id = agent.id
        #     # obs_data_lstm, obs_usefor_reward, a
        #     obs_lstm, obs_usefor_reward, obs = self._get_obs(agent, action_n[agent_id], [False])
        #     # print('environment get_obs输出的数据的shape：', 'obs_lstm:', np.shape(obs_lstm),
        #     #       'obs_usefor_reward:', np.shape(obs_usefor_reward), 'obs:', np.shape(obs))
        #     # obs_lstm: (21, 46) obs_usefor_reward: (1, 46) obs: (46,)
        #
        #     # 判断是否每个agent 都不会和自己的交互对象碰撞
        #     agent_x = obs_lstm[-1][0]
        #     agent_y = obs_lstm[-1][1]
        #
        #     if agent_x != 0:
        #         # 这个agent是有效的
        #         agent_x_real = agent_x * 38 - 4
        #         agent_y_real = agent_y * 23 + 14
        #         agent_angle_real = obs_lstm[-1][6] * 191 - 1
        #         agent_vx_real = obs_lstm[-1][2] * 21 - 14
        #         agent_vy_real = obs_lstm[-1][3] * 12 - 2
        #         agent_inf_real[agent_id][0] = agent_x_real
        #         agent_inf_real[agent_id][1] = agent_y_real
        #         agent_inf_real[agent_id][2] = agent_angle_real
        #         agent_inf_real[agent_id][3] = agent_vx_real
        #         agent_inf_real[agent_id][4] = agent_vy_real
        #
        #         # 先检查landmark left
        #         landmark_left_delta_x = obs_lstm[-1][38]
        #         landmark_left_delta_y = obs_lstm[-1][39]
        #         landmark_left_delta_vx = obs_lstm[-1][40]
        #         landmark_left_delta_vy = obs_lstm[-1][41]
        #         if landmark_left_delta_x != 0:
        #             # 有左侧的landmark交互
        #             # 求出真实的位置，并画出均质化矩形(不考虑landmark的类型)，可以解释为，行人和非机动车虽然较小，但是需要的安全空间也更大
        #             landmark_left_x_real = agent_x_real - landmark_left_delta_x * 29 - 14
        #             landmark_left_y_real = agent_y_real - landmark_left_delta_y * 30 - 15
        #             landmark_left_angle_real = obs_lstm[-1][53] * 360 - 90
        #             landmark_left_vx_real = agent_vx_real - landmark_left_delta_vx * 35 - 21
        #             landmark_left_vy_real = agent_vy_real - landmark_left_delta_vy * 16 - 5
        #             # 绘制两个矩形
        #             # 计算矩形的四个顶点坐标
        #             vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent, width_agent, agent_angle_real)
        #             vertices_ld_left = calculate_rectangle_vertices(landmark_left_x_real, landmark_left_y_real, length_landmark, width_landmark, landmark_left_angle_real)
        #             # 判断两个矩阵是否有交集
        #             intersect_a_ldf = check_intersection(vertices_agent, vertices_ld_left)
        #             if intersect_a_ldf == True:
        #                 # 说明agent和landmark在更新了这一位置之后相撞了
        #                 landmark_crash[agent_id][0] = 1
        #             else:
        #                 landmark_crash[agent_id][0] = 0
        #
        #         # 再检查landmark right
        #         landmark_right_delta_x = obs_lstm[-1][42]
        #         landmark_right_delta_y = obs_lstm[-1][43]
        #         landmark_right_delta_vx = obs_lstm[-1][44]
        #         landmark_right_delta_vy = obs_lstm[-1][45]
        #         if landmark_right_delta_x != 0:
        #             # 有right侧的landmark交互
        #             # 求出真实的位置，并画出均质化矩形(不考虑landmark的类型)，可以解释为，行人和非机动车虽然较小，但是需要的安全空间也更大
        #             landmark_right_x_real = agent_x_real - landmark_right_delta_x * 30 - 15
        #             landmark_right_y_real = agent_y_real - landmark_right_delta_y * 29 - 15
        #             landmark_right_angle_real = obs_lstm[-1][53] * 360 - 90
        #             landmark_right_vx_real = agent_vx_real - landmark_right_delta_vx * 25 - 14
        #             landmark_right_vy_real = agent_vy_real - landmark_right_delta_vy * 17 - 7
        #             # 绘制两个矩形
        #             # 计算矩形的四个顶点坐标
        #             vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent,
        #                                                           width_agent, agent_angle_real)
        #             vertices_ld_right = calculate_rectangle_vertices(landmark_right_x_real, landmark_right_y_real,
        #                                                             length_landmark, width_landmark,
        #                                                             landmark_right_angle_real)
        #             # 判断两个矩阵是否有交集
        #             intersect_a_ldr = check_intersection(vertices_agent, vertices_ld_right)
        #             if intersect_a_ldr == True:
        #                 # 说明agent和landmark在更新了这一位置之后相撞了
        #                 landmark_crash[agent_id][1] = 1
        #             else:
        #                 landmark_crash[agent_id][1] = 0
        #
        # # 判断agent是否会和其他所有的agent相撞
        # for i, agent1_pos in enumerate(agent_inf_real):
        #     if np.all(agent1_pos == 0):  # 如果代理的真实位置为0，则跳过
        #         continue
        #     for j, agent2_pos in enumerate(agent_inf_real):
        #         if i == j or np.all(agent2_pos == 0):  # 如果是同一个代理或者代理的真实位置为0，则跳过
        #             continue
        #         # 计算两个代理的矩形顶点坐标
        #         vertices_agent = calculate_rectangle_vertices(agent1_pos[0], agent1_pos[1], length_agent, width_agent,
        #                                                       agent1_pos[2])
        #         vertices_agent2 = calculate_rectangle_vertices(agent2_pos[0], agent2_pos[1],
        #                                                         length_agent, width_agent, agent2_pos[2])
        #         # 判断两个矩阵是否有交集
        #         intersect_a_a = check_intersection(vertices_agent, vertices_agent2)
        #         if intersect_a_a == True:
        #             # 说明agent和这个agent在更新了这一位置之后相撞了
        #             agent_crash[i][j] = 1
        #         else:
        #             agent_crash[i][j] = 0
        #
        # for i, landmark_crash_i in enumerate(landmark_crash):
        #     if np.any(landmark_crash_i == 1):
        #         # 至少和一个landmark碰撞, 这个agent就减速，加速度降低3m/s2
        #         action_n[i][0] = (action_n[i][0] * 4.9 - 3)/4.9
        #         dec_agent_inf[i][0] = 1
        # for i, agent_crash_i in enumerate(agent_crash):
        #     # 这辆车已经根据landmark减速过了，就不再处理了
        #     if dec_agent_inf[i][0] == 1:
        #         continue
        #     # 这辆车没有被landmark减速过，则需要再看情况
        #     # 分别对左转车和直行车进行判断，并且分别判断是否前方有同向车，以及是否和对向车撞
        #     else:
        #         # 先判断是否和前方同向车相撞
        #         # 左转车
        #         if i <= 2:
        #             # 先判断是否和前方左转车相撞，如果撞，则需要减速
        #             for j_left, crash_flag_j_left in enumerate(agent_crash_i[:3]):
        #                 if crash_flag_j_left == 1:
        #                     # 在前三个元素中找到了碰撞标志
        #                     # 判断是否是前车
        #                     agent_pos = agent_inf_real[i][:2]  # 获取车辆的坐标信息
        #                     veh_pos = agent_inf_real[j_left][:2]
        #                     # 计算和x轴正向的夹角
        #                     angle_with_x_axis = np.arctan2(veh_pos[1] - agent_pos[1], veh_pos[0] - agent_pos[0])
        #                     # 将弧度转换为角度
        #                     angle_degrees = np.degrees(angle_with_x_axis)
        #                     if angle_degrees >= 270:
        #                         angle_degrees = angle_degrees - 360  # 这里的角度范围是【-90，270】
        #
        #                     if angle_degrees < -90:
        #                         angle_degrees = angle_degrees + 360  # 这里的角度范围是【-90，270】
        #                     # 判断角度是否在-90~90之间
        #                     if -90 < angle_degrees <= 90:
        #                         # 在前方范围内，与前车相撞
        #                         # 判断这辆车是否减速过（因为要循环多个同向车，可能在前一个同向车处已经减速了）
        #                         if dec_agent_inf[i][0] == 1:
        #                             # 已经减速过了，就不再减速了
        #                             action_n[i][0] = action_n[i][0]
        #                         else:
        #                             # 没有减速，则需要减速，并进行减速标记
        #                             action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
        #                             dec_agent_inf[i][0] = 1
        #         # 直行车
        #         if i > 2:
        #             # 先判断是否和前方直行车相撞，如果撞，则需要减速
        #             for j_straight, crash_flag_j_straight in enumerate(agent_crash_i[3:]):
        #                 if crash_flag_j_straight == 1:
        #                     # 在后五个元素中找到了碰撞标志
        #                     # 判断是否是前车
        #                     agent_pos = agent_inf_real[i][:2]  # 获取车辆的坐标信息
        #                     veh_pos = agent_inf_real[j_straight][:2]
        #                     # 计算和x轴正向的夹角
        #                     angle_with_x_axis = np.arctan2(veh_pos[1] - agent_pos[1],
        #                                                    veh_pos[0] - agent_pos[0])
        #                     # 将弧度转换为角度
        #                     angle_degrees = np.degrees(angle_with_x_axis)
        #                     if angle_degrees >= 270:
        #                         angle_degrees = angle_degrees - 360  # 这里的角度范围是【-90，270】
        #                     if angle_degrees < -90:
        #                         angle_degrees = angle_degrees + 360  # 这里的角度范围是【-90，270】
        #
        #                     # 判断角度是否在90~270之间
        #                     if 90 < angle_degrees <= 270:
        #                         # 在前方范围内，与前车相撞
        #                         # 判断这辆车是否减速过（因为要循环多个同向车，可能在前一个同向车处已经减速了）
        #                         if dec_agent_inf[i][0] == 1:
        #                             # 已经减速过了，就不再减速了
        #                             action_n[i][0] = action_n[i][0]
        #                         else:
        #                             # 没有减速，则需要减速，并进行减速标记
        #                             action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
        #                             dec_agent_inf[i][0] = 1
        #
        #         # 再判断是否和对向车相撞，如果撞，则需要减速
        #         # 左转车
        #         if i <= 2:
        #             # 经历上述同向车的判断，已经减速了
        #             if dec_agent_inf[i][0] == 1:
        #                 continue
        #             else:
        #                 # 继续判断是否和对向车撞
        #                 for j_left_2, crash_flag_j_left_2 in enumerate(agent_crash_i[3:]):
        #                     if crash_flag_j_left_2 == 1:
        #                         # 在后五个元素中找到了碰撞标志
        #                         # 判断谁的速度大，谁就不减速
        #                         agent_v = np.linalg.norm(agent_inf_real[i, 3:5])  # 获取车辆的坐标信息
        #                         veh_v = np.linalg.norm(agent_inf_real[j_left_2, 3:5])
        #                         if agent_v < veh_v:
        #                             # 速度较低，可以减速让行
        #                             # 判断这辆车是否减速过（因为要循环多个对向车，可能在前一个对向车处已经减速了）
        #                             if dec_agent_inf[i][0] == 1:
        #                                 # 已经减速过了，就不再减速了
        #                                 action_n[i][0] = action_n[i][0]
        #                             else:
        #                                 # 没有减速，则需要减速，并进行减速标记
        #                                 action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
        #                                 dec_agent_inf[i][0] = 1
        #
        #         # 直行车
        #         if i > 2:
        #             # 经历上述同向车的判断，已经减速了
        #             if dec_agent_inf[i][0] == 1:
        #                 continue
        #             else:
        #                 # 继续判断是否和对向车撞
        #                 for j_straight_2, crash_flag_j_straight_2 in enumerate(agent_crash_i[:3]):
        #                     if crash_flag_j_straight_2 == 1:
        #                         # 在后五个元素中找到了碰撞标志
        #                         # 判断谁的速度大，谁就不减速
        #                         agent_v = np.linalg.norm(agent_inf_real[i, 3:5])  # 获取车辆的坐标信息
        #                         veh_v = np.linalg.norm(agent_inf_real[j_straight_2, 3:5])
        #                         if agent_v < veh_v:
        #                             # 速度较低，可以减速让行
        #                             # 判断这辆车是否减速过（因为要循环多个对向车，可能在前一个对向车处已经减速了）
        #                             if dec_agent_inf[i][0] == 1:
        #                                 # 已经减速过了，就不再减速了
        #                                 action_n[i][0] = action_n[i][0]
        #                             else:
        #                                 # 没有减速，则需要减速，并进行减速标记
        #                                 action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
        #                                 dec_agent_inf[i][0] = 1
        #
        #
        #
        #                 # 至少和一个agent碰撞, 这个agent就减速，判断谁的v当前更高，高的加速度降低3m/s2
        #                 # 检查最大速度
        #                 # 计算速度模长
        #                 agent_speed_norm = np.linalg.norm(agent_inf_real[:, 3:5], axis=1)  # 计算速度模长
        #                 max_speed = np.max(agent_speed_norm)  # 获取最大速度
        #                 agent_speed = np.linalg.norm(agent_inf_real[i, 3:5])  # 获取当前 agent 的速度模长
        #
        #                 # 如果当前 agent 的速度是最大速度，减小加速度
        #                 if agent_speed == max_speed:
        #                     print(f"Agent {i} 的速度是最大速度")
        #                     # 减小加速度
        #
        #                     action_n[i][0] = (action_n[i][0] * 4.9 - 3)/4.9
        #
        # # 进行完上述的碰撞减速处理之后，重新直行一遍
        # for i, agent in enumerate(self.agents):
        #     # 调用 _set_action 方法，将动作 action_n[i] 应用于智能体 agent。这个方法用于设置智能体的动作，根据智能体的属性，可能需要将连续动作转换为适当的形式。
        #     self._set_action(action_n[i], agent, self.action_space[i], 'second')
        # self.world.step('second')  # 使环境中的物理世界前进一步，这将更新每个智能体的状态，包括位置和速度。

        for agent in self.agents:
            agent_id = agent.id
            # obs_data_lstm, obs_usefor_reward, a
            obs_lstm, obs_usefor_reward, obs, collide_label, rew_social_generate = self._get_obs(agent, action_n[agent_id], [False])
            # obs_lstm, obs_usefor_reward, obs = self._get_obs_AV_test_nextstep(agent, action_n[agent_id], [False])
            obs_n.append(obs)  # 调用 _get_obs 方法，获取智能体 agent 的新观察值，并将其添加到 obs_n 列表中。
            obs_n_lstm.append(obs_lstm)
            rew_n_social_generate.append(rew_social_generate)
            collide_situation.append(collide_label)
            # obs_n.append(self._get_obs(agent)) # 调用 _get_obs 方法，获取智能体 agent 的新观察值，并将其添加到 obs_n 列表中。
            reward_n.append(self._get_reward(agent, obs_usefor_reward))  # 这里的reward是0
            # done_n.append(self._get_done(agent)) # 一个步长一个值,False或者是True
            if action_n[agent_id][2] >= 229:  # 训练的时候是185 在评估的时候扩宽为230
                done_n.append(True)
            else:
                done_n.append(self._get_done(agent))  # 一个步长一个值,False或者是True
            # print('agent:',agent.id,'的这个步长的done为:',done_n)

            info_n['n'].append(self._get_info(agent))

            actions_new_n[agent_id][0] = agent.action.u[0]
            actions_new_n[agent_id][1] = agent.action.u[1]

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)  # 计算所有智能体的奖励之和，将其存储在 reward 变量中。这是协作场景下的总奖励。
        if self.shared_reward:  # 如果共享奖励，将 reward 复制为一个长度为 self.n 的列表，表示每个智能体都获得相同的总奖励。
            reward_n = [reward] * self.n

        # print('environment step输出的数据的shape：', 'obs_n_lstm:', np.shape(obs_n_lstm),
        #       'obs_n:', np.shape(obs_n),'ini_obs:',np.shape(ini_obs), 'actions_new_n:',np.shape(actions_new_n))

        # obs_n_lstm: (8, 21, 57) obs_n: (8, 57) ini_obs: (8, 57) actions_new_n: (8, 2)
        return obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs, actions_new_n, rew_n_social_generate, collide_situation

    def reset(self, infs):  # infs为[scenario_test, traning_label]
        # reset world
        # print('reset环境！')
        # ini_obs, ini_steps, ini_obs_lstm
        print('environment reset中的infs：', infs)
        ini_obs, ini_steps, ini_obs_lstm = self.reset_callback(self.world, infs[0], infs[1])
        print('reset中的ini_obs agent 3的x和y：', ini_obs[3][:2])
        # 这个world中所有agent的初始obs 包括了ini_obs 【8*18】,ini_steps [8,1]
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        obs_n_lstm = []
        reset_infos = [True]
        rew_n_social_generate = []
        collide_situation = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            action_n_agent = np.concatenate((agent.state.p_pos, np.array([0]),
                                             np.array([0]),
                                             np.array([ini_steps[agent.id][0]]), ini_obs[agent.id]))

            obs_lstm_old, obs_usefor_reward, obs_old, collide_label, rew_social_generate = self._get_obs(agent, action_n_agent, reset_infos)
            obs_n.append(obs_old)
            obs_n_lstm.append(obs_lstm_old)
            rew_n_social_generate.append(rew_social_generate)
            collide_situation.append(collide_label)

        #print('env中的obs_n:', np.shape(obs_n), type(obs_n))  # (8, 56)
        # obs_lstm, ob, ini_step, ini_ob, reset_info
        return obs_n_lstm, obs_n, ini_steps, ini_obs, reset_infos, ini_obs_lstm, rew_n_social_generate, collide_situation

    def ini_obs_update(self, ini_obs_old_list):
        #print('ini_obs_update得到的ini_obs_old_list的shape为：',np.shape(ini_obs_old_list),ini_obs_old_list)  # (8, 63)
        self.agents = self.world.policy_agents
        for agent in self.agents:
            if ini_obs_old_list[agent.id][-1] == True:
                # 这个agent是初始时刻
                # print('ini_obs_update中 agent', agent.id, '的x和y:',ini_obs_old_list[agent.id][5:7])
                agent.state.p_pos = ini_obs_old_list[agent.id][5:7]
                # print('ini_obs_update中 赋值之后agent', agent.id, '的x和y:', ini_obs_old_list[agent.id][5:7])
                agent.state.p_vel = ini_obs_old_list[agent.id][7:9]
                agent.state.heading_angle_last1 = ini_obs_old_list[agent.id][11]
                agent.state.p_dis = ini_obs_old_list[agent.id][12]
                agent.state.p_ini_to_end_dis = ini_obs_old_list[agent.id][12]  # 对于这个agent来说，这个值就不再变了，无论更新迭代到哪一步，这个值都不会被更新
                agent.state.p_last_vx = ini_obs_old_list[agent.id][13]
                agent.state.p_last_vy = ini_obs_old_list[agent.id][14]
                agent.state.ini_step = ini_obs_old_list[agent.id][62]  # cpui agentk开始的步
                # getobs不需要参考线，所以这里可以先不初始化
                # agent.state.reference_line = np.array(all_expert_trj_[agent.state.scenario_id]['ob'][agent.id][:, :2])
                agent.collide = False
                agent.end_label = False
                # print('action[12]!!!!!!!!!!!!!!!!!!!!!:',action[12],agent.state.p_ini_to_end_dis)

                p_pos_x = agent.state.p_pos[0] * 38 - 4  # 当前真实位置
                p_pos_y = agent.state.p_pos[1] * 23 + 14

                dx = ini_obs_old_list[agent.id][4] * 59 - 37  # 当前和终点的真实距离
                dy = ini_obs_old_list[agent.id][5] * 27 - 4

                des_x = dx + p_pos_x  # 真实的终点坐标
                des_y = dy + p_pos_y

                # print('测试中的p_pos格式',init_points[i][:2],world.agents[i].state.p_pos[0],world.agents[i].state.p_pos[1],'测试中的p_pos_x:',p_pos_x,'测试中的p_pos_y:',p_pos_y,'测试中的dx:',dx,'测试中的dy:',dy,'测试中的des_x:',des_x,'测试中的des_y:',des_y)

                agent.state.p_des[0] = (des_x + 4) / 38
                agent.state.p_des[1] = (des_y - 14) / 23  # 轨迹终点的归一化坐标

        # 获得新的ini_obs
        obs_n = []
        obs_n_lstm = []
        collide_situation = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            action_n_agent = np.concatenate((agent.state.p_pos, np.array([ini_obs_old_list[agent.id][2]]),
                                             np.array([ini_obs_old_list[agent.id][3]]),
                                             np.array([ini_obs_old_list[agent.id][4]])))

            obs_lstm_old, obs_usefor_reward, obs_old, collide_label = self._get_obs_nowstep(agent, action_n_agent)
            obs_n.append(obs_old)
            obs_n_lstm.append(obs_lstm_old)
            collide_situation.append(collide_label)
            # print('action_n_agent[0]',action_n_agent[0],agent.state.p_pos)
            # if agent.state.p_pos[0] != 0:
            #     print('检查：',action_n_agent[2],action_n_agent[4])
            # if agent.id == 0:
            #     print('environment中agent的pos:',agent.state.p_pos[0],agent.collide)
            # obs_n.append(self._get_obs(agent))
        # print('env中的obs_n:',np.shape(obs_n),obs_n)
        return obs_n_lstm, obs_n, collide_situation

    def AVtest_get_ini_obs(self, AV_inf, scenario_id, target_agent_id, env_step, trj_step):
        # 初始化HV的时候AV恰好或者已经进入交叉口
        path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight' \
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        f_expert = open(path, 'rb')
        all_expert_trj_ = pickle.load(f_expert)

        init_pointss = np.load(
            r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
            r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight'
            r'\AV_test\DATA\init_sinD_nvnxuguan_9jiaohu_social_dayu1_v2.npy',
            allow_pickle=True)

        # path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_social_dayu1.pkl'
        # f_expert = open(path, 'rb')
        # all_expert_trj_ = pickle.load(f_expert)
        #
        # init_pointss = np.load(
        #     r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\init_east_left_social_dayu1.npy',
        #     allow_pickle=True)


        init_points = init_pointss[scenario_id]
        init_points[1] = np.zeros(58)  # 加了这个！需要个性化修改，对于测试场景来说，用AV替换哪一个agent
        ini_steps = init_points[:, 57:58]
        self.agents = self.world.policy_agents
        for agent in self.agents:
            if agent.id == target_agent_id:
                # 这个在更换场景的时候需要个性化修改，对于scenario98来说，我们把唯一的一辆左转车替换为了AV，所以是id为0的agent为AV
                agent.state.p_pos = init_points[agent.id][0:2]
                agent.state.p_vel = init_points[agent.id][2:4]
                agent.state.heading_angle_last1 = init_points[agent.id][6]
                agent.state.p_dis = init_points[agent.id][7]
                agent.state.p_ini_to_end_dis = init_points[agent.id][7]  # 对于这个agent来说，这个值就不再变了，无论更新迭代到哪一步，这个值都不会被更新
                agent.state.p_last_vx = init_points[agent.id][8]
                agent.state.p_last_vy = init_points[agent.id][9]
                agent.state.ini_step = init_points[agent.id][57]  # cpui agentk开始的步
                agent.state.reference_line = np.array(all_expert_trj_[agent.state.scenario_id]['ob'][agent.id][:, :2])
                agent.collide = False
                agent.end_label = False
                # print('action[12]!!!!!!!!!!!!!!!!!!!!!:',action[12],agent.state.p_ini_to_end_dis)

                p_pos_x = agent.state.p_pos[0] * 38 - 4  # 当前真实位置
                p_pos_y = agent.state.p_pos[1] * 23 + 14

                dx = init_points[agent.id][4] * 59 - 37  # 当前和终点的真实距离
                dy = init_points[agent.id][5] * 27 - 4

                des_x = dx + p_pos_x  # 真实的终点坐标
                des_y = dy + p_pos_y

                # print('测试中的p_pos格式',init_points[i][:2],world.agents[i].state.p_pos[0],world.agents[i].state.p_pos[1],'测试中的p_pos_x:',p_pos_x,'测试中的p_pos_y:',p_pos_y,'测试中的dx:',dx,'测试中的dy:',dy,'测试中的des_x:',des_x,'测试中的des_y:',des_y)

                agent.state.p_des[0] = (des_x + 4) / 38
                agent.state.p_des[1] = (des_y - 14) / 23  # 轨迹终点的归一化坐标

        # 这个world中所有agent的初始obs 包括了ini_obs 【8*18】,ini_steps [8,1]
        # reset renderer
        # record observations for each agent
        obs_n = []
        obs_n_lstm = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            action_n_agent = np.concatenate((agent.state.p_pos, np.array([env_step]),
                                             np.array([trj_step[0][agent.id]]),
                                             np.array([ini_steps[agent.id][0]])))

            obs_lstm_old, obs_usefor_reward, obs_old = self._get_obs_AV_test_nowstep(agent, action_n_agent, AV_inf)
            obs_n.append(obs_old)
            obs_n_lstm.append(obs_lstm_old)
            # print('action_n_agent[0]',action_n_agent[0],agent.state.p_pos)
            # if agent.state.p_pos[0] != 0:
            #     print('检查：',action_n_agent[2],action_n_agent[4])
            # if agent.id == 0:
            #     print('environment中agent的pos:',agent.state.p_pos[0],agent.collide)
            # obs_n.append(self._get_obs(agent))
        # print('env中的obs_n:',np.shape(obs_n),obs_n)
        return obs_n_lstm, obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent, action_n_agent, reset_infos):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, action_n_agent, reset_infos, self.world)

    # get observation for a particular agent
    def _get_obs_nowstep(self, agent, action_n_agent):
        if self.observation_callback_nowstep is None:
            return np.zeros(0)
        return self.observation_callback_nowstep(agent, action_n_agent, self.world)

    # get observation for a particular agent
    def _get_obs_AV_test_nextstep(self, agent, action_n_agent, reset_infos):
        if self.observation_callback_AVTEST_nextstep is None:
            return np.zeros(0)
        return self.observation_callback_AVTEST_nextstep(agent, action_n_agent, reset_infos, self.world)

    # get observation for a particular agent for AV test
    def _get_obs_AV_test_nowstep(self, agent, action_n_agent, AV_inf):
        if self.observation_callback_AVTEST_nowstep is None:
            return np.zeros(0)
        return self.observation_callback_AVTEST_nowstep(agent, action_n_agent, AV_inf, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent, obs_use_for_reward):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, obs_use_for_reward, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, cishu):  # [acc jiaodu]
        path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
               r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight' \
               r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        f_expert = open(path, 'rb')
        all_expert_trj_ = pickle.load(f_expert)

        # path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_social_dayu1.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        # f_expert = open(path, 'rb')
        # all_expert_trj_ = pickle.load(f_expert)

        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # agent.state.step = agent.state.step + 1  # 更新一步，虽然是并行环境，但是每个cpu到这里都是要加一步的，所以可以同时加，如果重新选了专家场景，step会归零
        # (self.actions[k][i], np.array([env_go_step]),
        # np.array([self.trj_GO_STEP[i][k]]),
        # np.array([self.ini_steps[k][i][0]]), self.ini_obs[k][i])
        # print('action:',np.shape(action))
        if (action[2] == action[4]) and (action[5] != 0 or action[6] != 0 or action[7] != 0 or action[8] != 0):  # action[5] != 0这个条件是为了不让场景内没有的agent被赋值
            print('agent:',agent.id,'环境步数：',action[2], '轨迹初始步：',action[4], '轨迹初始信息：',action[5], action[6], action[7], action[8])
            # cpui场景累计的前进步数，等于 cpui中agentk的ini_step
            agent.state.p_pos = action[5:7]
            agent.state.p_vel = action[7:9]
            agent.state.heading_angle_last1 = action[11]
            agent.state.p_dis = action[12]
            agent.state.p_ini_to_end_dis = action[12]  # 对于这个agent来说，这个值就不再变了，无论更新迭代到哪一步，这个值都不会被更新
            agent.state.p_last_vx = action[13]
            agent.state.p_last_vy = action[14]
            agent.state.ini_step = action[4]  # cpui agentk开始的步
            agent.state.reference_line = np.array(all_expert_trj_[agent.state.scenario_id]['ob'][agent.id][:, :2])
            agent.state.min_distance = 0  # 初始时刻是真实的位置，所以距离参考线的距离是0
            agent.collide = False
            agent.end_label = False
            # print('action[12]!!!!!!!!!!!!!!!!!!!!!:',action[12],agent.state.p_ini_to_end_dis)

            p_pos_x = agent.state.p_pos[0] * 38 - 4  # 当前真实位置
            p_pos_y = agent.state.p_pos[1] * 23 + 14

            dx = action[9] * 59 - 37  # 当前和终点的真实距离
            dy = action[10] * 27 - 4

            des_x = dx + p_pos_x  # 真实的终点坐标
            des_y = dy + p_pos_y

            # print('测试中的p_pos格式',init_points[i][:2],world.agents[i].state.p_pos[0],world.agents[i].state.p_pos[1],'测试中的p_pos_x:',p_pos_x,'测试中的p_pos_y:',p_pos_y,'测试中的dx:',dx,'测试中的dy:',dy,'测试中的des_x:',des_x,'测试中的des_y:',des_y)

            agent.state.p_des[0] = (des_x + 4) / 38
            agent.state.p_des[1] = (des_y - 14) / 23  # 轨迹终点的归一化坐标

        if action[3] >= 0 and cishu == 'first':  # 轨迹开始走了，并且是初次执行的时候。判断完碰撞再执行set_action，就不用再把step+1了
            agent.state.step = agent.state.step + 1
        # process action
        # print('action:',np.shape(action),action)
        acc = action[0]
        delta_theta = action[1]
        if agent.id < 3:  # 左转车
            acc = min(max(action[0], -1), 1)
            delta_theta = min(max(action[1], -1.3), 1.2)  # 改为最大是0.5（对应真实的是2.5）
        if agent.id >= 3:  # 直行车
            acc = min(max(action[0], -1), 1)
            delta_theta = min(max(action[1], -1), 1)

        agent.action.u[0] = acc
        agent.action.u[1] = delta_theta

        if agent.id <= 2:
            if delta_theta>1.2 or delta_theta<-1.3:
                print('environment左转车的转向角为：', agent.id, action[1], agent.action.u[1], ((2.8 * (agent.action.u[1] + 1))/2) - 0.3)
        else:
            if delta_theta > 1 or delta_theta < -1:
                print('environment直行车的转向角为：', agent.id, action[1], agent.action.u[1], ((2.4 * (agent.action.u[1] + 1))/2) - 1.2)

        # make sure we used all elements of action
        # assert len(action) == 0

    # reset rendering assets 用于重置渲染相关的变量或资源的函数
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame 这段代码用于创建接受器场（receptor field）的位置信息，这些接受器场通常用于在仿真环境中模拟感知或接受信息的位置。下面是这段代码的详细解释：
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
