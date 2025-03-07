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
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []

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
                u_action_space = spaces.Box(
                    low=np.array([acceleration_low, angle_low]),
                    high=np.array([acceleration_high, angle_high]),
                    dtype=np.float32)
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
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(57,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer  # shared_viewer=True
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        ini_step_n = [[item[4]] for item in action_n]
        ini_obs = [item[5:62] for item in action_n]
        obs_n_lstm = []
        actions_new_n = np.zeros((8, 2))

        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i], 'first')

        # advance world state
        self.world.step('first')

        # safe code
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

        landmark_crash = [np.zeros((2)) for _ in range(8)]
        agent_crash = [np.zeros((8)) for _ in range(8)]
        agent_inf_real = [np.zeros((5)) for _ in range(8)]
        dec_agent_inf = [np.zeros((1)) for _ in range(8)]
        length_agent = 5
        width_agent = 2
        length_landmark = 5
        width_landmark = 2
        for agent in self.agents:
            agent_id = agent.id
            obs_lstm, obs_usefor_reward, obs = self._get_obs(agent, action_n[agent_id], [False])

            agent_x = obs_lstm[-1][0]
            agent_y = obs_lstm[-1][1]

            if agent_x != 0:
                agent_x_real = agent_x * 38 - 4
                agent_y_real = agent_y * 23 + 14
                agent_angle_real = obs_lstm[-1][6] * 191 - 1
                agent_vx_real = obs_lstm[-1][2] * 21 - 14
                agent_vy_real = obs_lstm[-1][3] * 12 - 2
                agent_inf_real[agent_id][0] = agent_x_real
                agent_inf_real[agent_id][1] = agent_y_real
                agent_inf_real[agent_id][2] = agent_angle_real
                agent_inf_real[agent_id][3] = agent_vx_real
                agent_inf_real[agent_id][4] = agent_vy_real

                landmark_left_delta_x = obs_lstm[-1][38]
                landmark_left_delta_y = obs_lstm[-1][39]
                landmark_left_delta_vx = obs_lstm[-1][40]
                landmark_left_delta_vy = obs_lstm[-1][41]
                if landmark_left_delta_x != 0:
                    landmark_left_x_real = agent_x_real - landmark_left_delta_x * 29 - 14
                    landmark_left_y_real = agent_y_real - landmark_left_delta_y * 30 - 15
                    landmark_left_angle_real = obs_lstm[-1][53] * 360 - 90
                    landmark_left_vx_real = agent_vx_real - landmark_left_delta_vx * 35 - 21
                    landmark_left_vy_real = agent_vy_real - landmark_left_delta_vy * 16 - 5
                    vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent, width_agent, agent_angle_real)
                    vertices_ld_left = calculate_rectangle_vertices(landmark_left_x_real, landmark_left_y_real, length_landmark, width_landmark, landmark_left_angle_real)
                    intersect_a_ldf = check_intersection(vertices_agent, vertices_ld_left)
                    if intersect_a_ldf == True:
                        landmark_crash[agent_id][0] = 1
                    else:
                        landmark_crash[agent_id][0] = 0

                landmark_right_delta_x = obs_lstm[-1][42]
                landmark_right_delta_y = obs_lstm[-1][43]
                landmark_right_delta_vx = obs_lstm[-1][44]
                landmark_right_delta_vy = obs_lstm[-1][45]
                if landmark_right_delta_x != 0:
                    landmark_right_x_real = agent_x_real - landmark_right_delta_x * 30 - 15
                    landmark_right_y_real = agent_y_real - landmark_right_delta_y * 29 - 15
                    landmark_right_angle_real = obs_lstm[-1][53] * 360 - 90
                    landmark_right_vx_real = agent_vx_real - landmark_right_delta_vx * 25 - 14
                    landmark_right_vy_real = agent_vy_real - landmark_right_delta_vy * 17 - 7
                    vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent,
                                                                  width_agent, agent_angle_real)
                    vertices_ld_right = calculate_rectangle_vertices(landmark_right_x_real, landmark_right_y_real,
                                                                    length_landmark, width_landmark,
                                                                    landmark_right_angle_real)
                    intersect_a_ldr = check_intersection(vertices_agent, vertices_ld_right)
                    if intersect_a_ldr == True:
                        landmark_crash[agent_id][1] = 1
                    else:
                        landmark_crash[agent_id][1] = 0

        for i, agent1_pos in enumerate(agent_inf_real):
            if np.all(agent1_pos == 0):
                continue
            for j, agent2_pos in enumerate(agent_inf_real):
                if i == j or np.all(agent2_pos == 0):
                    continue
                vertices_agent = calculate_rectangle_vertices(agent1_pos[0], agent1_pos[1], length_agent, width_agent,
                                                              agent1_pos[2])
                vertices_agent2 = calculate_rectangle_vertices(agent2_pos[0], agent2_pos[1],
                                                                length_agent, width_agent, agent2_pos[2])
                intersect_a_a = check_intersection(vertices_agent, vertices_agent2)
                if intersect_a_a == True:
                    agent_crash[i][j] = 1
                else:
                    agent_crash[i][j] = 0

        for i, landmark_crash_i in enumerate(landmark_crash):
            if np.any(landmark_crash_i == 1):
                action_n[i][0] = (action_n[i][0] * 4.9 - 3)/4.9
                dec_agent_inf[i][0] = 1
        for i, agent_crash_i in enumerate(agent_crash):
            if dec_agent_inf[i][0] == 1:
                continue
            else:
                if i <= 2:
                    for j_left, crash_flag_j_left in enumerate(agent_crash_i[:3]):
                        if crash_flag_j_left == 1:
                            agent_pos = agent_inf_real[i][:2]
                            veh_pos = agent_inf_real[j_left][:2]
                            angle_with_x_axis = np.arctan2(veh_pos[1] - agent_pos[1], veh_pos[0] - agent_pos[0])
                            angle_degrees = np.degrees(angle_with_x_axis)
                            if angle_degrees >= 270:
                                angle_degrees = angle_degrees - 360

                            if angle_degrees < -90:
                                angle_degrees = angle_degrees + 360

                            if -90 < angle_degrees <= 90:
                                if dec_agent_inf[i][0] == 1:
                                    action_n[i][0] = action_n[i][0]
                                else:
                                    action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
                                    dec_agent_inf[i][0] = 1
                if i > 2:
                    for j_straight, crash_flag_j_straight in enumerate(agent_crash_i[3:]):
                        if crash_flag_j_straight == 1:
                            agent_pos = agent_inf_real[i][:2]
                            veh_pos = agent_inf_real[j_straight][:2]
                            angle_with_x_axis = np.arctan2(veh_pos[1] - agent_pos[1],
                                                           veh_pos[0] - agent_pos[0])
                            angle_degrees = np.degrees(angle_with_x_axis)
                            if angle_degrees >= 270:
                                angle_degrees = angle_degrees - 360
                            if angle_degrees < -90:
                                angle_degrees = angle_degrees + 360

                            if 90 < angle_degrees <= 270:
                                if dec_agent_inf[i][0] == 1:
                                    action_n[i][0] = action_n[i][0]
                                else:
                                    action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
                                    dec_agent_inf[i][0] = 1

                if i <= 2:
                    if dec_agent_inf[i][0] == 1:
                        continue
                    else:
                        for j_left_2, crash_flag_j_left_2 in enumerate(agent_crash_i[3:]):
                            if crash_flag_j_left_2 == 1:
                                agent_v = np.linalg.norm(agent_inf_real[i, 3:5])
                                veh_v = np.linalg.norm(agent_inf_real[j_left_2, 3:5])
                                if agent_v < veh_v:
                                    if dec_agent_inf[i][0] == 1:
                                        action_n[i][0] = action_n[i][0]
                                    else:
                                        action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
                                        dec_agent_inf[i][0] = 1

                if i > 2:
                    if dec_agent_inf[i][0] == 1:
                        continue
                    else:
                        for j_straight_2, crash_flag_j_straight_2 in enumerate(agent_crash_i[:3]):
                            if crash_flag_j_straight_2 == 1:
                                agent_v = np.linalg.norm(agent_inf_real[i, 3:5])
                                veh_v = np.linalg.norm(agent_inf_real[j_straight_2, 3:5])
                                if agent_v < veh_v:
                                    if dec_agent_inf[i][0] == 1:
                                        action_n[i][0] = action_n[i][0]
                                    else:
                                        action_n[i][0] = (action_n[i][0] * 4.9 - 3) / 4.9
                                        dec_agent_inf[i][0] = 1

                        agent_speed_norm = np.linalg.norm(agent_inf_real[:, 3:5], axis=1)
                        max_speed = np.max(agent_speed_norm)
                        agent_speed = np.linalg.norm(agent_inf_real[i, 3:5])

                        if agent_speed == max_speed:
                            action_n[i][0] = (action_n[i][0] * 4.9 - 3)/4.9

        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i], 'second')
        self.world.step('second')

        for agent in self.agents:
            agent_id = agent.id
            obs_lstm, obs_usefor_reward, obs = self._get_obs(agent, action_n[agent_id], [False])
            obs_n.append(obs)
            obs_n_lstm.append(obs_lstm)
            reward_n.append(self._get_reward(agent, obs_usefor_reward))
            if action_n[agent_id][2] >= 184:  # 184 for training 229 for testing
                done_n.append(True)
            else:
                done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

            actions_new_n[agent_id][0] = agent.action.u[0]
            actions_new_n[agent_id][1] = agent.action.u[1]

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs, actions_new_n

    def reset(self):
        # reset world
        ini_obs, ini_steps, ini_obs_lstm = self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        obs_n_lstm = []
        reset_infos = [True]
        self.agents = self.world.policy_agents

        for agent in self.agents:
            action_n_agent = np.concatenate((agent.state.p_pos, np.array([0]),
                                             np.array([0]),
                                             np.array([ini_steps[agent.id][0]]), ini_obs[agent.id]))

            obs_lstm_old, obs_usefor_reward, obs_old = self._get_obs(agent, action_n_agent, reset_infos)
            obs_n.append(obs_old)
            obs_n_lstm.append(obs_lstm_old)
        return obs_n_lstm, obs_n, ini_steps, ini_obs, reset_infos, ini_obs_lstm

    def ini_obs_update(self, ini_obs_old_list):
        self.agents = self.world.policy_agents
        for agent in self.agents:
            if ini_obs_old_list[agent.id][-1] == True:
                agent.state.p_pos = ini_obs_old_list[agent.id][0:2]
                agent.state.p_vel = ini_obs_old_list[agent.id][2:4]
                agent.state.heading_angle_last1 = ini_obs_old_list[agent.id][6]
                agent.state.p_dis = ini_obs_old_list[agent.id][7]
                agent.state.p_ini_to_end_dis = ini_obs_old_list[agent.id][7]
                agent.state.p_last_vx = ini_obs_old_list[agent.id][8]
                agent.state.p_last_vy = ini_obs_old_list[agent.id][9]
                agent.state.ini_step = ini_obs_old_list[agent.id][57]
                agent.collide = False
                agent.end_label = False

                p_pos_x = agent.state.p_pos[0] * 38 - 4
                p_pos_y = agent.state.p_pos[1] * 23 + 14

                dx = ini_obs_old_list[agent.id][4] * 59 - 37
                dy = ini_obs_old_list[agent.id][5] * 27 - 4

                des_x = dx + p_pos_x
                des_y = dy + p_pos_y

                agent.state.p_des[0] = (des_x + 4) / 38
                agent.state.p_des[1] = (des_y - 14) / 23

        obs_n = []
        obs_n_lstm = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            action_n_agent = np.concatenate((agent.state.p_pos, np.array([ini_obs_old_list[agent.id][2]]),
                                             np.array([ini_obs_old_list[agent.id][3]]),
                                             np.array([ini_obs_old_list[agent.id][4]])))

            obs_lstm_old, obs_usefor_reward, obs_old = self._get_obs_nowstep(agent, action_n_agent)
            obs_n.append(obs_old)
            obs_n_lstm.append(obs_lstm_old)

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
    def _set_action(self, action, agent, action_space, cishu):
        path = r'\Data\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'  # path='/root/……/aus_openface.pkl'
        f_expert = open(path, 'rb')
        all_expert_trj_ = pickle.load(f_expert)

        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if (action[2] == action[4]) and (action[5] != 0 or action[6] != 0 or action[7] != 0 or action[8] != 0):  # action[5] != 0这个条件是为了不让场景内没有的agent被赋值

            agent.state.p_pos = action[5:7]
            agent.state.p_vel = action[7:9]
            agent.state.heading_angle_last1 = action[11]
            agent.state.p_dis = action[12]
            agent.state.p_ini_to_end_dis = action[12]
            agent.state.p_last_vx = action[13]
            agent.state.p_last_vy = action[14]
            agent.state.ini_step = action[4]
            agent.state.reference_line = np.array(all_expert_trj_[agent.state.scenario_id]['ob'][agent.id][:, :2])
            agent.state.min_distance = 0
            agent.collide = False
            agent.end_label = False

            p_pos_x = agent.state.p_pos[0] * 38 - 4
            p_pos_y = agent.state.p_pos[1] * 23 + 14

            dx = action[9] * 59 - 37
            dy = action[10] * 27 - 4

            des_x = dx + p_pos_x
            des_y = dy + p_pos_y

            agent.state.p_des[0] = (des_x + 4) / 38
            agent.state.p_des[1] = (des_y - 14) / 23

        if action[3] >= 0 and cishu == 'first':
            agent.state.step = agent.state.step + 1
        # process action
        acc = action[0]
        delta_theta = action[1]
        if agent.id < 3:
            acc = min(max(action[0], -1), 1)
            delta_theta = min(max(action[1], -1.3), 1.2)
        if agent.id >= 3:
            acc = min(max(action[0], -1), 1)
            delta_theta = min(max(action[1], -1), 1)

        agent.action.u[0] = acc
        agent.action.u[1] = delta_theta


    # reset rendering assets
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

    # create receptor field locations in local coordinate frame
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
