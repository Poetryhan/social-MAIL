import numpy as np
from shapely import geometry
from shapely.geometry import MultiPoint, MultiPolygon, box, Polygon
from shapely.geometry import Point
import pandas as pd
from scipy import interpolate
# enter_points = np.load('enter_points.npy', allow_pickle = True) ####

from shapely import geometry

left_lane_data = pd.read_csv(r'\Data\left-turning_lane.csv')

left_points = left_lane_data[['smooth_x', 'smooth_y']].values
polygon_left = Polygon(left_points)

def left_contain(x,y):
    point_to_check = Point(x, y)
    if polygon_left.contains(point_to_check):
        in_left_lane = True
    else:
        in_left_lane = False
    return in_left_lane

def left_bianjie_angle(x,y):
    left_bianjie_up = left_lane_data[left_lane_data['type']=='up']
    left_bianjie_down = left_lane_data[left_lane_data['type'] == 'down']
    left_bianjie_up.index = range(len(left_bianjie_up))
    left_bianjie_down.index = range(len(left_bianjie_down))
    distances_up = np.sqrt((left_bianjie_up['smooth_x'] - x) ** 2 + (left_bianjie_up['smooth_y'] - y) ** 2)
    distances_down = np.sqrt((left_bianjie_down['smooth_x'] - x) ** 2 + (left_bianjie_down['smooth_y'] - y) ** 2)
    nearest_indexup = distances_up.idxmin()
    nearest_indexdown = distances_down.idxmin()

    if distances_up[nearest_indexup] < distances_down[nearest_indexdown]:
        x_closest = left_bianjie_up['smooth_x'][nearest_indexup]
        y_closest = left_bianjie_up['smooth_y'][nearest_indexup]

        if nearest_indexup != len(left_bianjie_up)-1:
            x_last = left_bianjie_up['smooth_x'][nearest_indexup]
            y_last = left_bianjie_up['smooth_y'][nearest_indexup]
            x_next = left_bianjie_up['smooth_x'][nearest_indexup + 1]
            y_next = left_bianjie_up['smooth_y'][nearest_indexup + 1]
        else:
            x_last = left_bianjie_up['smooth_x'][nearest_indexup - 1]
            y_last = left_bianjie_up['smooth_y'][nearest_indexup - 1]
            x_next = x_closest
            y_next = y_closest

        direction_refup = np.array([x_next - x_last, y_next - y_last])
        heading_rad_refup = np.arctan2(direction_refup[1], direction_refup[0])
        heading_angle_refup = np.degrees(heading_rad_refup)
        if heading_angle_refup >= 270:
            heading_angle_refup = heading_angle_refup - 360

        if heading_angle_refup < -90:
            heading_angle_refup = heading_angle_refup + 360
        return heading_angle_refup, 'up'
    else:
        x_closest = left_bianjie_down['smooth_x'][nearest_indexdown]
        y_closest = left_bianjie_down['smooth_y'][nearest_indexdown]

        if nearest_indexdown != 0:
            x_last = left_bianjie_down['smooth_x'][nearest_indexdown]
            y_last = left_bianjie_down['smooth_y'][nearest_indexdown]
            x_next = left_bianjie_down['smooth_x'][nearest_indexdown-1]
            y_next = left_bianjie_down['smooth_y'][nearest_indexdown-1]
        else:
            x_last = left_bianjie_down['smooth_x'][nearest_indexdown + 1]
            y_last = left_bianjie_down['smooth_y'][nearest_indexdown + 1]
            x_next = x_closest
            y_next = y_closest

        direction_refdown = np.array([x_next - x_last, y_next - y_last])
        heading_rad_refdown = np.arctan2(direction_refdown[1], direction_refdown[0])
        heading_angle_refdown = np.degrees(heading_rad_refdown)
        if heading_angle_refdown >= 270:
            heading_angle_refdown = heading_angle_refdown - 360

        if heading_angle_refdown < -90:
            heading_angle_refdown = heading_angle_refdown + 360
        return heading_angle_refdown, 'down'

straight_lane_data = pd.read_csv(r'\Data\straight-through_lane.csv')

straight_points = straight_lane_data[['smooth_x', 'smooth_y']].values
polygon_straight = Polygon(straight_points)

def straight_contain(x,y):
    point_to_check = Point(x, y)
    if polygon_straight.contains(point_to_check):
        in_straight_lane = True
    else:
        in_straight_lane = False
    return in_straight_lane

def straight_bianjie_angle(x, y):
    straight_bianjie_up = straight_lane_data[straight_lane_data['type'] == 'up']
    straight_bianjie_down = straight_lane_data[straight_lane_data['type'] == 'down']
    distances_up = np.sqrt((straight_bianjie_up['smooth_x'] - x) ** 2 + (straight_bianjie_up['smooth_y'] - y) ** 2)
    distances_down = np.sqrt(
        (straight_bianjie_down['smooth_x'] - x) ** 2 + (straight_bianjie_down['smooth_y'] - y) ** 2)
    nearest_indexup = distances_up.idxmin()
    nearest_indexdown = distances_down.idxmin()

    if distances_up[nearest_indexup] < distances_down[nearest_indexdown]:
        x_closest = straight_bianjie_up['smooth_x'][nearest_indexup]
        y_closest = straight_bianjie_up['smooth_y'][nearest_indexup]

        if nearest_indexup != 0:
            x_last = straight_bianjie_up['smooth_x'][nearest_indexup]
            y_last = straight_bianjie_up['smooth_y'][nearest_indexup]
            x_next = straight_bianjie_up['smooth_x'][nearest_indexup - 1]
            y_next = straight_bianjie_up['smooth_y'][nearest_indexup - 1]
        else:
            x_last = straight_bianjie_up['smooth_x'][nearest_indexup + 1]
            y_last = straight_bianjie_up['smooth_y'][nearest_indexup + 1]
            x_next = x_closest
            y_next = y_closest

        direction_refup = np.array([x_next - x_last, y_next - y_last])
        heading_rad_refup = np.arctan2(direction_refup[1], direction_refup[0])
        heading_angle_refup = np.degrees(heading_rad_refup)
        if heading_angle_refup >= 270:
            heading_angle_refup = heading_angle_refup - 360

        if heading_angle_refup < -90:
            heading_angle_refup = heading_angle_refup + 360
        return heading_angle_refup, 'up'
    else:
        x_closest = straight_bianjie_down['smooth_x'][nearest_indexdown]
        y_closest = straight_bianjie_down['smooth_y'][nearest_indexdown]

        if nearest_indexdown != len(straight_bianjie_down)-1:
            x_last = straight_bianjie_down['smooth_x'][nearest_indexdown]
            y_last = straight_bianjie_down['smooth_y'][nearest_indexdown]
            x_next = straight_bianjie_down['smooth_x'][nearest_indexdown + 1]
            y_next = straight_bianjie_down['smooth_y'][nearest_indexdown + 1]
        else:
            x_last = straight_bianjie_down['smooth_x'][nearest_indexdown - 1]
            y_last = straight_bianjie_down['smooth_y'][nearest_indexdown - 1]
            x_next = x_closest
            y_next = y_closest

        direction_refdown = np.array([x_next - x_last, y_next - y_last])
        heading_rad_refdown = np.arctan2(direction_refdown[1], direction_refdown[0])
        heading_angle_refdown = np.degrees(heading_rad_refdown)
        if heading_angle_refdown >= 270:
            heading_angle_refdown = heading_angle_refdown - 360

        if heading_angle_refdown < -90:
            heading_angle_refdown = heading_angle_refdown + 360
        return heading_angle_refdown, 'down'


def cal_lane_width(x, y, direction):
    if direction == 'straight':
        lane_width = 7.2

    else:
        left_up = left_lane_data[left_lane_data['type'] == 'up']
        points_up = left_up[['smooth_x', 'smooth_y']].values
        points_df_up = pd.DataFrame(points_up)
        points_df_up.columns = ['x', 'y']
        points_df_up.index = range(len(points_df_up))
        points_df_up.drop(points_df_up[points_df_up['x'] == 0].index, inplace=True)
        points_df_up['x_real'] = points_df_up['x']
        points_df_up['y_real'] = points_df_up['y']
        x__up = points_df_up['x_real'].values
        y__up = points_df_up['y_real'].values

        points_use_up = np.array([x__up, y__up]).T

        tck_up, u_up = interpolate.splprep(points_use_up.T, s=0.0)
        u_new_up = np.linspace(u_up.min(), u_up.max(), 1000)
        x_new_up, y_new_up = interpolate.splev(u_new_up, tck_up, der=0)

        x0 = x
        y0 = y

        distances_test_up = np.sqrt((x_new_up - x0) ** 2 + (y_new_up - y0) ** 2)

        min_distance_up = distances_test_up.min()

        index_closest_up = np.argmin(distances_test_up)

        x_closest_up = x_new_up[index_closest_up]
        y_closest_up = y_new_up[index_closest_up]

        left_down = left_lane_data[left_lane_data['type'] == 'down']
        points_down = left_down[['smooth_x', 'smooth_y']].values
        points_df_down = pd.DataFrame(points_down)
        points_df_down.columns = ['x', 'y']
        points_df_down.index = range(len(points_df_down))
        points_df_down.drop(points_df_down[points_df_down['x'] == 0].index, inplace=True)
        points_df_down['x_real'] = points_df_down['x']
        points_df_down['y_real'] = points_df_down['y']
        x__down = points_df_down['x_real'].values
        y__down = points_df_down['y_real'].values

        points_use_down = np.array([x__down, y__down]).T

        tck_down, u_down = interpolate.splprep(points_use_down.T, s=0.0)
        u_new_down = np.linspace(u_down.min(), u_down.max(), 1000)
        x_new_down, y_new_down = interpolate.splev(u_new_down, tck_down, der=0)

        x0 = x
        y0 = y

        distances_test_down = np.sqrt((x_new_down - x0) ** 2 + (y_new_down - y0) ** 2)

        min_distance_down = distances_test_down.min()

        index_closest_down = np.argmin(distances_test_down)

        x_closest_down = x_new_down[index_closest_down]
        y_closest_down = y_new_down[index_closest_down]

        lane_width = np.sqrt((x_closest_down-x_closest_up)**2+(y_closest_down-y_closest_up)**2)
    return lane_width


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # self.p_deg = None
        self.p_des = None
        self.p_dis = None
        self.p_ini_to_end_dis = None
        self.p_last_vx = None
        self.p_last_vy = None
        self.ini_step = None
        self.step = None

        self.acc_x = None
        self.acc_y = None
        self.delta_accx = None
        self.delta_accy = None

        self.delta_angle_now = None
        self.delta_angle_last1 = None
        self.delta_angle_last2 = None
        self.heading_angle_now_state = None
        self.heading_angle_last1 = None
        self.heading_angle_last2 = None

        self.ini_step = None
        self.des_rew = None
        self.lane_rew = None
        self.heading_angle_rew = None
        self.delta_angle_rew = None
        self.heading_std_rew = None
        self.heading_chaochufanwei_rew = None
        self.collide_rew = None
        self.scenario_id = None
        self.reference_line = None

        self.angle_difference = None
        self.min_distance = None
        self.lane_width = None

        for i in range(250):
            setattr(self, f"ob_trj_step_{i}", None)

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = True
        # entity collides with others
        self.collide = False
        self.end_label = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        self.id = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        self.collide = False
        self.end_label = False
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 5.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.adversary = False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.num_left = 0
        self.num_straight = 0
        self.num_agents = 0
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.time = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents #+ self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self, cishu):
        self.integrate_state(0, cishu)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, unknow, cishu):
        for i, entity in enumerate(self.entities):
            ation = entity.action.u[:2]
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
                point = geometry.Point(x_true, y_true)
                if poly.contains(point):
                    return True
                else:
                    return False

            if entity.state.p_pos[0] != 0 or entity.state.p_pos[1] != 0 or entity.state.p_vel[0] != 0 or entity.state.p_vel[1] != 0:  # 这一步的位置是有效的，会得到相应的奖励

                ation_accer_guiyihua = ation[0]
                ation_yaw_guiyihua = ation[1]

                if entity.id <= 2:
                    action_acc = ((9.2 * (ation_accer_guiyihua + 1)) / 2) - 4.8
                    action_delta_angle = ((2.8 * (ation_yaw_guiyihua + 1)) / 2) - 0.3
                else:
                    action_acc = ((8.5 * (ation_accer_guiyihua + 1)) / 2) - 3.6
                    action_delta_angle = ((2.4 * (ation_yaw_guiyihua + 1)) / 2) - 1.2

                vx_now = (entity.state.p_vel[0] * 21) - 14
                vy_now = (entity.state.p_vel[1] * 12) - 2

                speed_now = np.sqrt((vx_now) ** 2 + (vy_now) ** 2)

                p_pos_x_now = entity.state.p_pos[0] * 38 - 4
                p_pos_y_now = entity.state.p_pos[1] * 23 + 14

                des_x_always = entity.state.p_des[0] * 38 - 4
                des_y_always = entity.state.p_des[1] * 23 + 14

                dist = np.sqrt(
                    np.sum(np.square([des_x_always - p_pos_x_now, des_y_always - p_pos_y_now])))

                speed = speed_now + action_acc * 0.1

                if speed < 0:  # 标量速度无负值
                    speed = 0

                degree_last = entity.state.heading_angle_last1 * 191 - 1
                degree_now = degree_last + action_delta_angle

                if degree_now >= 270:
                    degree_now = degree_now - 360

                if degree_now < -90:
                    degree_now = degree_now + 360

                entity.state.p_vel[0] = ((speed * np.cos(np.deg2rad(degree_now))) + 14) / 21
                entity.state.p_vel[1] = ((speed * np.sin(np.deg2rad(degree_now))) + 2) / 12

                a_x = action_acc * np.cos(np.deg2rad(degree_now))
                a_y = action_acc * np.sin(np.deg2rad(degree_now))

                dis_now_to_next = speed * 0.1

                p_pos_x_next = p_pos_x_now + dis_now_to_next * np.cos(
                    np.deg2rad(degree_now))
                p_pos_y_next = p_pos_y_now + dis_now_to_next * np.sin(np.deg2rad(degree_now))

                entity.state.p_pos[0] = (p_pos_x_next + 4) / 38
                entity.state.p_pos[1] = (p_pos_y_next - 14) / 23

                dis_to_end_next = np.sqrt((p_pos_x_next - des_x_always) ** 2 + (p_pos_y_next - des_y_always) ** 2)
                entity.state.p_dis = dis_to_end_next / 37

                entity.state.p_last_vx = (vx_now + 14) / 22
                entity.state.p_last_vy = (vy_now + 2) / 12

                if entity.state.step == 1:
                    heading_angle_last2_real = entity.state.heading_angle_last1 * 191 - 1
                else:
                    heading_angle_last2_real = entity.state.heading_angle_last2 * 191 - 1

                heading_angle_last1_real = entity.state.heading_angle_last1 * 191 - 1

                heading_angle_now_real = degree_now

                mean_value = np.mean([heading_angle_last1_real, heading_angle_last2_real, heading_angle_now_real])
                squared_differences = [(x - mean_value) ** 2 for x in
                                       [heading_angle_last1_real, heading_angle_last2_real, heading_angle_now_real]]
                mean_squared_difference = np.mean(squared_differences)
                std_dev = np.sqrt(mean_squared_difference)

                entity.state.heading_angle_last1 = (degree_now + 1) / 191
                entity.state.heading_angle_last2 = (heading_angle_last1_real + 1) / 191

                entity.state.delta_angle_last2 = entity.state.delta_angle_last1
                entity.state.delta_angle_last1 = ation_yaw_guiyihua

                points = entity.state.reference_line
                points_df = pd.DataFrame(points)
                points_df.columns = ['x','y']
                points_df.index = range(len(points_df))
                points_df.drop(points_df[points_df['x']==0].index, inplace=True)
                points_df['x_real'] = points_df['x'] * 38 - 4
                points_df['y_real'] = points_df['y'] * 23 + 14
                x_ = points_df['x_real'].values
                y_ = points_df['y_real'].values

                points_use = np.array([x_, y_]).T

                tck, u = interpolate.splprep(points_use.T, s=0.0)
                u_new = np.linspace(u.min(), u.max(), 1000)
                x_new, y_new = interpolate.splev(u_new, tck, der=0)

                x0 = entity.state.p_pos[0] * 38 - 4
                y0 = entity.state.p_pos[1] * 23 + 14

                distances_test = np.sqrt((x_new - x0) ** 2 + (y_new - y0) ** 2)

                min_distance = distances_test.min()

                index_closest = np.argmin(distances_test)

                x_closest = x_new[index_closest]
                y_closest = y_new[index_closest]

                if index_closest == 0:
                    x_last = x_new[index_closest]
                    y_last = y_new[index_closest]
                    x_closest = x_new[index_closest + 1]
                    y_closest = y_new[index_closest + 1]
                else:
                    x_last = x_new[index_closest - 1]
                    y_last = y_new[index_closest - 1]

                direction_ref = np.array([x_closest - x_last, y_closest - y_last])
                heading_rad_ref = np.arctan2(direction_ref[1], direction_ref[0])
                heading_angle_ref = np.degrees(heading_rad_ref)
                if heading_angle_ref >= 270:
                    heading_angle_ref = heading_angle_ref - 360

                if heading_angle_ref < -90:
                    heading_angle_ref = heading_angle_ref + 360

                if dis_to_end_next < 0.5:
                    entity.end_label = True
                    entity.state.p_pos = np.array([0, 0])
                    entity.state.p_vel = np.array([0, 0])

                if entity.id < 3:
                    if y0 > des_y_always:
                        entity.end_label = True
                        entity.state.p_pos = np.array([0, 0])
                        entity.state.p_vel = np.array([0, 0])
                if entity.id >= 3:
                    if x0 < des_x_always:
                        entity.end_label = True
                        entity.state.p_pos = np.array([0, 0])
                        entity.state.p_vel = np.array([0, 0])

                if entity.id <= 2:
                    direction = 'left'
                    lane_label = left_contain(x0, y0)
                else:
                    direction = 'straight'
                    lane_label = straight_contain(x0, y0)

                if lane_label == False:
                    heading_angle_xiuzheng2 = heading_angle_ref
                    delta_angle_xiuzheng = action_delta_angle
                    x_next_xiuzheng = p_pos_x_next
                    y_next_xiuzheng = p_pos_y_next
                    if direction == 'left':
                        heading_angle_xiuzheng, type = left_bianjie_angle(p_pos_x_now, p_pos_y_now)
                        if type == 'up':
                            if degree_now > heading_angle_xiuzheng:
                                if degree_now - heading_angle_xiuzheng >= 8:
                                    heading_angle_xiuzheng2 = degree_now - 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now <= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now
                            x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                        np.deg2rad(heading_angle_xiuzheng2))
                            y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                        np.deg2rad(heading_angle_xiuzheng2))
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last

                        if type == 'down':
                            if degree_now < heading_angle_xiuzheng:
                                if heading_angle_xiuzheng - degree_now >= 8:
                                    heading_angle_xiuzheng2 = degree_now + 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now >= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now

                            x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2))
                            y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2))
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last

                    else:
                        heading_angle_xiuzheng, type = straight_bianjie_angle(p_pos_x_now, p_pos_y_now)
                        if type == 'up':
                            if degree_now < heading_angle_xiuzheng:
                                if heading_angle_xiuzheng - degree_now >= 8:
                                    heading_angle_xiuzheng2 = degree_now + 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now >= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now
                            x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2))
                            y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2))
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                        if type == 'down':
                            if degree_now > heading_angle_xiuzheng:
                                if degree_now - heading_angle_xiuzheng >= 8:
                                    heading_angle_xiuzheng2 = degree_now - 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now <= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now

                            x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2))
                            y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2))
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last

                    if direction == 'left':
                        entity.action.u[1] = ((2*(delta_angle_xiuzheng+0.3))/2.8)-1
                    else:
                        entity.action.u[1] = ((2*(delta_angle_xiuzheng+1.2))/2.4)-1

                    degree_now_xiuzheng = heading_angle_xiuzheng2

                    if degree_now_xiuzheng >= 270:
                        degree_now_xiuzheng = degree_now_xiuzheng - 360

                    if degree_now_xiuzheng < -90:
                        degree_now_xiuzheng = degree_now_xiuzheng + 360

                    entity.state.p_vel[0] = ((speed * np.cos(np.deg2rad(degree_now_xiuzheng))) + 14) / 21
                    entity.state.p_vel[1] = ((speed * np.sin(np.deg2rad(degree_now_xiuzheng))) + 2) / 12

                    entity.state.p_pos[0] = (x_next_xiuzheng + 4) / 38
                    entity.state.p_pos[1] = (y_next_xiuzheng - 14) / 23

                    dis_to_end_next_xiuzheng = np.sqrt(
                        (x_next_xiuzheng - des_x_always) ** 2 + (y_next_xiuzheng - des_y_always) ** 2)
                    entity.state.p_dis = dis_to_end_next_xiuzheng / 37

                    entity.state.p_last_vx = (vx_now + 14) / 22
                    entity.state.p_last_vy = (vy_now + 2) / 12

                    entity.state.heading_angle_last1 = (degree_now_xiuzheng + 1) / 191
                    entity.state.heading_angle_last2 = (heading_angle_last1_real + 1) / 191

                x_next_new = entity.state.p_pos[0] * 38 - 4
                y_next_new = entity.state.p_pos[1] * 23 + 14

                x0_xiuzheng = entity.state.p_pos[0] * 38 - 4
                y0_xiuzheng = entity.state.p_pos[1] * 23 + 14

                distances_test_xiuzheng = np.sqrt((x_new - x0_xiuzheng) ** 2 + (y_new - y0_xiuzheng) ** 2)

                index_closest_xiuzheng = np.argmin(distances_test_xiuzheng)

                x_closest_xiuzheng = x_new[index_closest_xiuzheng]
                y_closest_xiuzheng = y_new[index_closest_xiuzheng]

                if index_closest_xiuzheng == 0:
                    x_last_xiuzheng = x_new[index_closest_xiuzheng]
                    y_last_xiuzheng = y_new[index_closest_xiuzheng]
                    x_closest_xiuzheng = x_new[index_closest_xiuzheng + 1]
                    y_closest_xiuzheng = y_new[index_closest_xiuzheng + 1]
                else:
                    x_last_xiuzheng = x_new[index_closest_xiuzheng - 1]
                    y_last_xiuzheng = y_new[index_closest_xiuzheng - 1]

                heading_rad_ref_xiuzheng = np.arctan2(direction_ref[1], direction_ref[0])
                heading_angle_ref_xiuzheng = np.degrees(heading_rad_ref_xiuzheng)
                if heading_angle_ref_xiuzheng >= 270:
                    heading_angle_ref_xiuzheng = heading_angle_ref_xiuzheng - 360

                if heading_angle_ref_xiuzheng < -90:
                    heading_angle_ref_xiuzheng = heading_angle_ref_xiuzheng + 360

                heading_angle_generated_xiuzheng = degree_now

                angle_difference_xiuzheng = abs(heading_angle_generated_xiuzheng - heading_angle_ref_xiuzheng)

                entity.state.angle_difference = angle_difference_xiuzheng
                if cishu == 'first':
                    lane_width = cal_lane_width(x0, y0, direction)
                    entity.state.lane_width = lane_width
                    entity.state.min_distance = min_distance/entity.state.lane_width

                insection_label = in_insection(x_next_new, y_next_new)
                if insection_label == False:
                    entity.end_label = True
                    entity.state.p_pos = np.array([0, 0])
                    entity.state.p_vel = np.array([0, 0])

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]