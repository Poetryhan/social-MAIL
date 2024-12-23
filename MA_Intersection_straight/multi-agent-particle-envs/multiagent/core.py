import numpy as np
from shapely import geometry
from shapely.geometry import MultiPoint, MultiPolygon, box, Polygon
from shapely.geometry import Point
import pandas as pd
from scipy import interpolate
# enter_points = np.load('enter_points.npy', allow_pickle = True) ####

# dependency = np.array([[0,4,5,6,7,13,7,13,6,7,13,9,10,0],[0,2, 0 ,0 ,0 ,0 ,0 , 0, 0,0 ,0 ,12,0,0 ]])
# dependency= np.transpose(dependency)

print('MAAIRL intersection!!!!')
from shapely import geometry
import math
# 运动规则
# 运动规则
# 需要加上左转车和直行车的范围
# 先定义左转车的范围
# 读取左转车道范围点
# left_lane_data = pd.read_csv(r'C:\Users\60106\Desktop\code-MA-ARIL\nvn_sametimego\lane_left.csv')
# 创建左转车道多边形
# left_points = left_lane_data[['x', 'y']].values
# polygon_left = Polygon(left_points)

left_lane_data = pd.read_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
                             r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight'
                             r'\AV_test\DATA\左转车道范围_平滑.csv')
# 创建左转车道多边形
left_points = left_lane_data[['smooth_x', 'smooth_y']].values
polygon_left = Polygon(left_points)

# 绘制左转车道多边形
def left_contain(x,y):

    # 创建一个点对象
    point_to_check = Point(x, y)  # 替换x_coordinate和y_coordinate为要检查的点的坐标

    # 判断点是否在多边形内
    if polygon_left.contains(point_to_check):
        # print("点在多边形内")
        in_left_lane = True
    else:
        # print("点在多边形外")
        in_left_lane = False
    return in_left_lane

def left_bianjie_angle(x,y):
    left_bianjie_up = left_lane_data[left_lane_data['type']=='up']
    left_bianjie_down = left_lane_data[left_lane_data['type'] == 'down']
    left_bianjie_up.index = range(len(left_bianjie_up))
    left_bianjie_down.index = range(len(left_bianjie_down))
    distances_up = np.sqrt((left_bianjie_up['smooth_x'] - x) ** 2 + (left_bianjie_up['smooth_y'] - y) ** 2)
    distances_down = np.sqrt((left_bianjie_down['smooth_x'] - x) ** 2 + (left_bianjie_down['smooth_y'] - y) ** 2)
    # 找到每条边界上距离点 (x, y) 最近的点的索引
    nearest_indexup = distances_up.idxmin()
    nearest_indexdown = distances_down.idxmin()

    # 比较距离，找到距离最小的那个边界
    if distances_up[nearest_indexup] < distances_down[nearest_indexdown]:
        # 更靠近上边线
        # 获取最近点的坐标
        x_closest = left_bianjie_up['smooth_x'][nearest_indexup]
        y_closest = left_bianjie_up['smooth_y'][nearest_indexup]

        # 获取上一个点的坐标（假设曲线上有足够的点）
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
        # 计算车道参考线上的方向向量
        direction_refup = np.array([x_next - x_last, y_next - y_last])
        # 计算方向角度（弧度）
        heading_rad_refup = np.arctan2(direction_refup[1], direction_refup[0])
        heading_angle_refup = np.degrees(heading_rad_refup)
        if heading_angle_refup >= 270:
            heading_angle_refup = heading_angle_refup - 360  # 这里的角度范围是【-90，270】

        if heading_angle_refup < -90:
            heading_angle_refup = heading_angle_refup + 360  # 这里的角度范围是【-90，270】
        return heading_angle_refup, 'up'
    else:
        # 更靠近下边线
        # 获取最近点的坐标
        x_closest = left_bianjie_down['smooth_x'][nearest_indexdown]
        y_closest = left_bianjie_down['smooth_y'][nearest_indexdown]

        # 获取上一个点的坐标（假设曲线上有足够的点）
        if nearest_indexdown != 0:  # 对于下边线来说，这不是最后一个点的index 0
            x_last = left_bianjie_down['smooth_x'][nearest_indexdown]
            y_last = left_bianjie_down['smooth_y'][nearest_indexdown]
            x_next = left_bianjie_down['smooth_x'][nearest_indexdown-1]
            y_next = left_bianjie_down['smooth_y'][nearest_indexdown-1]
        else:  # 是最后一个点
            x_last = left_bianjie_down['smooth_x'][nearest_indexdown + 1]
            y_last = left_bianjie_down['smooth_y'][nearest_indexdown + 1]
            x_next = x_closest
            y_next = y_closest

        # 计算车道参考线上的方向向量
        direction_refdown = np.array([x_next - x_last, y_next - y_last])
        # 计算方向角度（弧度）
        heading_rad_refdown = np.arctan2(direction_refdown[1], direction_refdown[0])
        heading_angle_refdown = np.degrees(heading_rad_refdown)
        if heading_angle_refdown >= 270:
            heading_angle_refdown = heading_angle_refdown - 360  # 这里的角度范围是【-90，270】

        if heading_angle_refdown < -90:
            heading_angle_refdown = heading_angle_refdown + 360  # 这里的角度范围是【-90，270】
        return heading_angle_refdown, 'down'

# 先定义直行车的范围
# 读取直行车道范围点
straight_lane_data = pd.read_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
                                 r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight'
                                 r'\AV_test\DATA\直行车道范围_平滑.csv')
# 创建直行车道多边形
straight_points = straight_lane_data[['smooth_x', 'smooth_y']].values
polygon_straight = Polygon(straight_points)

# 绘制直行车道多边形
def straight_contain(x,y):

    # 创建一个点对象
    point_to_check = Point(x, y)  # 替换x_coordinate和y_coordinate为要检查的点的坐标

    # 判断点是否在多边形内
    if polygon_straight.contains(point_to_check):
        # print("点在多边形内")
        in_straight_lane = True
    else:
        # print("点在多边形外")
        in_straight_lane = False
    return in_straight_lane


def straight_bianjie_angle(x, y):
    straight_bianjie_up = straight_lane_data[straight_lane_data['type'] == 'up']
    straight_bianjie_down = straight_lane_data[straight_lane_data['type'] == 'down']
    distances_up = np.sqrt((straight_bianjie_up['smooth_x'] - x) ** 2 + (straight_bianjie_up['smooth_y'] - y) ** 2)
    distances_down = np.sqrt(
        (straight_bianjie_down['smooth_x'] - x) ** 2 + (straight_bianjie_down['smooth_y'] - y) ** 2)
    # 找到每条边界上距离点 (x, y) 最近的点的索引
    nearest_indexup = distances_up.idxmin()
    nearest_indexdown = distances_down.idxmin()

    # 比较距离，找到距离最小的那个边界
    if distances_up[nearest_indexup] < distances_down[nearest_indexdown]:
        # 更靠近上边线
        # 获取最近点的坐标
        x_closest = straight_bianjie_up['smooth_x'][nearest_indexup]
        y_closest = straight_bianjie_up['smooth_y'][nearest_indexup]

        # 获取上一个点的坐标（假设曲线上有足够的点）
        if nearest_indexup != 0:  # 不是最后一个点
            x_last = straight_bianjie_up['smooth_x'][nearest_indexup]
            y_last = straight_bianjie_up['smooth_y'][nearest_indexup]
            x_next = straight_bianjie_up['smooth_x'][nearest_indexup - 1]
            y_next = straight_bianjie_up['smooth_y'][nearest_indexup - 1]
        else:  # 是最后一个点
            x_last = straight_bianjie_up['smooth_x'][nearest_indexup + 1]
            y_last = straight_bianjie_up['smooth_y'][nearest_indexup + 1]
            x_next = x_closest
            y_next = y_closest
        # 计算车道参考线上的方向向量
        direction_refup = np.array([x_next - x_last, y_next - y_last])
        # 计算方向角度（弧度）
        heading_rad_refup = np.arctan2(direction_refup[1], direction_refup[0])
        heading_angle_refup = np.degrees(heading_rad_refup)
        if heading_angle_refup >= 270:
            heading_angle_refup = heading_angle_refup - 360  # 这里的角度范围是【-90，270】

        if heading_angle_refup < -90:
            heading_angle_refup = heading_angle_refup + 360  # 这里的角度范围是【-90，270】
        return heading_angle_refup, 'up'
    else:
        # 更靠近下边线
        # 获取最近点的坐标
        x_closest = straight_bianjie_down['smooth_x'][nearest_indexdown]
        y_closest = straight_bianjie_down['smooth_y'][nearest_indexdown]

        # 获取上一个点的坐标（假设曲线上有足够的点）
        if nearest_indexdown != len(straight_bianjie_down)-1:  # 对于下边线来说，这不是最后一个点的index
            x_last = straight_bianjie_down['smooth_x'][nearest_indexdown]
            y_last = straight_bianjie_down['smooth_y'][nearest_indexdown]
            x_next = straight_bianjie_down['smooth_x'][nearest_indexdown + 1]
            y_next = straight_bianjie_down['smooth_y'][nearest_indexdown + 1]
        else:  # 是最后一个点
            x_last = straight_bianjie_down['smooth_x'][nearest_indexdown - 1]
            y_last = straight_bianjie_down['smooth_y'][nearest_indexdown - 1]
            x_next = x_closest
            y_next = y_closest

        # 计算车道参考线上的方向向量
        direction_refdown = np.array([x_next - x_last, y_next - y_last])
        # 计算方向角度（弧度）
        heading_rad_refdown = np.arctan2(direction_refdown[1], direction_refdown[0])
        heading_angle_refdown = np.degrees(heading_rad_refdown)
        if heading_angle_refdown >= 270:
            heading_angle_refdown = heading_angle_refdown - 360  # 这里的角度范围是【-90，270】

        if heading_angle_refdown < -90:
            heading_angle_refdown = heading_angle_refdown + 360  # 这里的角度范围是【-90，270】
        return heading_angle_refdown, 'down'


def cal_lane_width(x, y, direction):  # 输入的是真实的轨迹点位置
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

        # 准备轨迹点的数据
        points_use_up = np.array([x__up, y__up]).T

        # 利用scipy的spline插值
        # print('轨迹参考线的points', points_use, points_use.T)

        tck_up, u_up = interpolate.splprep(points_use_up.T, s=0.0)
        u_new_up = np.linspace(u_up.min(), u_up.max(), 1000)
        x_new_up, y_new_up = interpolate.splev(u_new_up, tck_up, der=0)

        # 下一个时刻的点的位置A(x,y)
        x0 = x
        y0 = y

        # 计算点A到曲线的欧式距离
        distances_test_up = np.sqrt((x_new_up - x0) ** 2 + (y_new_up - y0) ** 2)

        # 获取最小距离
        min_distance_up = distances_test_up.min()
        # print('distances_test:',distances_test)

        index_closest_up = np.argmin(distances_test_up)

        # 获取最近点的坐标
        x_closest_up = x_new_up[index_closest_up]
        y_closest_up = y_new_up[index_closest_up]

        # 下边界
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

        # 准备轨迹点的数据
        points_use_down = np.array([x__down, y__down]).T

        # 利用scipy的spline插值
        # print('轨迹参考线的points', points_use, points_use.T)

        tck_down, u_down = interpolate.splprep(points_use_down.T, s=0.0)
        u_new_down = np.linspace(u_down.min(), u_down.max(), 1000)
        x_new_down, y_new_down = interpolate.splev(u_new_down, tck_down, der=0)

        # 下一个时刻的点的位置A(x,y)
        x0 = x
        y0 = y

        # 计算点A到曲线的欧式距离
        distances_test_down = np.sqrt((x_new_down - x0) ** 2 + (y_new_down - y0) ** 2)

        # 获取最小距离
        min_distance_down = distances_test_down.min()
        # print('distances_test:',distances_test)

        index_closest_down = np.argmin(distances_test_down)

        # 获取最近点的坐标
        x_closest_down = x_new_down[index_closest_down]
        y_closest_down = y_new_down[index_closest_down]

        lane_width = np.sqrt((x_closest_down-x_closest_up)**2+(y_closest_down-y_closest_up)**2)
        # print('左转车的车道宽度',lane_width)
    return lane_width


# 这一部分的创建交叉口的范围，坐标是如何确定的？根据实际的坐标吗？拍摄的实地交叉口
# poly = geometry.Polygon([(997,985),
# (1045,987),
# (1040,1017),
# (997,1015),
# (997,985)])

# 导入int_map文件,110*4，代表了什么？
# int_shape = np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\int_map.npy',allow_pickle=True)
# int_shape2 = np.array(int_shape)
# int_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape]
# poly_int = geometry.Polygon(int_shape)


# for i in range(len(int_shape)):
#     if int_shape2[i,0]==980:
#         int_shape2[i,0] = 975
#     if int_shape2[i,0]==1060:
#         int_shape2[i,0] = 1065
#     if int_shape2[i,1]==970:
#         int_shape2[i,1] = 965
#     if int_shape2[i,1]==1030:
#         int_shape2[i,1] = 1035
# int_shape2 = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape2]
# poly_int2 = geometry.Polygon(int_shape2)

# def in_intersection(x,y):
#    point = geometry.Point(x,y)
#    if poly_int.contains(point):
#        return True
#    else :
#        return False


def collision_intersection(x,y, degree = 0, scale = 80):
    point = np.zeros([4,2])
    x_diff = 2.693 *np.cos(np.radians(degree + 21.8)) /scale
    y_diff = 2.693 *np.sin(np.radians(degree + 21.8)) /scale
    x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
    y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
    point[0] = [x + x_diff, y + y_diff]
    point[1] = [x + x_diff1, y + y_diff1]
    point[2] = [x - x_diff, y - y_diff]
    point[3] = [x - x_diff1, y - y_diff1]
    # print(point)

    for i in range(4):
        point1 = geometry.Point(point[i])

        if poly_int2.contains(point1):
            continue
        else :
            return True

    return False

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # self.p_deg = None
        self.p_des = None
        self.p_dis = None # 和终点的距离
        self.p_ini_to_end_dis = None # 初始点和终点的距离
        self.p_last_vx = None  # 上一个点的vx
        self.p_last_vy = None  # 上一个点的vy
        self.ini_step = None  # 轨迹开始的step
        self.step = None # agent当前的步长

        self.acc_x = None  # 当前时刻的横向加速度，目前没有值，必须执行了环境才有值，但是在step时，不更新这个值，也就是对于step之后的值来说，这个acc_x是上一时刻的值
        self.acc_y = None
        self.delta_accx = None  # 上一时刻的agent和交互对象的加速度差
        self.delta_accy = None

        self.delta_angle_now = None  # 当前时刻的steering_rad
        self.delta_angle_last1 = None  # 上一时刻的steering_rad
        self.delta_angle_last2 = None  # 上上时刻的steering_rad
        self.heading_angle_now_state = None  # 当前时刻的heading_rad
        self.heading_angle_last1 = None  # 上一时刻的heading_rad
        self.heading_angle_last2 = None  # 上上一时刻的heading_rad

        self.ini_step = None  # 轨迹开始的step
        self.des_rew = None  # 当生成轨迹到达终点时，赋予的奖励，到了就是10，没有就是0
        self.lane_rew = None  # 当生成轨迹离开车道范围时，赋予的奖励，离开了就是-10，没有就是0
        self.heading_angle_rew = None  # 当生成轨迹点的交互和车道中心线对应的角度差别太大时，给与惩罚，差别不大就是0
        self.delta_angle_rew = None  # 当前时刻、上一时刻、上上时刻的steering_rad的rew  正负号
        self.heading_std_rew = None  # 当前时刻、上一时刻、上上时刻的heading_rad的rew  标准差过大带来的惩罚
        self.heading_chaochufanwei_rew = None  # 当前时刻的角度如果超出了真实数据的范围，就会给负值
        self.collide_rew = None  # 当前时刻的位置信息为0，是因为此前时刻导致的无效，所以不再给trj_intersection_4中的GT_rew，直接给collide_rew=0
        self.scenario_id = None  # 场景编号
        self.reference_line = None  # 这辆车的参考线

        self.angle_difference = None  # 当前时刻和车道参考线轨迹角度的差异
        self.min_distance = None  # 当前时刻和车道参考线的距离
        self.lane_width = None  # 当前时刻对应的车道宽度

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
        self.end_label = False  #  车辆到达轨迹终点附近
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
        self.adversary = None

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
        self.end_label = False  # 车辆到达轨迹终点附近
        # cannot send communication signals
        self.silent = True # 原来是False
        self.adversary = True
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

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
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
        self.num_left = 0
        self.num_straight = 0
        self.num_agents = 0


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
        # print('运行的是nvn的core')
        # set actions for scripted agents
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        # p_force = [None] * len(self.entities)
        # # apply agent physical controls
        # p_force = self.apply_action_force(p_force)
        # # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(0, cishu)
        # update agent state
        # for agent in self.agents:
        #     self.update_agent_state(agent)

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
        # print('len(entities):',len(self.entities))
        for i, entity in enumerate(self.entities):
            ation = entity.action.u[:2]
            # 原本的交叉口边界
            # poly = geometry.Polygon([(5.21, 34.36),
            #                          (2.10, 27.47), (-4.50, 25.39),
            #                          (-4.13, 6.41),
            #                          (3.87, 4.70),
            #                          (7.29, -2.64),
            #                          (22.26, -2.57),
            #                          (25.23, 3.74),
            #                          (33.61, 6.41),
            #                          (33.83, 25.76),
            #                          (24.56, 28.06),
            #                          (20.34, 34.73)])

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

            if entity.state.p_pos[0] != 0:
            # if entity.state.p_pos[0] != 0 or entity.state.p_pos[1] != 0 or entity.state.p_vel[0] != 0 or entity.state.p_vel[1] != 0:  # 这一步的位置是有效的，会得到相应的奖励
            #     print('core中的scenario_test:',entity.state.scenario_id, '归一化的坐标：',entity.state.p_pos, 'step:',entity.state.step)
                ation_accer_guiyihua = ation[0]
                ation_yaw_guiyihua = ation[1]

                if entity.id <= 2:
                    action_acc = ((9.2 * (ation_accer_guiyihua + 1)) / 2) - 4.8
                    action_delta_angle = ((2.8 * (ation_yaw_guiyihua + 1)) / 2) - 0.3
                    # if ation_yaw_guiyihua > 1.2 or ation_yaw_guiyihua < -1.3:
                    # print('core左转车的转向角为：', entity.id, ation_yaw_guiyihua, ation[1], action_delta_angle)
                else:
                    action_acc = ((8.5 * (ation_accer_guiyihua + 1)) / 2) - 3.6
                    action_delta_angle = ((2.4 * (ation_yaw_guiyihua + 1)) / 2) - 1.2
                    # if ation_yaw_guiyihua > 1 or ation_yaw_guiyihua < -1:
                    # print('core直行车的转向角为：', entity.id, ation_yaw_guiyihua, ation[1], action_delta_angle)

                vx_now = (entity.state.p_vel[0] * 21) - 14
                vy_now = (entity.state.p_vel[1] * 12) - 2

                speed_now = np.sqrt((vx_now) ** 2 + (vy_now) ** 2)  # 当前时刻真实的速度 根号下(vx2+vy2)

                p_pos_x_now = entity.state.p_pos[0] * 38 - 4
                p_pos_y_now = entity.state.p_pos[1] * 23 + 14

                des_x_always = entity.state.p_des[0] * 38 - 4
                des_y_always = entity.state.p_des[1] * 23 + 14

                # dist = np.sqrt(
                #     np.sum(np.square([des_x_always - p_pos_x_now, des_y_always - p_pos_y_now])))  # 当前时刻和终点的距离

                speed = speed_now + action_acc * 0.1  # v1 = v0 + a0t

                if speed < 0:  # 标量速度无负值
                    speed = 0

                degree_last = entity.state.heading_angle_last1 * 191 - 1  # 上一时刻的角度，这里的角度范围是【-90，270】

                # # 先转换为0-360度
                # if degree_last < 0:
                #     degree_last = degree_last + 360 # 这里的角度范围是【0，360】

                degree_now = degree_last + action_delta_angle  # 【0-，360+】之间，可能会大于360，或者小于0

                if degree_now >= 270:
                    degree_now = degree_now - 360  # 这里的角度范围是【-90，270】

                if degree_now < -90:
                    degree_now = degree_now + 360  # 这里的角度范围是【-90，270】

                # # 如果当前时刻这个agent的angle超出了真实轨迹的范围，就给予一定的惩罚
                # if entity.id < 3:  # 左转车
                #     if degree_now > 104 or degree_now < -1:
                #         entity.state.heading_chaochufanwei_rew = -1
                #     else:
                #         entity.state.heading_chaochufanwei_rew = 0
                # else:  # 直行车
                #     if degree_now > 190 or degree_now < 176:
                #         entity.state.heading_chaochufanwei_rew = -1
                #     else:
                #         entity.state.heading_chaochufanwei_rew = 0

                entity.state.p_vel[0] = ((speed * np.cos(np.deg2rad(degree_now))) + 14) / 21
                entity.state.p_vel[1] = ((speed * np.sin(np.deg2rad(degree_now))) + 2) / 12

                # a_x = action_acc * np.cos(np.deg2rad(degree_now))  # 真实的横向加速度
                # a_y = action_acc * np.sin(np.deg2rad(degree_now))

                dis_now_to_next = speed * 0.1

                p_pos_x_next = p_pos_x_now + (dis_now_to_next * np.cos(
                    np.deg2rad(degree_now)))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                p_pos_y_next = p_pos_y_now + (dis_now_to_next * np.sin(np.deg2rad(degree_now)))  # 计算下一个时刻的y

                entity.state.p_pos[0] = (p_pos_x_next + 4) / 38
                entity.state.p_pos[1] = (p_pos_y_next - 14) / 23

                dis_to_end_next = np.sqrt((p_pos_x_next - des_x_always) ** 2 + (p_pos_y_next - des_y_always) ** 2)
                entity.state.p_dis = dis_to_end_next / 37

                entity.state.p_last_vx = (vx_now + 14) / 22
                entity.state.p_last_vy = (vy_now + 2) / 12

                # # 计算一些rew
                # # 计算当前时刻，上个时刻，上上个时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                # if entity.state.step == 1:  # 第一步的时候没有上上时刻的角度，所以以上个时刻的角度来代替（一开始无论左转还是直行几乎都是直行的角度）
                #     heading_angle_last2_real = (entity.state.heading_angle_last1 * 191) - 1  # [0,1]
                # else:
                #     heading_angle_last2_real = (entity.state.heading_angle_last2 * 191) - 1  # [0,1]
                #
                # heading_angle_last1_real = (entity.state.heading_angle_last1 * 191) - 1  # [0,1]

                heading_angle_now_real = degree_now  # 当前时刻的heading_rad_now

                # # 计算平均值
                # mean_value = np.mean([heading_angle_last1_real, heading_angle_last2_real, heading_angle_now_real])
                # # 计算每个数据与平均值的差的平方
                # squared_differences = [(x - mean_value) ** 2 for x in
                #                        [heading_angle_last1_real, heading_angle_last2_real, heading_angle_now_real]]
                # # 计算平方差的平均值
                # mean_squared_difference = np.mean(squared_differences)
                # # 计算标准差
                # std_dev = np.sqrt(mean_squared_difference)
                # if entity.id < 3:  # 左转车
                #     if std_dev > 3:
                #         entity.state.heading_std_rew = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                #     else:
                #         entity.state.heading_std_rew = 1
                # else:  # 直行车
                #     if std_dev > 1.05:
                #         entity.state.heading_std_rew = np.exp(-(std_dev - 1.05))
                #     else:
                #         entity.state.heading_std_rew = 1

                # 更新下一个时刻的 上个时刻，上上个时刻 heading
                entity.state.heading_angle_last1 = (degree_now + 1) / 191
                # entity.state.heading_angle_last2 = (heading_angle_last1_real + 1) / 191

                # # 计算steering angle正负来回变化带来的惩罚
                # if entity.state.step == 1:
                #     entity.state.delta_angle_last2 = entity.state.delta_angle_last1  #
                #     entity.state.delta_angle_last1 = entity.state.delta_angle_last1  #
                #
                # if entity.id <= 2:
                #     delta_angle_last2_real = ((2.8*(entity.state.delta_angle_last2 + 1))/2) - 0.3
                #     delta_angle_last1_real = ((2.8*(entity.state.delta_angle_last1 + 1))/2) - 0.3
                # else:
                #     delta_angle_last2_real = ((2.4 * (entity.state.delta_angle_last2 + 1)) / 2) - 1.2
                #     delta_angle_last1_real = ((2.4 * (entity.state.delta_angle_last1 + 1)) / 2) - 1.2
                # # entity.state.delta_angle_now = (action_delta_angle + 3) / 7  # [0,1]
                #
                # if delta_angle_last2_real > 0 and delta_angle_last1_real < 0 and action_delta_angle > 0:
                #     entity.state.delta_angle_rew = -1
                # elif delta_angle_last2_real < 0 and delta_angle_last1_real > 0 and action_delta_angle < 0:
                #     entity.state.delta_angle_rew = -1
                # else:
                #     entity.state.delta_angle_rew = 0
                #
                # # 更新下一个时刻 上一时刻 上上时刻的steering rad
                #
                # entity.state.delta_angle_last2 = entity.state.delta_angle_last1  #
                # entity.state.delta_angle_last1 = ation_yaw_guiyihua  #


                points = entity.state.reference_line
                points_df = pd.DataFrame(points)
                points_df.columns = ['x','y']
                points_df.index = range(len(points_df))
                points_df.drop(points_df[points_df['x']==0].index, inplace=True)
                points_df['x_real'] = points_df['x'] * 38 - 4
                points_df['y_real'] = points_df['y'] * 23 + 14
                x_ = points_df['x_real'].values
                y_ = points_df['y_real'].values

                # 准备轨迹点的数据
                points_use = np.array([x_, y_]).T

                # 利用scipy的spline插值
                # print('轨迹参考线的points', points_use, points_use.T)

                tck, u = interpolate.splprep(points_use.T, s=0.0)
                u_new = np.linspace(u.min(), u.max(), 1000)
                x_new, y_new = interpolate.splev(u_new, tck, der=0)

                # plt.figure()
                # plt.plot(points[:, 0], points[:, 1], 'ro')
                # plt.plot(x_new, y_new, 'b--')
                # plt.show()

                # 下一个时刻的点的位置A(x,y)
                x0 = entity.state.p_pos[0] * 38 - 4
                y0 = entity.state.p_pos[1] * 23 + 14

                # 计算点A到曲线的欧式距离
                distances_test = np.sqrt((x_new - x0) ** 2 + (y_new - y0) ** 2)

                # 获取最小距离
                min_distance = distances_test.min()
                # print('distances_test:',distances_test)

                index_closest = np.argmin(distances_test)

                # 获取最近点的坐标
                x_closest = x_new[index_closest]
                y_closest = y_new[index_closest]

                # 获取上一个点的坐标（假设曲线上有足够的点）
                if index_closest == 0:
                    x_last = x_new[index_closest]
                    y_last = y_new[index_closest]
                    x_closest = x_new[index_closest + 1]
                    y_closest = y_new[index_closest + 1]
                else:
                    x_last = x_new[index_closest - 1]
                    y_last = y_new[index_closest - 1]
                # 计算车道参考线上的方向向量
                direction_ref = np.array([x_closest - x_last, y_closest - y_last])
                # 计算方向角度（弧度）
                heading_rad_ref = np.arctan2(direction_ref[1], direction_ref[0])
                heading_angle_ref = np.degrees(heading_rad_ref)
                if heading_angle_ref >= 270:
                    heading_angle_ref = heading_angle_ref - 360 # 这里的角度范围是【-90，270】

                if heading_angle_ref < -90:
                    heading_angle_ref = heading_angle_ref + 360 # 这里的角度范围是【-90，270】
                #
                # # 获取生成器生成的点的方向角度（假设是某个变量）
                # heading_angle_generated = degree_now  # 下一时刻的上一时刻的heading
                #
                # # 计算角度差异
                # angle_difference = abs(heading_angle_generated - heading_angle_ref)
                #
                # # 设置阈值，如果差异过大，则给一个惩罚
                # threshold_angle_difference = 15  # 替换为实际的阈值
                #
                # if angle_difference > threshold_angle_difference:
                #     # 给一个惩罚，例如将奖励值减小
                #     entity.state.heading_angle_rew = np.exp(-angle_difference)  # 1 / abs(angle_difference - threshold_angle_difference)
                #     # entity.state.heading_angle_rew = -1 * abs(angle_difference - threshold_angle_difference)  # 替换为实际的奖励计算
                #     # print("Apply Penalty!")
                # else:
                #     entity.state.heading_angle_rew = np.exp(-angle_difference)  # 0 没有惩罚

                # if min_distance > 1:  # 超过了轨迹线1m，需要把动作约束回来
                #     # print(f"The minimum distance from point A to the curve is: {min_distance, entity.id, x0, y0}")
                #     # entity.collide = True
                #     entity.state.lane_rew = np.exp(-min_distance)
                #     # entity.state.lane_rew = -1*(min_distance-1)
                #     # entity.state.p_pos = np.array([0, 0])
                # else:
                #     entity.state.lane_rew = np.exp(-min_distance)  # 当最靠近轨迹的时候，奖励最大为1，越远越小。其实不太需要上述if then的，但是为了和之前的形式保持一致，就没有改


                if dis_to_end_next < 0.5: # 下一个点接近终点
                    entity.end_label = True
                    entity.state.des_rew = 1
                    entity.state.p_pos = np.array([0, 0])
                    entity.state.p_vel = np.array([0, 0])
                    # print('core函数里这个轨迹点接近终点', entity.id, p_pos_x_now, p_pos_y_now)
                    # entity.state.p_deg = degree_now
                else:
                    entity.state.des_rew = 0

                if entity.id < 3: # 左转车，根据y来判断是否超过边界
                    # print('左转车：', 'y0', y0, 'des_y_always', des_y_always)
                    if y0 > des_y_always:  #  and x0 > des_x_always:
                        # print('左转车core函数里这个轨迹点的下一时刻超出终点了','左转车：', 'x0', x0, 'y0', y0, 'p_pos_y_now', p_pos_y_now, 'des_x_always', des_x_always, 'des_y_always', des_y_always)
                        # entity.state.p_pos = np.array([(p_pos_x_now + 28) / 55, (p_pos_y_now + 4) / 26])
                        # entity.state.p_vel = np.array([0, 0])
                        # entity.state.p_deg = degree_now

                        entity.end_label = True
                        entity.state.p_pos = np.array([0, 0])
                        entity.state.p_vel = np.array([0, 0])


                if entity.id >= 3:  # 直行车，根据x来判断是否超过边界
                    # if entity.id == 3:
                    #     print('直行车：', 'x0', x0, 'des_x_always', des_x_always)
                    if x0 < des_x_always:
                        # print('直行车core函数里这个轨迹点的下一时刻超出终点了','直行车：', 'x0', x0,'y0',y0, 'p_pos_x_now', p_pos_x_now, 'des_x_always', des_x_always, 'des_y_always', des_y_always)
                        # entity.state.p_pos = np.array([(p_pos_x_now + 28) / 55, (p_pos_y_now + 4) / 26])
                        # entity.state.p_vel = np.array([0, 0])
                        # entity.state.p_deg = degree_now

                        entity.end_label = True
                        entity.state.p_pos = np.array([0, 0])
                        entity.state.p_vel = np.array([0, 0])

                if entity.id <= 2:
                    direction = 'left'
                    lane_label = left_contain(x0, y0)
                else:
                    direction = 'straight'
                    lane_label = straight_contain(x0, y0)

                if lane_label == False:  # 下一时刻这个点不在车道内
                    # print('core函数里这个轨迹点的下一时刻超出车道边界了', x0, y0, '上一时刻的位置：', p_pos_x_now, p_pos_y_now,
                    #       '动作：', action_acc, action_delta_angle, 'speed:', speed, 'degree:', degree_now)
                    # entity.collide = True
                    # entity.state.p_pos = np.array([0, 0])
                    entity.state.collide_rew = -1.5
                    # # 如果轨迹点超出范围或者下一时刻到达终点，会赋予奖励值，但是与此同时state.pos也为0，
                    # 所以这个rew没有起作用 # 这种情况在trj_intersection_4中进行了处理

                    # 修改动作，重新更新状态，作为无越界约束
                    # 首先找到距离车辆最近的车道边界
                    heading_angle_xiuzheng2 = heading_angle_ref
                    delta_angle_xiuzheng = action_delta_angle
                    x_next_xiuzheng = p_pos_x_next
                    y_next_xiuzheng = p_pos_y_next
                    if direction == 'left':
                        heading_angle_xiuzheng, type = left_bianjie_angle(p_pos_x_now, p_pos_y_now)
                        if type == 'up':
                            # 轨迹靠近上边线
                            if degree_now > heading_angle_xiuzheng:
                                if degree_now - heading_angle_xiuzheng >= 8:
                                    heading_angle_xiuzheng2 = degree_now - 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now <= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now

                            # else:
                            #     heading_angle_xiuzheng2 = heading_angle_xiuzheng
                                # for i in range(6):
                                #     heading_angle_xiuzheng2 = heading_angle_xiuzheng - 2 * i
                                #     x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                #     y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的y
                                #     lane_label_xiuzheng = left_contain(x_next_xiuzheng, y_next_xiuzheng)
                                #     if lane_label_xiuzheng == True:  # 下个点在车道内
                                #         delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                                #         break  # 退出循环
                                #     else:
                                #         continue  # 继续下一次迭代
                            x_next_xiuzheng = p_pos_x_now + (dis_now_to_next * np.cos(
                                        np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                            y_next_xiuzheng = p_pos_y_now + (dis_now_to_next * np.sin(
                                        np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的y
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                            # print('左转车原本上一时刻角度:', degree_last, '原本的这一时刻角度:', degree_now,
                            #       '上边界的角度', heading_angle_xiuzheng,
                            #       '左转车接近上边界,修正之后的这一时刻的角度：', heading_angle_xiuzheng2, '位置', x_next_xiuzheng,
                            #       y_next_xiuzheng,
                            #       '修正之后的转向角', delta_angle_xiuzheng)
                        if type == 'down':
                            # 轨迹靠近下边线
                            if degree_now < heading_angle_xiuzheng:
                                if heading_angle_xiuzheng - degree_now >= 8:
                                    heading_angle_xiuzheng2 = degree_now + 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now >= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now

                            x_next_xiuzheng = p_pos_x_now + (dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                            y_next_xiuzheng = p_pos_y_now + (dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的y
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                            # print('左转车原本上一时刻角度:', degree_last, '原本的这一时刻角度:', degree_now,
                            #       '下边界的角度', heading_angle_xiuzheng,
                            #       '左转车接近下边界,修正之后的这一时刻的角度：', heading_angle_xiuzheng2, '位置', x_next_xiuzheng,
                            #       y_next_xiuzheng,
                            #       '修正之后的转向角', delta_angle_xiuzheng)
                    else:  # 直行车
                        heading_angle_xiuzheng, type = straight_bianjie_angle(p_pos_x_now, p_pos_y_now)

                        if type == 'up':
                            # 轨迹靠近上边线
                            if degree_now < heading_angle_xiuzheng:
                                if heading_angle_xiuzheng - degree_now >= 8:
                                    heading_angle_xiuzheng2 = degree_now + 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now >= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now
                            # if abs(heading_angle_xiuzheng - degree_last) >= 10:
                            #     heading_angle_xiuzheng2 = degree_last + 10
                            # else:
                            #     heading_angle_xiuzheng2 = heading_angle_xiuzheng
                                # for i in range(18):
                                #     heading_angle_xiuzheng2 = heading_angle_xiuzheng + 5 * i
                                #     x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                #     y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的y
                                #     lane_label_xiuzheng = straight_contain(x_next_xiuzheng, y_next_xiuzheng)
                                #     if lane_label_xiuzheng == True:  # 下个点在车道内
                                #         delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                                #         break  # 退出循环
                                #     else:
                                #         continue  # 继续下一次迭代
                            x_next_xiuzheng = p_pos_x_now + (dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                            y_next_xiuzheng = p_pos_y_now + (dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的y
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                            # print('直行车原本上一时刻角度:', degree_last, '原本的这一时刻角度:', degree_now,
                            #       '上边界的角度', heading_angle_xiuzheng,
                            #       '直行车接近上边界,修正之后的这一时刻的角度：', heading_angle_xiuzheng2, '位置', x_next_xiuzheng,
                            #       y_next_xiuzheng,
                            #       '修正之后的转向角', delta_angle_xiuzheng)
                        if type == 'down':
                            # 轨迹靠近下边线
                            if degree_now > heading_angle_xiuzheng:
                                if degree_now - heading_angle_xiuzheng >= 8:
                                    heading_angle_xiuzheng2 = degree_now - 8
                                else:
                                    heading_angle_xiuzheng2 = heading_angle_xiuzheng
                            elif degree_now <= heading_angle_xiuzheng:
                                heading_angle_xiuzheng2 = degree_now
                            # if abs(heading_angle_xiuzheng - degree_last) >= 10:
                            #     heading_angle_xiuzheng2 = degree_last - 10
                            # else:
                            #     heading_angle_xiuzheng2 = heading_angle_xiuzheng
                                # for i in range(18):
                                #     heading_angle_xiuzheng2 = heading_angle_xiuzheng - 5 * i
                                #     x_next_xiuzheng = p_pos_x_now + dis_now_to_next * np.cos(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                #     y_next_xiuzheng = p_pos_y_now + dis_now_to_next * np.sin(
                                #         np.deg2rad(heading_angle_xiuzheng2))  # 计算下一个时刻的y
                                #     lane_label_xiuzheng = straight_contain(x_next_xiuzheng, y_next_xiuzheng)
                                #     print('直行车第',i,'次改变角度：',heading_angle_xiuzheng2,x_next_xiuzheng,y_next_xiuzheng,lane_label_xiuzheng)
                                #     if lane_label_xiuzheng == True:  # 下个点在车道内
                                #         delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                                #         break  # 退出循环
                                #     else:
                                #         continue  # 继续下一次迭代

                            x_next_xiuzheng = p_pos_x_now + (dis_now_to_next * np.cos(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                            y_next_xiuzheng = p_pos_y_now + (dis_now_to_next * np.sin(
                                np.deg2rad(heading_angle_xiuzheng2)))  # 计算下一个时刻的y
                            delta_angle_xiuzheng = heading_angle_xiuzheng2 - degree_last
                            # print('直行车原本上一时刻角度:', degree_last, '原本的这一时刻角度:', degree_now,
                            #       '下边界的角度', heading_angle_xiuzheng,
                            #       '直行车接近下边界,修正之后的这一时刻的角度：', heading_angle_xiuzheng2, '位置', x_next_xiuzheng,
                            #       y_next_xiuzheng,
                            #       '修正之后的转向角', delta_angle_xiuzheng)

                    # 更新智能体的状态参数
                    if direction == 'left':
                        entity.action.u[1] = ((2*(delta_angle_xiuzheng+0.3))/2.8)-1
                    else:
                        entity.action.u[1] = ((2*(delta_angle_xiuzheng+1.2))/2.4)-1

                    degree_now_xiuzheng = heading_angle_xiuzheng2  # 【0-，360+】之间，可能会大于360，或者小于0

                    if degree_now_xiuzheng >= 270:
                        degree_now_xiuzheng = degree_now_xiuzheng - 360  # 这里的角度范围是【-90，270】

                    if degree_now_xiuzheng < -90:
                        degree_now_xiuzheng = degree_now_xiuzheng + 360  # 这里的角度范围是【-90，270】

                    entity.state.p_vel[0] = ((speed * np.cos(np.deg2rad(degree_now_xiuzheng))) + 14) / 21
                    entity.state.p_vel[1] = ((speed * np.sin(np.deg2rad(degree_now_xiuzheng))) + 2) / 12

                    entity.state.p_pos[0] = (x_next_xiuzheng + 4) / 38
                    entity.state.p_pos[1] = (y_next_xiuzheng - 14) / 23

                    dis_to_end_next_xiuzheng = np.sqrt(
                        (x_next_xiuzheng - des_x_always) ** 2 + (y_next_xiuzheng - des_y_always) ** 2)
                    entity.state.p_dis = dis_to_end_next_xiuzheng / 37

                    entity.state.p_last_vx = (vx_now + 14) / 22
                    entity.state.p_last_vy = (vy_now + 2) / 12

                    # 更新下一个时刻的 上个时刻，上上个时刻 heading
                    entity.state.heading_angle_last1 = (degree_now_xiuzheng + 1) / 191
                    # entity.state.heading_angle_last2 = (heading_angle_last1_real + 1) / 191

                else:
                    # entity.collide = False
                    entity.state.collide_rew = 0

                x_next_new = (entity.state.p_pos[0] * 38) - 4
                y_next_new = (entity.state.p_pos[1] * 23) + 14
                #
                # # 再计算一遍新的轨迹点距离车道的距离
                # # 下一个时刻的点的位置A(x,y)
                # x0_xiuzheng = (entity.state.p_pos[0] * 38) - 4
                # y0_xiuzheng = (entity.state.p_pos[1] * 23) + 14
                #
                # # 计算点A到曲线的欧式距离
                # distances_test_xiuzheng = np.sqrt((x_new - x0_xiuzheng) ** 2 + (y_new - y0_xiuzheng) ** 2)
                #
                # # 获取最小距离
                # min_distance_xiuzheng = distances_test_xiuzheng.min()
                # # print('distances_test:',distances_test)
                #
                # index_closest_xiuzheng = np.argmin(distances_test_xiuzheng)
                #
                # # 获取最近点的坐标
                # x_closest_xiuzheng = x_new[index_closest_xiuzheng]
                # y_closest_xiuzheng = y_new[index_closest_xiuzheng]
                #
                # # 获取上一个点的坐标（假设曲线上有足够的点）
                # if index_closest_xiuzheng == 0:
                #     x_last_xiuzheng = x_new[index_closest_xiuzheng]
                #     y_last_xiuzheng = y_new[index_closest_xiuzheng]
                #     x_closest_xiuzheng = x_new[index_closest_xiuzheng + 1]
                #     y_closest_xiuzheng = y_new[index_closest_xiuzheng + 1]
                # else:
                #     x_last_xiuzheng = x_new[index_closest_xiuzheng - 1]
                #     y_last_xiuzheng = y_new[index_closest_xiuzheng - 1]
                # # 计算车道参考线上的方向向量
                # direction_ref_xiuzheng = np.array([x_closest_xiuzheng - x_last_xiuzheng, y_closest_xiuzheng - y_last_xiuzheng])
                # # 计算方向角度（弧度）
                # heading_rad_ref_xiuzheng = np.arctan2(direction_ref[1], direction_ref[0])
                # heading_angle_ref_xiuzheng = np.degrees(heading_rad_ref_xiuzheng)
                # if heading_angle_ref_xiuzheng >= 270:
                #     heading_angle_ref_xiuzheng = heading_angle_ref_xiuzheng - 360  # 这里的角度范围是【-90，270】
                #
                # if heading_angle_ref_xiuzheng < -90:
                #     heading_angle_ref_xiuzheng = heading_angle_ref_xiuzheng + 360  # 这里的角度范围是【-90，270】
                #
                # # 获取生成器生成的点的方向角度（假设是某个变量）
                # heading_angle_generated_xiuzheng = degree_now  # 下一时刻的上一时刻的heading
                #
                # # 计算角度差异
                # angle_difference_xiuzheng = abs(heading_angle_generated_xiuzheng - heading_angle_ref_xiuzheng)
                #
                # # 设置阈值，如果差异过大，则给一个惩罚
                # threshold_angle_difference_xiuzheng = 15  # 替换为实际的阈值
                # entity.state.angle_difference = angle_difference_xiuzheng
                # 计算这个点对应的车道宽度
                if cishu == 'first':
                    lane_width = cal_lane_width(x0, y0, direction)
                    entity.state.lane_width = lane_width
                    entity.state.min_distance = min_distance/entity.state.lane_width

                insection_label = in_insection(x_next_new, y_next_new)
                if insection_label == False:  # 这个点不在交叉口内
                    # print('core函数里这个轨迹点的下一时刻超出交叉口边界了',x_next_new, y_next_new,'上一时刻的位置：', p_pos_x_now,p_pos_y_now,
                    #       '动作：',action_acc, action_delta_angle,'speed:',speed,'degree:',degree_now)
                    entity.end_label = True
                    entity.state.p_pos = np.array([0, 0])
                    entity.state.p_vel = np.array([0, 0])
                    # entity.state.collide_rew = -1.5
                    # # 如果轨迹点超出范围或者下一时刻到达终点，会赋予奖励值，但是与此同时state.pos也为0，
                    # 所以这个rew没有起作用 # 这种情况在trj_intersection_4中进行了处理
                # else:
                #     # 这个点在交叉口内
                #     if entity.end_label == True:
                #         entity.state.p_pos = np.array([0, 0])
                #     else:
                #         entity.end_label = False
                    # entity.state.collide_rew = 0

            else:  # 这一步的位置是无效的（因为上一步有效，但是导致这一步无效；或者上一步也是无效），那么奖励就为0，但是
                entity.state.des_rew = 0  # 终点奖励
                entity.state.lane_rew = 0  # 车道中心线距离奖励
                entity.state.heading_angle_rew = 0  # 车道中心线角度奖励
                entity.state.delta_angle_rew = 0  # 转向角平滑奖励
                entity.state.heading_std_rew = 0  # 航向角平滑奖励
                entity.state.heading_chaochufanwei_rew = 0
                entity.state.collide_rew = 0
            # if entity.id < 36: # 左转车
            #     ation = entity.action.u[:2]
            #     # print('上次报错的地方的action:', ation, np.shape(ation))
            #     # print('上次报错的地方的action[0]:', ation[0][0], np.shape(ation))
            #     # print('上次报错的地方的action[1]:', ation[0][1], np.shape(ation))
            #     ation[0][1] = min(max(ation[0][1],-3.51), 99.09)
            #
            # elif entity.id >= 36: # 直行车
            #     # ation = entity.action.u[2:4]
            #     ation[0][1] = min(max(ation[0][1], 164.08), 176.8)



            # if  entity.state.p_pos[0] != 0:


            # speed = (entity.state.p_vel[0] **2 + entity.state.p_vel[1] **2) **0.5
            #
            # dist = np.sqrt(np.sum(np.square(entity.state.p_des - entity.state.p_pos)))
            # # potential_pos =  entity.action.u
            #
            # # if not collision_intersection(potential_pos[0],potential_pos[1], entity.state.p_deg,80):
            # # if not collision_intersection(potential_pos[0],potential_pos[1], entity.state.p_deg,80):
            # # # if in_intersection(potential_pos[0],potential_pos[1]):
            #
            #
            # speed += ation[0][0]*0.125  # 更新下一个时刻的速度
            # if speed < 0:
            #     speed = 0

            # else: potential_pos = entity.state.p_pos
            # if entity.action.u[0] < 10 and entity.action.u[0] >0:
                # move_x = (entity.action.u[0]) * np.cos(np.deg2rad(entity.state.p_deg)) /15
                # move_y = (entity.action.u[0]) * np.sin(np.deg2rad(entity.state.p_deg)) /15

                # if np.sign(np.cos(np.deg2rad(entity.state.p_deg))) == np.sign(entity.state.p_vel[0] + move_x)
            # 原始的写法，p_deg应该是角度
            # entity.state.p_vel[0] = speed * np.cos(np.deg2rad(entity.state.p_deg))
            # entity.state.p_vel[1] = speed * np.sin(np.deg2rad(entity.state.p_deg))

            # # 更新下一个时刻的分速度
            # entity.state.p_vel[0] = speed * np.cos(math.radians(ation[0][1]))
            # entity.state.p_vel[1] = speed * np.sin(math.radians(ation[0][1]))
            #
            # potential_pos = entity.state.p_pos + entity.state.p_vel * 0.125
            # entity.state.p_pos = potential_pos

                # if entity.size == 1:
                #     entity.action.u[1] = min(max(entity.action.u[1],0.0),0.7)
                # elif entity.size == 2:
                #     entity.action.u[1] = min(max(entity.action.u[1],-0.1),0.1)
                # elif entity.size == 3:
                #     entity.action.u[1] = min(max(entity.action.u[1],-0.6),0.0)
            # entity.state.p_deg = ation[0][1]

                # movement = np.array([0,0])
                # movement[0] = abs(entity.action.u[0]) * np.cos(np.deg2rad(entity.state.p_deg))
                # movement[1] = abs(entity.action.u[0]) * np.sin(np.deg2rad(entity.state.p_deg))

                # for other in self.entities:
                #     distance = np.sqrt(np.sum(np.square(entity.state.p_pos - other.state.p_pos)))
                #     # print(entity.state.p_pos,distance)
                #     if distance < 2/80 and distance > 0  : #and entity.state.p_pos[0] < 0.3779
                #         if not (entity.state.p_pos[0] == 0.42 and entity.state.p_pos[1] == 0.37):
                #             entity.state.p_pos = np.array([0,0])
                #             entity.state.p_vel = np.array([0,0])
                #             entity.state.p_deg = 10
                #             entity.state.p_des = np.array([0,0])
                #             print('collision!!!')
                #             # print(entity.state.p_pos)
                #             break

            # 感觉不需要这一部分代码 这是最后时刻的修正
            # if dist < 0.01 and entity.id < 36: # 左转车
            #     # entity.state.p_pos = np.array([0,0])
            #     entity.state.p_vel = np.array([0,0])
            #     # entity.state.p_deg = 90 # 90度
            #     entity.state.p_des = np.array([0,0])
            #
            # if dist < 0.01 and entity.id >= 36:  # 直行车
            #     # entity.state.p_pos = np.array([0, 0])
            #     entity.state.p_vel = np.array([0, 0])
            #     # entity.state.p_deg = 0  # 0度
            #     entity.state.p_des = np.array([0, 0])

                # if entity.state.p_pos[0] == 0.42 and entity.state.p_pos[1] == 0.37:
                #     # entity.state.p_pos = np.array([0.4,0.4])
                #     entity.state.p_vel = np.array([0,0])
                    # np.sqrt(np.sum(np.square(np.array([0.37,0.41]) - np.array([0.37795,	0.424775]))))

                # potential_pos = entity.state.p_pos + movement / 80

            # else:
            #     entity.state.p_deg = 10
            #
            #     entity.state.p_pos = np.array([0,1])
            #     entity.state.p_vel = np.array([0,0])
            #     entity.state.p_deg = 10
            #     entity.state.p_des = np.array([0,0])

                # if (entity.state.p_pos[0] > 1 or entity.state.p_pos[0] < 0 or entity.state.p_pos[1] > 0.75 or entity.state.p_pos[1] < 0):
                #
                #     entity.state.p_deg = 0

           # if entity.action.u[1] and entity.size == 3: #link
            #     a = entity.state.p_pos + entity.action.u
            #     if a[1]>0 and a[1]<=3:
            #         entity.state.p_pos[1] += entity.action.u[1]
            # elif entity.action.u[0]>0 and entity.size == 1: #veh
            #     # if entity.state.p_pos[0]==0:
            #     #     if entity.action.u[0]==1:
            #     #         entity.state.p_pos[0] = np.random.choice([1,2],[3,4])
            #     #     elif entity.action.u[0]==2:
            #     #         entity.state.p_pos[0] = np.random.choice([8,11])
            #     # else:
            #     #     if entity.action.u[0]<= len(dependency[int(entity.state.p_pos[0])]):
            #     #         entity.state.p_pos[0] = dependency[int(entity.state.p_pos[0])][int(entity.action.u[0])-1]



            #     entity.state.p_pos[0] = dependency[int(entity.state.p_pos[0])][int(entity.action.u[0])-1]


                # nex_link = dependency[int(entity.action.u[0])-1][int(entity.state.p_pos[0])]
                # if nex_link:
                #     entity.state.p_pos[0] = nex_link[0]
                # else:
                #     entity.action.u[0] = 10



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