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
import joblib
from rl.acktr.utils import Scheduler, find_trainable_variables
from rl.acktr.utils import fc, mse
from rl.acktr import kfac
from irl.mack.tf_util import relu_layer, linear, tanh_layer

# init_pointss_4   117*18*26
# init_pointss = np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\multi-agent-trj\expert_trjs\init_pointss_tess_6times_guiyihua.npy', allow_pickle=True)  ####
# init_num = np.load('init_num.npy', allow_pickle=True)
# dependency = np.load('dependency.npy', allow_pickle = True)
# all_vehss = (list(np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\all_vehss.npy', allow_pickle= True)))  # 117*179
# int_shape = np.load(r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\int_map.npy',allow_pickle=True) # 88*2
# int_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in int_shape]
# poly_int = geometry.Polygon(int_shape)

# 这里的agent是否指每个进口道的agent数目？是如何确定的？和轨迹数之间的关系是什么？
n1 = list(range(0, 2))
n2 = list(range(2, 6))

# 运动规则
# 需要加上左转车和直行车的范围
# 先定义左转车的范围
# 读取左转车道范围点
# left_lane_data = pd.read_csv(r'C:\Users\60106\Desktop\code-MA-ARIL\nvn_sametimego\lane_left.csv')
# 创建左转车道多边形
# left_points = left_lane_data[['x', 'y']].values
# polygon_left = Polygon(left_points)

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

# 先定义直行车的范围
# 读取直行车道范围点
# straight_lane_data = pd.read_csv(r'C:\Users\60106\Desktop\code-MA-ARIL\nvn_sametimego\lane_straight.csv')
# 创建直行车道多边形
# straight_points = straight_lane_data[['x', 'y']].values
# polygon_straight = Polygon(straight_points)

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


# int_shape2 = [tuple([0,0]), tuple([0,0.75]), tuple([1,0.75]) ,tuple([1,0]),tuple([0,0])]

# poly_int = geometry.Polygon(int_shape)
# poly_int2 = geometry.Polygon(int_shape2)

# intera_shape = [(997,985),
# (1045,987),
# (1040,1017),
# (997,1015),
# (997,985)]
# intera_shape = [tuple([(ac[0]-980)/80,(ac[1]-970)/80]) for ac in intera_shape]
# poly_intera = geometry.Polygon(intera_shape)
# plt.plot(a)

def collision_intersection2(x, y, degree=0, scale=80):
    # point = np.zeros([4,2])
    x_diff = 50 * np.cos(np.radians(degree)) / scale
    y_diff = 50 * np.sin(np.radians(degree)) / scale
    # x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
    # y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
    point = geometry.Point([x + x_diff, y + y_diff])
    point2 = geometry.Point([x, y])
    # point[1] = [x + x_diff1, y + y_diff1]
    # point[2] = [x - x_diff, y - y_diff]
    # point[3] = [x - x_diff1, y - y_diff1]
    # print(point)

    if poly_int.contains(point):
        return 1
    else:

        path = geometry.LineString([point, point2])
        a = path.intersection(poly_int)
        # path.distance(poly_int)

        if not a.intersects(poly_int2.boundary):
            return 50 / scale - a.distance(point)
        else:
            return 1


def dis_stop_line(x, y, degree=0, scale=80):
    # point = np.zeros([4,2])

    x_diff = 50 * np.cos(np.radians(degree)) / scale
    y_diff = 50 * np.sin(np.radians(degree)) / scale
    # x_diff1 = 2.693 *np.cos(np.radians(degree - 21.8))/scale
    # y_diff1 = 2.693 *np.sin(np.radians(degree - 21.8))/scale
    point = geometry.Point([x + x_diff, y + y_diff])
    point2 = geometry.Point([x, y])
    # point[1] = [x + x_diff1, y + y_diff1]
    # point[2] = [x - x_diff, y - y_diff]
    # point[3] = [x - x_diff1, y - y_diff1]
    # print(point)

    if poly_intera.contains(point2):
        return 1
    else:

        path = geometry.LineString([point, point2])
        a = path.intersection(poly_intera)

        if a.intersects(poly_intera):
            return a.distance(point2)
        else:
            return 1


def in_intersection(x, y):
    point = geometry.Point(x, y)
    if poly_int.contains(point):
        return True
    else:
        return False


def collision_intersection(x, y, degree=0, scale=80):
    point = np.zeros([4, 2])
    x_diff = 2.693 * np.cos(np.radians(degree + 21.8)) / scale
    y_diff = 2.693 * np.sin(np.radians(degree + 21.8)) / scale
    x_diff1 = 2.693 * np.cos(np.radians(degree - 21.8)) / scale
    y_diff1 = 2.693 * np.sin(np.radians(degree - 21.8)) / scale
    point[0] = [x + x_diff, y + y_diff]
    point[1] = [x + x_diff1, y + y_diff1]
    point[2] = [x - x_diff, y - y_diff]
    point[3] = [x - x_diff1, y - y_diff1]
    # print(point)

    for i in range(4):
        point1 = geometry.Point(point[i])

        if poly_int.contains(point1):
            continue
        else:
            return True

    return False


# collision_intersection2 (0.6,0.5, -45)

class Scenario(BaseScenario):
    # def __init__(self):

    # 上面这个函数，也是为了确保每个场景都会被按顺序选择，直到循环到所有场景而设置的初始函数
    def make_world(self, scenario_test, training_label):
        # print('运行的是nvn的Scenario')
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
        print('world中的num_agents为：',world.num_agents)

        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        # for i, agent in enumerate(world.agents):
        #     agent.name = 'agent %d' % i
        #     agent.collide = False
        #     agent.silent = True
        #     agent.adversary =  False
        #     agent.size = 0.15

        for i in range(num_agents):
            world.agents[i].name = 'agent %d' % i
            world.agents[i].collide = False
            world.agents[i].silent = True  # world.agents[i].silent = True 通常表示智能体在仿真中处于 "静默" 或 "不通信" 的状态。这意味着智能体被禁止与其他智能体进行通信，但通常并不表示它没有观测值。
            world.agents[i].adversary = False  # 通过将 world.agents[i].adversary 设置为 False，您可以明确指示该智能体不属于对抗性智能体，而是属于非对抗性或协作性智能体。这个标记有助于区分不同类型的智能体，并可能在环境中的任务和规则方面产生不同的影响。

        self.reset_world(world, scenario_test, training_label)  # 调用 reset_world 函数来初始化仿真世界的其他属性和状态。
        # print('运行这个函数了吗？')
        return world

    def reset_world(self, world, scenario_test, training_label):  # 这段代码是用于重置仿真世界的状态和智能体的初始配置的函数
        init_pointss = np.load(
            r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
            r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight'
            r'\AV_test\DATA\init_sinD_nvnxuguan_9jiaohu_social_dayu1_v2.npy', allow_pickle=True)

        path_expert = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight' \
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

        path_landmark = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                        r'\ATT-social-iniobs_rlyouhua' \
                        r'\MA_Intersection_straight\AV_test\DATA\sinD_nvnxuguan_9jiaohu_landmarks_buxianshi.pkl'  # [95,97,6,14,6]
        # (list(np.load('all_vehss.npy', allow_pickle=True)))

        # init_pointss = np.load(
        #     r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\init_east_left_social_dayu1.npy', allow_pickle=True)
        #
        # path_expert = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_social_dayu1.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        #
        # path_landmark = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\测试数据_东左\east_left_landmarks.pkl'  # [95,97,6,14,6]

        f_expert = open(path_expert, 'rb')
        all_expert_trj = pickle.load(f_expert)

        f_landmark = open(path_landmark, 'rb')
        all_landmark_trj = pickle.load(f_landmark)

        # aa = np.random.choice([np.random.randint(0, 49), np.random.randint(50, 90)])
        # aa = np.random.randint(0, 90) #0(0, 450) # 44现在是训练集的数据  所有的scenario (0，36)

        if training_label:  # 训练环境
            choices = [x for x in range(0, 49)] + [x for x in range(50, 69)] + [x for x in range(70, 90)]
            # 从列表中随机选择一个数字
            aa = np.random.choice(choices)
            print('初始场景：', aa)
        else:  # 测试环境
            aa = np.random.randint(scenario_test, scenario_test + 1)
            print('初始场景：', aa)

        init_points = init_pointss[aa]
        # print('init_points:',init_points[0])
        # init_points[1] = np.zeros(58)  # 加了这个！需要个性化修改，对于测试场景来说，用AV替换哪一个agent
        ini_steps = init_points[:, 57:58]
        world.landmarks = np.array(all_landmark_trj[aa]['ob'])  # [30,185,7] 归一化之后的数据
        # print('world.landmarks:',np.shape(world.landmarks))  # (30, 185, 7)

        # 下面这几行代码也是随机选择训练的场景，加了一个位置信息是不0的判断
        # reset_inf = False
        # while not reset_inf:
        #     if init_points[0][0] != 0 and init_points[4][0] != 0:  # 第一辆左转车为有效值并且第一辆直行车也为有效值，也就是pos_x不为0
        #         init_points = init_pointss[aa]
        #         reset_inf = True
        #     else:
        #         # 至少有一辆是无效的，也就是这个场景要么只有一辆左转车，要么只有一辆直行车，要么都没有
        #         aa = np.random.randint(1000, 1001)
        #         init_points = init_pointss[aa]
        #         reset_inf = False

        for i in range(world.num_agents):  # 初始化
            world.agents[i].state.p_pos = np.zeros(2)
            world.agents[i].state.p_vel = np.zeros(2)

            world.agents[i].state.p_des = np.zeros(2)
            world.agents[i].state.p_dis = 0

            # world.agents[i].state.heading_rad = 0  # 当前时刻的heading_rad
            world.agents[i].state.p_ini_to_end_dis = 0
            world.agents[i].state.p_last_vx = 0
            world.agents[i].state.p_last_vy = 0
            world.agents[i].state.delta_angle_now = 0  # 当前时刻的steering_rad
            world.agents[i].state.delta_angle_last1 = 0  # 上一时刻的steering_rad
            world.agents[i].state.delta_angle_last2 = 0  # 上上时刻的steering_rad
            world.agents[i].state.heading_angle_last1 = 0  # 上一时刻的steering_rad
            world.agents[i].state.heading_angle_last2 = 0  # 上上时刻的steering_rad
            world.agents[i].state.acc_x = None  # 当前时刻的横向加速度，目前没有值，必须执行了环境才有值，但是在step时，不更新这个值，也就是对于step之后的值来说，这个acc_x是上一时刻的值
            world.agents[i].state.acc_y = None
            world.agents[i].state.delta_accx = 0  # 上一时刻的agent和交互对象的加速度差
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
            world.agents[i].state.min_distance = 0  # 初始时刻是真实的位置，所以距离参考线的距离是0
            world.agents[i].state.delta_angle_last1 = 0  # 上一时刻的转向角

            world.agents[i].size = 2
            world.agents[i].id = i


        for i in range(world.num_agents):
            #  'x', 'y', 'Vx[m/s]', 'Vy[m/s]', 'dx', 'dy', 'heading_angle_last_1','des_length','Vx_last_1','Vy_last_1'
            #  'left_deltax', 'left_deltay', 'left_deltavx', 'left_deltavy',
            #  'right_deltax', 'right_deltay', 'right_deltavx', 'right_deltavy'

            if init_points[i][57] == 0:  # 如果这个agent在第一个时刻就可以进入交叉口，那么就将这些初始值赋给这个agent，否则的话agent所有的state都为0
                print(i,'init_points[i][0]',init_points[i][0], init_points[i][1], init_points[i][2], init_points[i][3])
                if init_points[i][0] == 0:
                # if init_points[i][0] == 0 and init_points[i][1] == 0 and init_points[i][2] == 0 and init_points[i][3] == 0:
                    print('无效，都是0', i, 'init_points[i][0]', init_points[i][0], init_points[i][1], init_points[i][2],
                          init_points[i][3])

                    # 这个agent没有
                    world.agents[i].state.p_pos = init_points[i][:2]
                    world.agents[i].state.p_vel = init_points[i][2:4]
                    world.agents[i].state.heading_angle_last1 = init_points[i][6]
                    world.agents[i].state.p_dis = init_points[i][7]
                    world.agents[i].state.p_ini_to_end_dis = init_points[i][7]  # 对于这个agent来说，这个值就不再变了，无论更新迭代到哪一步，这个值都不会被更新
                    world.agents[i].state.p_last_vx = init_points[i][8]
                    world.agents[i].state.p_last_vy = init_points[i][9]
                    world.agents[i].collide = False # 没有碰撞
                    world.agents[i].end_label = True # 到这个轨迹的终点了
                    world.agents[i].state.des_rew = 0
                    world.agents[i].state.lane_rew = 0
                    world.agents[i].state.reference_line = np.array(all_expert_trj[aa]['ob'][i][:, :2])
                    world.agents[i].state.step = 0  # 因为是初始化场景，所以step为0
                    world.agents[i].state.min_distance = 0  # 初始时刻是真实的位置，所以距离参考线的距离是0
                    world.agents[i].state.delta_angle_last1 = init_points[i][56]  # 上一时刻的转向角
                    # world.agents[i].state.ini_step = init_points[i][22]
                    # print('init_points[i][:2]!!!!!!!!!!!!!!!!!!!!!:',init_points[i][:2],type(init_points[i][:2]))

                    p_pos_x = (world.agents[i].state.p_pos[0] * 38) - 4  # 当前真实位置
                    p_pos_y = (world.agents[i].state.p_pos[1] * 23) + 14

                    dx = (init_points[i][4] * 59) - 37  # 当前和终点的真实距离
                    dy = (init_points[i][5] * 27) - 4

                    des_x = dx + p_pos_x  # 真实的终点坐标
                    des_y = dy + p_pos_y

                    # print('测试中的p_pos格式',init_points[i][:2],world.agents[i].state.p_pos[0],world.agents[i].state.p_pos[1],'测试中的p_pos_x:',p_pos_x,'测试中的p_pos_y:',p_pos_y,'测试中的dx:',dx,'测试中的dy:',dy,'测试中的des_x:',des_x,'测试中的des_y:',des_y)

                    world.agents[i].state.p_des[0] = (des_x + 4) / 38
                    world.agents[i].state.p_des[1] = (des_y - 14) / 23  # 轨迹终点的归一化坐标
                else:
                    print('有效', i, 'init_points[i][0]', init_points[i][0], init_points[i][1], init_points[i][2],
                          init_points[i][3])

                    world.agents[i].state.p_pos = init_points[i][:2]
                    world.agents[i].state.p_vel = init_points[i][2:4]
                    world.agents[i].state.heading_angle_last1 = init_points[i][6]
                    world.agents[i].state.p_dis = init_points[i][7]
                    world.agents[i].state.p_ini_to_end_dis = init_points[i][7] # 对于这个agent来说，这个值就不再变了，无论更新迭代到哪一步，这个值都不会被更新
                    world.agents[i].state.p_last_vx = init_points[i][8]
                    world.agents[i].state.p_last_vy = init_points[i][9]
                    world.agents[i].collide = False
                    world.agents[i].end_label = False
                    world.agents[i].state.des_rew = 0
                    world.agents[i].state.lane_rew = 0
                    world.agents[i].state.reference_line = np.array(all_expert_trj[aa]['ob'][i][:, :2])
                    world.agents[i].state.step = 0  # 因为是初始化场景，所以step为0
                    world.agents[i].state.min_distance = 0  # 初始时刻是真实的位置，所以距离参考线的距离是0
                    world.agents[i].state.delta_angle_last1 = init_points[i][56]  # 上一时刻的转向角
                    # world.agents[i].state.ini_step = init_points[i][22]
                    # print('init_points[i][:2]!!!!!!!!!!!!!!!!!!!!!:',init_points[i][:2],type(init_points[i][:2]))


                    p_pos_x = (world.agents[i].state.p_pos[0] * 38) - 4 # 当前真实位置
                    p_pos_y = (world.agents[i].state.p_pos[1] * 23) + 14

                    dx = (init_points[i][4] * 59) - 37  # 当前和终点的真实距离
                    dy = (init_points[i][5] * 27) - 4

                    des_x = dx + p_pos_x  # 真实的终点坐标
                    des_y = dy + p_pos_y

                    # print('测试中的p_pos格式',init_points[i][:2],world.agents[i].state.p_pos[0],world.agents[i].state.p_pos[1],'测试中的p_pos_x:',p_pos_x,'测试中的p_pos_y:',p_pos_y,'测试中的dx:',dx,'测试中的dy:',dy,'测试中的des_x:',des_x,'测试中的des_y:',des_y)

                    world.agents[i].state.p_des[0] = (des_x + 4) / 38
                    world.agents[i].state.p_des[1] = (des_y - 14) / 23  # 轨迹终点的归一化坐标
            else:
                world.agents[i].state.p_pos = np.zeros(2)
                world.agents[i].state.p_vel = np.zeros(2)

                world.agents[i].state.p_des = np.zeros(2)
                world.agents[i].state.p_dis = 0

                # world.agents[i].state.heading_rad = 0  # 当前时刻的heading_rad
                world.agents[i].state.p_ini_to_end_dis = 0
                world.agents[i].state.p_last_vx = 0
                world.agents[i].state.p_last_vy = 0
                world.agents[i].state.delta_angle_now = 0  # 当前时刻的steering_rad
                world.agents[i].state.delta_angle_last1 = 0  # 上一时刻的steering_rad
                world.agents[i].state.delta_angle_last2 = 0  # 上上时刻的steering_rad
                world.agents[i].state.heading_angle_last1 = 0  # 上一时刻的steering_rad
                world.agents[i].state.heading_angle_last2 = 0  # 上上时刻的steering_rad
                world.agents[i].state.acc_x = None  # 当前时刻的横向加速度，目前没有值，必须执行了环境才有值，但是在step时，不更新这个值，也就是对于step之后的值来说，这个acc_x是上一时刻的值
                world.agents[i].state.acc_y = None
                world.agents[i].state.delta_accx = 0  # 上一时刻的agent和交互对象的加速度差
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
                world.agents[i].state.min_distance = 0  # 初始时刻是真实的位置，所以距离参考线的距离是0
                world.agents[i].state.delta_angle_last1 = 0  # 上一时刻的转向角

                world.agents[i].size = 2
                world.agents[i].id = i

        ini_obs = init_points[:, :57]  # shape为【46】
        ini_obs_lstm = np.zeros((8, 21, 57))
        print('reste中的ini_obs_lstm:',np.shape(ini_obs_lstm))  # (8, 21, 46)

        # 将每个第一维的前20个第二维都设为0，第21个第二维的参数等于 init_points[i, :46]
        for i in range(8):
            ini_obs_lstm[i, :20, :] = 0
            ini_obs_lstm[i, 20, :] = init_points[i, :57]

        return ini_obs, ini_steps, ini_obs_lstm


    def reward(self, agent, obs_use_for_reward, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # 提取车辆状态
        # 初始化总reward

        def Cal_GT(Agent_x,Agent_y,Agent_vx,Agent_vy,Agent_angle_last, Jiaohu_x, Jiaohu_y,Jiaohu_vx,Jiaohu_vy,Jiaohu_angle_last):  # time_trj,neig_left均为1行的dataframe
            # 计算和这个车辆的GT

            agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
            neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

            veh_length = 4.6
            veh_width = 1.8

            # 两辆车的k，斜率
            a_agent = math.tan(np.radians(Agent_angle_last))  # 斜率a_agent
            a_neig = math.tan(np.radians(Jiaohu_angle_last))  # 斜率a_neig

            # 两辆车的b
            b_agent = (Agent_y) - (a_agent * (Agent_x))
            b_neig = (Jiaohu_y) - (a_neig * (Jiaohu_x))

            # 两车的交点
            # 计算两直线的交点
            GT_value = None
            if a_neig == a_agent:  # 无交点，GT无穷大
                GT_value = 999999999999999  # 安全
            else:
                jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)  # 真实的坐标
                jiaodiany = (a_neig * jiaodianx) + b_neig
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

                if dot_product_agent > 0 and dot_product_neig > 0:
                    # 有冲突
                    agent_dis = np.sqrt(((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2))
                    neig_dis = np.sqrt(((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2))

                    agent_first_dis = agent_dis + (0.5 * veh_width) + (0.5 * veh_length)
                    neig_last_dis = neig_dis - (0.5 * veh_width) - (0.5 * veh_length)
                    agent_last_dis = agent_dis - (0.5 * veh_width) - (0.5 * veh_length)
                    neig_first_dis = neig_dis + (0.5 * veh_width) + (0.5 * veh_length)

                    if (agent_first_dis / agent_v) < (neig_last_dis / neig_v):  # agent先通过
                        GT_value = abs(neig_last_dis / neig_v - agent_first_dis / agent_v)
                    elif (neig_first_dis / neig_v) < (agent_last_dis / agent_v):  # neig先通过
                        GT_value = abs(agent_last_dis / agent_v - neig_first_dis / neig_v)
                    else:
                        GT_value = min(abs(agent_last_dis / agent_v - neig_first_dis / neig_v),
                                       abs(neig_last_dis / neig_v - agent_first_dis / agent_v))
                else:
                    GT_value = 999999999999999

            return GT_value

        rew = 0

        agent_des_rew = agent.state.des_rew # 如果轨迹点超出范围或者下一时刻到达终点，会赋予奖励值，但是与此同时state.pos也为0，因而不会有下面的操作，所以这两个rew都没有起作用
        agent_lane_rew = agent.state.lane_rew # 车道中心线距离奖励
        agent_heading_rew = agent.state.heading_angle_rew # 车道中心线角度奖励
        agent_delta_angle_rew = agent.state.delta_angle_rew # 转向角平滑奖励delta_angle_rew
        agent_heading_std_rew = agent.state.heading_std_rew # 航向角平滑奖励
        agent_heading_chaochufanwei_rew = agent.state.heading_chaochufanwei_rew # 航向角范围内奖励 如果当前时刻这个agent的angle超出了真实轨迹的范围，就给予一定的惩罚
        agent_collide_rew = agent.state.collide_rew  # 车辆和交叉口边界相撞的奖励
        rew_GT = 0
        # print('agent_des_rew:', agent_des_rew, 'agent_lane_rew:', agent_lane_rew)
        # if agent_des_rew == 200:
        #     print('生成的这个轨迹在终点啦')
        # if agent_lane_rew == -200:
        #     print('生成的这个轨迹在车道外面啦')

        if agent.state.p_pos[0] != 0:
            # 提取代理的状态和终点坐标
            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2
            agent_xend = (agent.state.p_des[0] * 38) - 4
            agent_yend = (agent.state.p_des[1] * 23) + 14
            agent_angle_last = (agent.state.heading_angle_last1*191)-1  # 上一个点的前进方向
            dis_to_end = agent.state.p_dis*37 # 当前点和终点的距离
            dis_ini_to_end = agent.state.p_ini_to_end_dis*37 # 这个agent的初始位置和终点的距离，是个定值


            # agent_scenario = agent.state.scenario  # 这个agent的场景
            # print('看看场景是不是一个？',scenario, agent_scenario)
            agent_v = np.sqrt(((agent_vx) ** 2 + (agent_vy) ** 2))

            # 避免碰撞
            # 计算agent和周围最密切的三个交互对象的GT
            # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
            # 安全奖励
            rew_GT = 0
            GT = [] # 存储和三辆车的GT
            # 计算和前方车辆的GT
            # print('obs_use_for_reward',np.shape(obs_use_for_reward),obs_use_for_reward)
            # 计算前方车辆的GT
            same_jiaohu_agent_x = obs_use_for_reward[0][0]
            same_jiaohu_agent_y = obs_use_for_reward[0][1]
            same_jiaohu_agentk_vx = obs_use_for_reward[0][2]
            same_jiaohu_agent_vy = obs_use_for_reward[0][3]
            same_jiaohu_agent_angle_last = obs_use_for_reward[0][4]

            if same_jiaohu_agent_x != -4:
                same_jiaohu_agent_GT_value = Cal_GT(agent_x, agent_y, agent_vx, agent_vy, agent_angle_last,
                                                       same_jiaohu_agent_x, same_jiaohu_agent_y,
                                                       same_jiaohu_agentk_vx, same_jiaohu_agent_vy,
                                                       same_jiaohu_agent_angle_last)
            else:
                same_jiaohu_agent_GT_value = None

            # 在交互的agent当中先找最近的左右车
            # 左侧视野的三个agent
            left_jiaohu_agent1_x = obs_use_for_reward[0][5]
            left_jiaohu_agent1_y = obs_use_for_reward[0][6]
            left_jiaohu_agent1_vx = obs_use_for_reward[0][7]
            left_jiaohu_agent1_vy = obs_use_for_reward[0][8]
            left_jiaohu_agent1_angle_last = obs_use_for_reward[0][9]

            left_jiaohu_agent2_x = obs_use_for_reward[0][10]
            left_jiaohu_agent2_y = obs_use_for_reward[0][11]
            left_jiaohu_agent2_vx = obs_use_for_reward[0][12]
            left_jiaohu_agent2_vy = obs_use_for_reward[0][13]
            left_jiaohu_agent2_angle_last = obs_use_for_reward[0][14]

            left_jiaohu_agent3_x = obs_use_for_reward[0][15]
            left_jiaohu_agent3_y = obs_use_for_reward[0][16]
            left_jiaohu_agent3_vx = obs_use_for_reward[0][17]
            left_jiaohu_agent3_vy = obs_use_for_reward[0][18]
            left_jiaohu_agent3_angle_last = obs_use_for_reward[0][19]

            left_jiaohu_agent_x = None
            left_jiaohu_agent_y = None
            left_jiaohu_agent_vx = None
            left_jiaohu_agent_vy = None
            left_jiaohu_agent_angle_last = None

            if left_jiaohu_agent1_x != -4:
                # 这个交互对象agent是有效的
                dis_agent1 = np.sqrt(((left_jiaohu_agent1_x-agent_x)**2+(left_jiaohu_agent1_y-agent_y)**2))
            else:
                dis_agent1 = None
            if left_jiaohu_agent2_x != -4:
                # 这个交互对象agent是有效的
                dis_agent2 = np.sqrt(((left_jiaohu_agent2_x-agent_x)**2+(left_jiaohu_agent2_y-agent_y)**2))
            else:
                dis_agent2 = None
            if left_jiaohu_agent3_x != -4:
                # 这个交互对象agent是有效的
                dis_agent3 = np.sqrt(((left_jiaohu_agent3_x-agent_x)**2+(left_jiaohu_agent3_y-agent_y)**2))
            else:
                dis_agent3 = None
            # 找到距离最近的left_agent
            non_none_count_leftagent = sum(1 for distance in (dis_agent1, dis_agent2, dis_agent3) if distance is not None)
            if non_none_count_leftagent != 0:
                min_distance_leftagent = min(distance for distance in (dis_agent1, dis_agent2, dis_agent3) if distance is not None)

                # 找到最小距离对应的编号
                min_distance_index_leftagent = (dis_agent1, dis_agent2, dis_agent3).index(min_distance_leftagent) + 1  # 加1是因为编号从1开始
                if min_distance_index_leftagent == 1:
                    left_jiaohu_agent_x = left_jiaohu_agent1_x
                    left_jiaohu_agent_y = left_jiaohu_agent1_y
                    left_jiaohu_agent_vx = left_jiaohu_agent1_vx
                    left_jiaohu_agent_vy = left_jiaohu_agent1_vy
                    left_jiaohu_agent_angle_last = left_jiaohu_agent1_angle_last
                elif min_distance_index_leftagent == 2:
                    left_jiaohu_agent_x = left_jiaohu_agent2_x
                    left_jiaohu_agent_y = left_jiaohu_agent2_y
                    left_jiaohu_agent_vx = left_jiaohu_agent2_vx
                    left_jiaohu_agent_vy = left_jiaohu_agent2_vy
                    left_jiaohu_agent_angle_last = left_jiaohu_agent2_angle_last
                elif min_distance_index_leftagent == 3:
                    left_jiaohu_agent_x = left_jiaohu_agent3_x
                    left_jiaohu_agent_y = left_jiaohu_agent3_y
                    left_jiaohu_agent_vx = left_jiaohu_agent3_vx
                    left_jiaohu_agent_vy = left_jiaohu_agent3_vy
                    left_jiaohu_agent_angle_last = left_jiaohu_agent3_angle_last

                left_jiaohu_agent_GT_value = Cal_GT(agent_x, agent_y, agent_vx, agent_vy, agent_angle_last, left_jiaohu_agent_x,
                                              left_jiaohu_agent_y, left_jiaohu_agent_vx, left_jiaohu_agent_vy, left_jiaohu_agent_angle_last)
            else:
                left_jiaohu_agent_GT_value = None

            # 右侧视野的三个agent
            right_jiaohu_agent1_x = obs_use_for_reward[0][20]
            right_jiaohu_agent1_y = obs_use_for_reward[0][21]
            right_jiaohu_agent1_vx = obs_use_for_reward[0][22]
            right_jiaohu_agent1_vy = obs_use_for_reward[0][23]
            right_jiaohu_agent1_angle_last = obs_use_for_reward[0][24]

            right_jiaohu_agent2_x = obs_use_for_reward[0][25]
            right_jiaohu_agent2_y = obs_use_for_reward[0][26]
            right_jiaohu_agent2_vx = obs_use_for_reward[0][27]
            right_jiaohu_agent2_vy = obs_use_for_reward[0][28]
            right_jiaohu_agent2_angle_last = obs_use_for_reward[0][29]

            right_jiaohu_agent3_x = obs_use_for_reward[0][30]
            right_jiaohu_agent3_y = obs_use_for_reward[0][31]
            right_jiaohu_agent3_vx = obs_use_for_reward[0][32]
            right_jiaohu_agent3_vy = obs_use_for_reward[0][33]
            right_jiaohu_agent3_angle_last = obs_use_for_reward[0][34]

            right_jiaohu_agent_x = None
            right_jiaohu_agent_y = None
            right_jiaohu_agent_vx = None
            right_jiaohu_agent_vy = None
            right_jiaohu_agent_angle_last = None

            if right_jiaohu_agent1_x != -4:
                # 这个交互对象agent是有效的
                dis_agent1 = np.sqrt((
                    (right_jiaohu_agent1_x - agent_x) ** 2 + (right_jiaohu_agent1_y - agent_y) ** 2))
            else:
                dis_agent1 = None
            if right_jiaohu_agent2_x != -4:
                # 这个交互对象agent是有效的
                dis_agent2 = np.sqrt((
                    (right_jiaohu_agent2_x - agent_x) ** 2 + (right_jiaohu_agent2_y - agent_y) ** 2))
            else:
                dis_agent2 = None
            if right_jiaohu_agent3_x != -4:
                # 这个交互对象agent是有效的
                dis_agent3 = np.sqrt((
                    (right_jiaohu_agent3_x - agent_x) ** 2 + (right_jiaohu_agent3_y - agent_y) ** 2))
            else:
                dis_agent3 = None
            # 找到距离最近的right_agent
            non_none_count_rightagent = sum(
                1 for distance in (dis_agent1, dis_agent2, dis_agent3) if distance is not None)
            if non_none_count_rightagent != 0:
                min_distance_rightagent = min(
                    distance for distance in (dis_agent1, dis_agent2, dis_agent3) if distance is not None)

                # 找到最小距离对应的编号
                min_distance_index_rightagent = (dis_agent1, dis_agent2, dis_agent3).index(
                    min_distance_rightagent) + 1  # 加1是因为编号从1开始
                if min_distance_index_rightagent == 1:
                    right_jiaohu_agent_x = right_jiaohu_agent1_x
                    right_jiaohu_agent_y = right_jiaohu_agent1_y
                    right_jiaohu_agent_vx = right_jiaohu_agent1_vx
                    right_jiaohu_agent_vy = right_jiaohu_agent1_vy
                    right_jiaohu_agent_angle_last = right_jiaohu_agent1_angle_last
                elif min_distance_index_rightagent == 2:
                    right_jiaohu_agent_x = right_jiaohu_agent2_x
                    right_jiaohu_agent_y = right_jiaohu_agent2_y
                    right_jiaohu_agent_vx = right_jiaohu_agent2_vx
                    right_jiaohu_agent_vy = right_jiaohu_agent2_vy
                    right_jiaohu_agent_angle_last = right_jiaohu_agent2_angle_last
                elif min_distance_index_rightagent == 3:
                    right_jiaohu_agent_x = right_jiaohu_agent3_x
                    right_jiaohu_agent_y = right_jiaohu_agent3_y
                    right_jiaohu_agent_vx = right_jiaohu_agent3_vx
                    right_jiaohu_agent_vy = right_jiaohu_agent3_vy
                    right_jiaohu_agent_angle_last = right_jiaohu_agent3_angle_last

                right_jiaohu_agent_GT_value = Cal_GT(agent_x, agent_y, agent_vx, agent_vy, agent_angle_last,
                                               right_jiaohu_agent_x,
                                               right_jiaohu_agent_y, right_jiaohu_agent_vx, right_jiaohu_agent_vy,
                                               right_jiaohu_agent_angle_last)
            else:
                right_jiaohu_agent_GT_value = None


            # 左侧视野的landmark
            left_jiaohu_landmark_x = obs_use_for_reward[0][35]
            left_jiaohu_landmark_y = obs_use_for_reward[0][36]
            left_jiaohu_landmark_vx = obs_use_for_reward[0][37]
            left_jiaohu_landmark_vy = obs_use_for_reward[0][38]
            left_jiaohu_landmark_angle_last = obs_use_for_reward[0][39]

            # 右侧视野的landmark
            right_jiaohu_landmark_x = obs_use_for_reward[0][40]
            right_jiaohu_landmark_y = obs_use_for_reward[0][41]
            right_jiaohu_landmark_vx = obs_use_for_reward[0][42]
            right_jiaohu_landmark_vy = obs_use_for_reward[0][43]
            right_jiaohu_landmark_angle_last = obs_use_for_reward[0][44]

            if left_jiaohu_landmark_x != -5:
                left_jiaohu_landmark_GT_value = Cal_GT(agent_x,agent_y,agent_vx,agent_vy,agent_angle_last,
                                                       left_jiaohu_landmark_x, left_jiaohu_landmark_y,
                                                       left_jiaohu_landmark_vx,left_jiaohu_landmark_vy,
                                                       left_jiaohu_landmark_angle_last)
            else:
                left_jiaohu_landmark_GT_value = None

            if right_jiaohu_landmark_x != -5:
                right_jiaohu_landmark_GT_value = Cal_GT(agent_x,agent_y,agent_vx,agent_vy,agent_angle_last,
                                                        right_jiaohu_landmark_x, right_jiaohu_landmark_y,
                                                        right_jiaohu_landmark_vx,right_jiaohu_landmark_vy,
                                                        right_jiaohu_landmark_angle_last)
            else:
                right_jiaohu_landmark_GT_value = None

            if same_jiaohu_agent_GT_value is not None and same_jiaohu_agent_GT_value <= 1.5:
                rew_GT = rew_GT - 1 * (1.5 - same_jiaohu_agent_GT_value)

            if left_jiaohu_agent_GT_value is not None and left_jiaohu_agent_GT_value <= 1.5:
                rew_GT = rew_GT - 1 * (1.5 - left_jiaohu_agent_GT_value)
                # print('和左侧交互agent对象严重冲突')
            if right_jiaohu_agent_GT_value is not None and right_jiaohu_agent_GT_value <= 1.5:
                rew_GT = rew_GT - 1 * (1.5 - right_jiaohu_agent_GT_value)
                # print('和右侧交互agent对象严重冲突')

            if left_jiaohu_landmark_GT_value is not None and left_jiaohu_landmark_GT_value <= 1.5:
                rew_GT = rew_GT - 1 * (1.5 - left_jiaohu_landmark_GT_value)
                # print('和左侧交互agent对象严重冲突')
            if right_jiaohu_landmark_GT_value is not None and right_jiaohu_landmark_GT_value <= 1.5:
                rew_GT = rew_GT - 1 * (1.5 - right_jiaohu_landmark_GT_value)

            # rew_GT = 1 - np.exp(-rew_GT)  # GT越大，应该越安全，奖励越多


            # 计算效率的奖励
            # rew_effi = 1 - np.exp(-agent_v)  # 速度越高，reward越高

            # 继续补充

            # 终点奖励 正向， GT越接近于0越危险
            # print('rew_des:',rew_des,'rew_lane:',rew_lane,'rew_GT:',rew_GT,'rew_effi:',rew_effi)
            # rew = 0.3*rew_des + 0.7*rew_lane + 0.0*rew_GT + 0.0*rew_effi
            # print('rew_pos:',rew_pos,'rew_des:', rew_des, 'rew_GT:', rew_GT, 'rew_effi:', rew_effi)
            # rew = 0.9 * rew_des + 0.01 * rew_GT + 0.09 * rew_effi

            # rew = 50 * rew_des + rew_GT + 50 * rew_effi

        # elif agent.state.p_pos[0] == 0 and agent.collide == False: # 到了轨迹终点附近，但是没有和交叉口边界发生碰撞 或者是这个agent无效
        #     rew_collision = 0  # 无碰撞
        # elif agent.state.p_pos[0] == 0 and agent.collide == True: # 有和交叉口边界发生碰撞
        #     rew_collision = agent.state.collide_rew  # 碰撞
        # rew = rew + agent_des_rew + 3*agent_lane_rew + agent_heading_rew + agent_delta_angle_rew + agent_heading_std_rew + agent_collide_rew + rew_GT
        rew = 0

        # rew = rew_des
        return rew
        # return rew[0][0]

    def observation(self, agent, action_n_agent, reset_infos, world):

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

        def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                      Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last):  # time_trj,neig_left均为1行的dataframe

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

        trj_go_step = action_n_agent[3]  # 轨迹走的步数
        if agent.id <= 2:
            direction_agent = 'left'
        else:
            direction_agent = 'straight'

        if agent.state.p_pos[0] != 0:
        # if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            # 加上agent的step!,然后记录每个step的obs，然后再根据当前的step汇总
            t = world.time
            # world.landmarks = np.array(all_vehss[aa]['landmarks'])  # [30,185,7] 归一化之后的数据
            # print('reset_infos:',reset_infos)
            if reset_infos == [True]:  # 这个环境刚重置过，需要获得初始时刻的obs
                # print('action_n_agent:',action_n_agent)
                # action_n_agent：agent.state.p_pos, np.array([0]),
                # np.array([0]),
                # np.array([ini_steps[agent.id][0]]), ini_obs[agent.id]
                if action_n_agent[2] == action_n_agent[4]:  #  and action_n_agent[2] <= 183:
                    # 环境当前的步数==轨迹开始时的步数，且环境当前的步长<=183，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                    # print('world.landmarks:',np.shape(world.landmarks)) # (30, 185, 7)
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]    # (30, 7)
                    # print('True_if_shape_landmarks_veh:',np.shape(landmarks_veh))  # (30, 7)
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]  # (30, 5) 可能小于30
                    # print('True_if_shape_landmarks_veh_use:',np.shape(landmarks_veh_use))  # (30, 5)
                # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
                else:  # 这个agent不开始前进，所以没有landmarks，30个全部为0  没有这种情况其实
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]
                    print('observation重置时这种情况不会有',action_n_agent[2], action_n_agent[4])
                    # print('True_else_shape_landmarks_veh:', np.shape(landmarks_veh))  # (30, 7)
                    # print('True_else_shape_landmarks_veh_use:', np.shape(landmarks_veh_use))  # (30, 5)
            else:  # 没有重置，但这个agent也是有效的，说明开始前进了
                # (self.actions[k][i], np.array([env_go_step]),
                # np.array([self.trj_GO_STEP[i][k]]),
                # np.array([self.ini_steps[k][i][0]]),
                # self.ini_obs[k][i])
                if action_n_agent[2] >= action_n_agent[4]: #  and action_n_agent[2] <= 183:
                    # 环境当前的步数>=轨迹开始时的步数，且环境当前的步长<=183，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2])+1,:]    # [14,7]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]
                # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
                else:  # 这种情况也不会有
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]
                    print('这种情况也不会有',action_n_agent[2],action_n_agent[4])

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0]!=0)]  # x,y,vx,vy,deg
            # print('vehs_agent:',vehs_agent)
            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            # print('0000000:',np.shape(vehs_agent), vehs_agent)  # (7, 6)
            # vehs = vehs_agent + landmarks_veh_use
            # print('vehs:', np.shape(vehs), vehs)

            veh_same = []  # 车辆同方向最近的一辆车
            veh_left_agents = []  # 车辆左侧的三辆车 agent
            veh_right_agents = []  # 车辆右侧的三辆车 agent
            veh_left_landmark = []  # 车辆左侧的最近一辆车 landmark
            veh_right_landmark = []  # 车辆右侧侧的最近一辆车 landmark

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4))**2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14))**2) for ve in
                                        vehs_agent])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5))**2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3))**2)
                                           for ve_landmark in
                                           landmarks_veh_use])  # 两辆车在这一帧的距离

            # distances_agent = np.array([np.linalg.norm([agent.state.p_pos[0]*38 - 4 - ve[0]*38 - 4,
            #                                       agent.state.p_pos[1] * 23 + 14 - ve[1]* 23 + 14]) for ve in vehs_agent]) # 两辆车在这一帧的距离
            # # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # distances_landmark = np.array([np.linalg.norm([agent.state.p_pos[0] * 38 - 4 - ve_landmark[0] * 39 - 5,
            #                                             agent.state.p_pos[1] * 23 + 14 - ve_landmark[1] * 38 - 3]) for ve_landmark in
            #                             landmarks_veh_use])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # 重新写，agent和landmark，分开找

            # 先判断是否会和周围的车辆发生碰撞

            agent_x_real = (agent.state.p_pos[0] * 38) - 4
            agent_y_real = (agent.state.p_pos[1] * 23) + 14
            agent_vx_real = (agent.state.p_vel[0] * 21) - 14
            agent_vy_real = (agent.state.p_vel[1] * 12) - 2
            agent_v_real = np.sqrt(((agent_vx_real) ** 2) + ((agent_vy_real) ** 2))
            # 当前时刻的角度，也就是当前时刻的上一时刻的角度
            agent_degree_last_real = (agent.state.heading_angle_last1 * 191) - 1
            length_agent = 5
            width_agent = 2

            collide_label = False  # 判断是否发生碰撞的标签，True代表发生了碰撞，False代表没发生碰撞 (只要和一辆车发生碰撞，就算有碰撞)
            # 直接计算GT 内含判断是否和其他agent以及landmark发生碰撞
            use_GT = []  # 存放和所有agent以及所有landmark的GT

            # 计算和agent的GT
            if len(vehs_agent) != 0:
                for i in range(len(vehs_agent)):
                    other_obj_x_real = (vehs_agent[i][0] * 38) - 4
                    other_obj_y_real = (vehs_agent[i][1] * 23) + 14
                    other_obj_vx_real = (vehs_agent[i][2] * 21) - 14
                    other_obj_vy_real = (vehs_agent[i][3] * 12) - 2
                    other_obj_degree_last_real = (vehs_agent[i][4] * 191) - 1
                    # 当前时刻的角度，也就是当前时刻的上一时刻的角度

                    jiaohu_agent_GT_value = Cal_GT_gt(agent_x_real, agent_y_real, agent_vx_real, agent_vy_real,
                                                      agent_degree_last_real, direction_agent,
                                                      other_obj_x_real, other_obj_y_real,
                                                      other_obj_vx_real, other_obj_vy_real,
                                                      other_obj_degree_last_real)
                    use_GT.append(jiaohu_agent_GT_value)
                    if jiaohu_agent_GT_value == 0:
                        collide_label = True  # 发现碰撞
                        # break  # 退出当前这个循环，已经发现碰撞了，就不再继续了

            # 计算和landmark的GT
            if len(landmarks_veh_use) != 0:
                for i in range(len(landmarks_veh_use)):
                    other_obj_x_real = (landmarks_veh_use[i][0] * 39) - 5
                    other_obj_y_real = (landmarks_veh_use[i][1] * 38) - 3
                    other_obj_vx_real = (landmarks_veh_use[i][2] * 31) - 16
                    other_obj_vy_real = (landmarks_veh_use[i][3] * 21) - 10
                    other_obj_degree_last_real = (landmarks_veh_use[i][4] * 360) - 90
                    # 当前时刻的角度，也就是当前时刻的上一时刻的角度

                    left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x_real, agent_y_real, agent_vx_real, agent_vy_real,
                                                              agent_degree_last_real, direction_agent,
                                                              other_obj_x_real,
                                                              other_obj_y_real,
                                                              other_obj_vx_real,
                                                              other_obj_vy_real,
                                                              other_obj_degree_last_real)

                    use_GT.append(left_jiaohu_landmark_GT_value)
                    if left_jiaohu_landmark_GT_value == 0:
                        collide_label = True  # 发现碰撞
                        agent.collide = True
                        # break  # 退出当前这个循环，已经发现碰撞了，就不再继续了

            # 以下是直接计算 用于计算后面airl_con_ac.py文件中reward用到的rew_social_allbatch
            # generate rew_social_allbatch
            rew_social_generate = []
            penalty = 1  # 惩罚系数
            delta_angle_last1 = agent.state.delta_angle_last1
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
                        # comfort_adj = -(np.exp(1)-1) * penalty
                        comfort_adj = -1 * penalty
                    else:
                        comfort_adj_normalized = dis_left_delta_angle_last1 / left_delta_angle_last1_realstd
                        # comfort_adj = -(np.exp(comfort_adj_normalized)-1) * penalty
                        comfort_adj = -(dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
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
                        # comfort_adj = -(np.exp(1) - 1) * penalty
                        comfort_adj = -1 * penalty
                    else:
                        comfort_adj_normalized = dis_right_delta_angle_last1 / right_delta_angle_last1_realstd
                        # comfort_adj = -(np.exp(comfort_adj_normalized) - 1) * penalty
                        comfort_adj = -(dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                        # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1

            # 利己-效率
            if direction_agent == 'left':
                guiyihua_v = agent_v_real / 10
            else:
                guiyihua_v = agent_v_real / 14

            # 利己-终点（只有到终点才有）
            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - agent_x_real
            d_y_new = des_y_always_ - agent_y_real

            des = np.array([(d_x_new + 37) / 59, (d_y_new + 4) / 27])  # 和终点的x和y方向的归一化距离

            agent_dis_todes = np.sqrt((d_x_new) ** 2 + (d_y_new) ** 2)

            if agent_dis_todes < 0.5:
                rew_des_normalized = 1
            else:
                rew_des_normalized = 0

            rew_avespeed_normalized = guiyihua_v + rew_des_normalized

            # rew_avespeed_normalized = agent_v / 6.8  ## 除以85分位速度 归一化速度 【0，1+】，但也不会太大
            # rew_avespeed = np.exp(rew_avespeed_normalized)-1  # 目的是速度越接近于期望速度甚至更高，这个奖励越大
            rew_avespeed = rew_avespeed_normalized

            # 利己-车道偏移
            # rew_lane_pianyi = np.exp(pianyi_distance)-1  # 目的是车道偏移越接近于半个车道宽度甚至更大，这个惩罚越大
            pianyi_distance = agent.state.min_distance
            rew_lane_pianyi = pianyi_distance

            # 利他-GT
            # print("use_GT：",use_GT)  # 9个元素的list，例如[None, None, None, None, None, None, None, 0.32294667405015254, None]
            use_GT_list_0 = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
            use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
            rew_minGT_mapped = 0
            # print('use_GT_list:',use_GT_list)
            if len(use_GT_list) != 0:
                # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                rew_minGT = min(use_GT_list)  # min(use_GT_list)
                # if rew_minGT < 4:
                #     rew_minGT_normalized = agent_v / 4
                #     rew_minGT_mapped = np.exp(rew_minGT_normalized) - 1
                # elif rew_minGT >= 4:
                #     rew_minGT_mapped = np.exp(1) - 1

                # rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                if rew_minGT <= 1.5:
                    # 归一化
                    normalized_data = (rew_minGT - 0) / (1.5 - 0)
                    # 映射到目标范围
                    rew_minGT_mapped = normalized_data * (0.25 - 0.0) - 0.0
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
                # rew_minGT_mapped = -1  # 只是一个标记，代表没有交互对象
                rew_minGT_mapped = 0  # 只是一个标记，代表没有交互对象

                social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0


            rew_social_generate = [10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped]


            for ii in range(len(vehs_agent)):  # vehs_agent (7, 5)
                if vehs_agent[ii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
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
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
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

                    # print('agent.id:',agent.id,'agent_x:',agent.state.p_pos[0],'agent_y:',agent.state.p_pos[1],
                    #       'agent_x_real:',agent.state.p_pos[0] * 38 - 4,'agent_y_real:',agent.state.p_pos[1] * 23 + 14,
                    #       'heading_angle_last_1:',agent.state.heading_angle_last1,'heading_angle_last_1_real:',angle,
                    #       'jiaohu_id:',vehs_agent[ii][5],'jiaohu_x:',vehs_agent[ii][0],'jiaohu_y:',vehs_agent[ii][1],
                    #       'jiaohu_x_real:',vehs_agent[ii][0] * 38 - 4,'jiaohu_y_real:',vehs_agent[ii][1] * 23 + 14,
                    #       'b[0]:',b[0],'b[1]:',b[1],'cross:',cross,'angle_jiaodu:',angle_jiaodu)
                    # print()

                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到同方向前方的车辆
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                        # print('agent.id:',agent.id,'有同方向前方的车辆：',vehs_agent[ii][5],veh_same)
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向左前方的车辆
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                        # print('agent.id:', agent.id, '有异向左前方的车辆：', vehs_agent[ii][5], veh_left_agents)
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向右前方的车辆
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])
                        # print('agent.id:', agent.id, '有异向右前方的车辆：', vehs_agent[ii][5], veh_right_agents)

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []
            # print('veh_same：',veh_same)
            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                    # print('agent.id:',agent.id,'有同方向前方的车辆：',veh_same, veh_neig)
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))


            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    # print('修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(), distances_old[0])
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    # print('sorted_veh:',sorted_veh)
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                                # print('agent.id:', agent.id, '有异方向的车辆：', veh_neig)
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    # 视野里不止三辆车，那就把最近的三辆车存起来
                    # 找到最近的三辆车
                    # 按照距离升序排序，然后选择前三辆车
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    # 取前三辆车
                    top_3_veh = sorted_veh[:3, :]
                    # 将结果存储为 veh_new
                    veh_new = top_3_veh.copy()
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                                # print('agent.id:', agent.id, '有异方向的车辆：', veh_neig,'视野内超过3辆')
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            # print('veh_neig:',np.shape(veh_neig),veh_neig)
            for iii in range(len(landmarks_veh_use)):  # vehs_landmark (n, 5)
                if landmarks_veh_use[iii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1


                    # if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                    #     angle = agent.state.heading_angle_last1 * 191 - 1 + 360
                    # else:
                    #     angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
                    cross = np.cross((b[0], b[1]), ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                                    (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到左前方的landmark
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到右前方的landmark
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []
            # 找到距离agent最近的左右侧视野的landmark，先找左侧
            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            # 再找右侧
            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            # obs = np.array([ (agent_x-(veh_neig[0][0] * 38 - 4)+12)/27, (agent_y-(veh_neig[0][1] * 23 + 14)+14)/18,
            #                  (agent_vx-(veh_neig[0][2] * 21 - 14)+7)/14, (agent_vy-(veh_neig[0][3] * 12 - 2)+5)/6,
            #
            #                  (agent_x-(veh_neig[1][0] * 38 - 4)+15)/30, (agent_y-(veh_neig[1][1] * 23 + 14)+7)/14,
            #                  (agent_vx-(veh_neig[1][2] * 21 - 14)+18)/36, (agent_vy-(veh_neig[1][3] * 12 - 2)+5)/7,
            #
            #                (agent_x - (veh_neig[2][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[2][1] * 23 + 14) + 7) / 14,
            #                (agent_vx - (veh_neig[2][2] * 21 - 14) + 14) / 29,(agent_vy - (veh_neig[2][3] * 12 - 2) + 1) / 2,
            #
            #                (agent_x - (veh_neig[3][0] * 38 - 4) + 14) / 14,(agent_y - (veh_neig[3][1] * 23 + 14) + 7) / 2,
            #                (agent_vx - (veh_neig[3][2] * 21 - 14) - 4) / 9, (agent_vy - (veh_neig[3][3] * 12 - 2) - 0) / 1,
            #
            #                (agent_x - (veh_neig[4][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[4][1] * 23 + 14) + 15) / 24,
            #                (agent_vx - (veh_neig[4][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[4][3] * 12 - 2) + 9) / 15,
            #
            #                (agent_x - (veh_neig[5][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[5][1] * 23 + 14) + 9) / 17,
            #                (agent_vx - (veh_neig[5][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[5][3] * 12 - 2) + 6) / 12,
            #
            #                (agent_x - (veh_neig[6][0] * 38 - 4) + 15) / 10,(agent_y - (veh_neig[6][1] * 23 + 14) + 1) / 7,
            #                (agent_vx - (veh_neig[6][2] * 21 - 14) - 3) / 5,(agent_vy - (veh_neig[6][3] * 12 - 2) - 1) / 3,
            #
            #                (agent_x - (veh_neig_landmark[0][0] * 39 - 5) + 14) / 29,(agent_y - (veh_neig_landmark[0][1] * 38 - 3) + 15) / 30,
            #                (agent_vx - (veh_neig_landmark[0][2] * 31 - 16) + 21) / 35,(agent_vy - (veh_neig_landmark[0][3] * 21 - 10) + 5) / 16,
            #
            #                (agent_x - (veh_neig_landmark[1][0] * 39 - 5) + 15) / 30,(agent_y - (veh_neig_landmark[1][1] * 38 - 3) + 15) / 29,
            #                (agent_vx - (veh_neig_landmark[1][2] * 31 - 16) + 14) / 25,(agent_vy - (veh_neig_landmark[1][3] * 21 - 10) + 7) / 17]).reshape([1, -1])

            # obs_old = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # 从这里接着改！
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
                           # 加的这两个是额外的，在后面训练判别器的train的时候需要去掉


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

            # obs_usefor_reward = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # print('可用于转换的obs是什么形态呢???????????????????????????????:', np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]], veh_neig[0][[0,1,2,3]] )
            # print('转换前的obs是什么形态呢???????????????????????????????:',np.array([abs(np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]] - veh_neig[0][[0,1,2,3]]) for ii in range(3)]))
            # print('转换后的obs是什么形态呢???????????????????????????????:', np.shape(obs),obs)
            # print('之前的obs是什么形态呢???????????????????????????????:', np.shape(obs_old),obs_old)

            des = np.zeros(2)

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new+37) / 59, (d_y_new+4) / 27])  # 和终点的x和y方向的归一化距离

            # print('agent.state.p_last_vx:',agent.state.p_last_vx,'agent.state.p_last_vy',agent.state.p_last_vy)
            # print('obs[0]:', np.shape(obs[0]))
            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]),np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))  # [[0,1,2,3,4,5,6]]  这里的a是下一个时刻的obs
            ini_steps = np.array([agent.state.ini_step])
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
                current_agent_step = int(trj_go_step)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
                current_agent_step = int(trj_go_step) + 1

            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = current_agent_step - i
                # 如果当前步数合法，使用 getattr 获取属性值
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)  # 追加到列表前端
                    # obs_data.append(step_data)  # 追加到列表末尾
                    # print('step_data:', current_step, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))  # 追加到列表前端
                    # obs_data.append(np.zeros([46]))
                    # print('np.zeros([46]:',np.zeros([46]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同a：', np.shape(a))  # (46,)
            # 测试原来的a和obs_data_row的内容形式相不相同a： (46,)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同obs_data_row：', np.shape(obs_data_lstm))  # (21, 46)
            # 测试原来的a和obs_data_row的内容形式相不相同obs_data_row： (21, 46)
            # print('测试原来的a和obs_data_row的格式相不相同：', np.shape(a), np.shape(obs_data_row))
                # else:  # 当前时刻的前i步，已经没有了，那就赋值为0，adjusted_ob_data已经设置好了

            # print('有效的a:',np.shape(a),a)
            # print('有效的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)
        else:
            a = np.zeros([57])  # 原来是24
            rew_social_generate = [0.1, 0.1, 0.1, 0.1]
            if agent.collide == True:
                collide_label = True  # 这个车之前已经撞了
            else:
                collide_label = False  # 没有这个车
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)

            obs_usefor_reward = np.zeros([45]).reshape([1, -1])
            # ini_steps = np.zeros([1])
            # print('0的a:', np.shape(a), a)
            # print('0的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)

        # print('get_obs输出的数据的shape：','obs_data_lstm:',np.shape(obs_data_lstm),
        #       'obs_usefor_reward:',np.shape(obs_usefor_reward),'a:',np.shape(a))
        # obs_data_lstm: (21, 46) obs_usefor_reward: (1, 46) a: (46,)
        return obs_data_lstm, obs_usefor_reward, a, collide_label, rew_social_generate
        # return np.concatenate((agent.state.p_pos,obs))

    def observation_nowstep(self, agent, action_n_agent, world):  # HV刚进入交叉口的时候，其他车已经在交叉口并且不按专家轨迹走，需要更新HV的iniobs
        # print('observation_nowstep的action_n_agent：',np.shape(action_n_agent))  # (5,)
        trj_go_step = action_n_agent[3]  # 轨迹走的步数
        if agent.state.p_pos[0] != 0:
        # if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            # 加上agent的step!,然后记录每个step的obs，然后再根据当前的step汇总
            if action_n_agent[2] >= action_n_agent[4]:  #  and action_n_agent[2] <= 183:
                # 环境当前的步数>=轨迹开始时的步数，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                # print('world.landmarks:',np.shape(world.landmarks)) # (30, 185, 7)
                landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]    # (30, 7) 当前步的landmark
                # print('True_if_shape_landmarks_veh:',np.shape(landmarks_veh))  # (30, 7)
                landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]  # (30, 5) 可能小于30
                # print('True_if_shape_landmarks_veh_use:',np.shape(landmarks_veh_use))  # (30, 5)
            # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
            else:  # 这个agent不开始前进，所以没有landmarks，30个全部为0  没有这种情况其实
                landmarks_veh = [np.zeros([7]) for _ in range(30)]
                landmarks_veh_use = [other[:5] for other in landmarks_veh]
                print('observation_nowstep这种情况不会有', action_n_agent[2], action_n_agent[4])
                # print('True_else_shape_landmarks_veh:', np.shape(landmarks_veh))  # (30, 7)
                # print('True_else_shape_landmarks_veh_use:', np.shape(landmarks_veh_use))  # (30, 5)

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0]!=0)]  # x,y,vx,vy,deg
            # 把AV的信息加入vehs_agent 对于不同的测试场景，这个要个性化处理，比如对于98场景，AV作为agent0，所以按照如下方式append，agent.id为0

            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            # print('作为agent的vehs:',np.shape(vehs_agent), vehs_agent)  # (7, 6)
            # vehs = vehs_agent + landmarks_veh_use
            # print('vehs:', np.shape(vehs), vehs)

            veh_same = []  # 车辆同方向最近的一辆车
            veh_left_agents = []  # 车辆左侧的三辆车 agent
            veh_right_agents = []  # 车辆右侧的三辆车 agent
            veh_left_landmark = []  # 车辆左侧的最近一辆车 landmark
            veh_right_landmark = []  # 车辆右侧侧的最近一辆车 landmark

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4)) ** 2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14)) ** 2) for ve in
                                        vehs_agent])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5)) ** 2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3)) ** 2)
                                           for ve_landmark in
                                           landmarks_veh_use])  # 两辆车在这一帧的距离


            # distances_agent = np.array([np.linalg.norm([agent.state.p_pos[0]*38 - 4 - ve[0]*38 - 4,
            #                                       agent.state.p_pos[1] * 23 + 14 - ve[1]* 23 + 14]) for ve in vehs_agent]) # 两辆车在这一帧的距离
            # # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # distances_landmark = np.array([np.linalg.norm([agent.state.p_pos[0] * 38 - 4 - ve_landmark[0] * 39 - 5,
            #                                             agent.state.p_pos[1] * 23 + 14 - ve_landmark[1] * 38 - 3]) for ve_landmark in
            #                             landmarks_veh_use])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)

            # 先判断是否会和周围的车辆发生碰撞

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

            agent_x_real = (agent.state.p_pos[0] * 38) - 4
            agent_y_real = (agent.state.p_pos[1] * 23) + 14
            agent_vx_real = (agent.state.p_vel[0] * 21) - 14
            agent_vy_real = (agent.state.p_vel[1] * 12) - 2
            # 当前时刻的角度，也就是当前时刻的上一时刻的角度
            agent_degree_last_real = (agent.state.heading_angle_last1 * 191) - 1
            length_agent = 5
            width_agent = 2
            collide_label = False  # 判断是否发生碰撞的标签，True代表发生了碰撞，False代表没发生碰撞 (只要和一辆车发生碰撞，就算有碰撞)
            # 先判断是否和其他agent发生碰撞
            if len(vehs_agent) != 0:
                for i in range(len(vehs_agent)):
                    other_obj_x_real = (vehs_agent[i][0] * 38) - 4
                    other_obj_y_real = (vehs_agent[i][1] * 23) + 14
                    other_obj_vx_real = (vehs_agent[i][2] * 21) - 14
                    other_obj_vy_real = (vehs_agent[i][3] * 12) - 2
                    other_obj_degree_last_real = (vehs_agent[i][4] * 191) - 1
                    # 当前时刻的角度，也就是当前时刻的上一时刻的角度

                    # 绘制两个矩形
                    other_obj_length = length_agent
                    other_obj_width = width_agent
                    # 计算矩形的四个顶点坐标
                    vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent, width_agent,
                                                                  agent_degree_last_real)
                    vertices_other_obj = calculate_rectangle_vertices(other_obj_x_real, other_obj_y_real, other_obj_length,
                                                                      other_obj_width, other_obj_degree_last_real)
                    # 判断两个矩阵是否有交集
                    intersect_a_other_obj = check_intersection(vertices_agent, vertices_other_obj)
                    if intersect_a_other_obj == True:
                        collide_label = True  # 发现碰撞
                        break  # 退出当前这个循环，已经发现碰撞了，就不再继续了

            # 再判断是否和landmark发生碰撞
            if collide_label == False:
                if len(landmarks_veh_use) != 0:
                    for i in range(len(landmarks_veh_use)):
                        other_obj_x_real = (landmarks_veh_use[i][0] * 39) - 5
                        other_obj_y_real = (landmarks_veh_use[i][1] * 38) - 3
                        other_obj_degree_last_real = (landmarks_veh_use[i][4] * 360) - 90
                        # 当前时刻的角度，也就是当前时刻的上一时刻的角度

                        # 绘制两个矩形
                        other_obj_length = length_agent
                        other_obj_width = width_agent
                        # 计算矩形的四个顶点坐标
                        vertices_agent = calculate_rectangle_vertices(agent_x_real, agent_y_real, length_agent,
                                                                      width_agent,
                                                                      agent_degree_last_real)
                        vertices_other_obj = calculate_rectangle_vertices(other_obj_x_real, other_obj_y_real,
                                                                          other_obj_length,
                                                                          other_obj_width, other_obj_degree_last_real)
                        # 判断两个矩阵是否有交集
                        intersect_a_other_obj = check_intersection(vertices_agent, vertices_other_obj)
                        if intersect_a_other_obj == True:
                            collide_label = True  # 发现碰撞
                            break  # 退出当前这个循环，已经发现碰撞了，就不再继续了


            # 重新写，agent和landmark，分开找
            for ii in range(len(vehs_agent)):  # vehs_agent (7, 5)
                if vehs_agent[ii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
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
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
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

                    print('agent.id:',agent.id,'agent_x:',agent.state.p_pos[0],'agent_y:',agent.state.p_pos[1],
                          'agent_x_real:',agent.state.p_pos[0] * 38 - 4,'agent_y_real:',agent.state.p_pos[1] * 23 + 14,
                          'heading_angle_last_1:',agent.state.heading_angle_last1,'heading_angle_last_1_real:',angle,
                          'jiaohu_id:',vehs_agent[ii][5],'jiaohu_x:',vehs_agent[ii][0],'jiaohu_y:',vehs_agent[ii][1],
                          'jiaohu_x_real:',vehs_agent[ii][0] * 38 - 4,'jiaohu_y_real:',vehs_agent[ii][1] * 23 + 14,
                          'b[0]:',b[0],'b[1]:',b[1],'cross:',cross,'angle_jiaodu:',angle_jiaodu)
                    # print()

                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到同方向前方的车辆
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:',agent.id,'有同方向前方的车辆：',vehs_agent[ii][5],veh_same)
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向左前方的车辆
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向左前方的车辆：', vehs_agent[ii][5], veh_left_agents)
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向右前方的车辆
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向右前方的车辆：', vehs_agent[ii][5], veh_right_agents)

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []
            # print('veh_same：',veh_same)
            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                    print('agent.id:',agent.id,'有同方向前方的车辆：',veh_same, veh_neig)
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))


            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    # print('修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(), distances_old[0])
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    print('sorted_veh:',sorted_veh)
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig)
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    # 视野里不止三辆车，那就把最近的三辆车存起来
                    # 找到最近的三辆车
                    # 按照距离升序排序，然后选择前三辆车
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    # 取前三辆车
                    top_3_veh = sorted_veh[:3, :]
                    # 将结果存储为 veh_new
                    veh_new = top_3_veh.copy()
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig,'视野内超过3辆')
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            # print('veh_neig:',np.shape(veh_neig),veh_neig)
            for iii in range(len(landmarks_veh_use)):  # vehs_landmark (n, 5)
                if landmarks_veh_use[iii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1


                    # if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                    #     angle = agent.state.heading_angle_last1 * 191 - 1 + 360
                    # else:
                    #     angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
                    cross = np.cross((b[0], b[1]), ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                                    (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到左前方的landmark
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到右前方的landmark
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []
            # 找到距离agent最近的左右侧视野的landmark，先找左侧
            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            # 再找右侧
            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            # obs = np.array([ (agent_x-(veh_neig[0][0] * 38 - 4)+12)/27, (agent_y-(veh_neig[0][1] * 23 + 14)+14)/18,
            #                  (agent_vx-(veh_neig[0][2] * 21 - 14)+7)/14, (agent_vy-(veh_neig[0][3] * 12 - 2)+5)/6,
            #
            #                  (agent_x-(veh_neig[1][0] * 38 - 4)+15)/30, (agent_y-(veh_neig[1][1] * 23 + 14)+7)/14,
            #                  (agent_vx-(veh_neig[1][2] * 21 - 14)+18)/36, (agent_vy-(veh_neig[1][3] * 12 - 2)+5)/7,
            #
            #                (agent_x - (veh_neig[2][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[2][1] * 23 + 14) + 7) / 14,
            #                (agent_vx - (veh_neig[2][2] * 21 - 14) + 14) / 29,(agent_vy - (veh_neig[2][3] * 12 - 2) + 1) / 2,
            #
            #                (agent_x - (veh_neig[3][0] * 38 - 4) + 14) / 14,(agent_y - (veh_neig[3][1] * 23 + 14) + 7) / 2,
            #                (agent_vx - (veh_neig[3][2] * 21 - 14) - 4) / 9, (agent_vy - (veh_neig[3][3] * 12 - 2) - 0) / 1,
            #
            #                (agent_x - (veh_neig[4][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[4][1] * 23 + 14) + 15) / 24,
            #                (agent_vx - (veh_neig[4][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[4][3] * 12 - 2) + 9) / 15,
            #
            #                (agent_x - (veh_neig[5][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[5][1] * 23 + 14) + 9) / 17,
            #                (agent_vx - (veh_neig[5][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[5][3] * 12 - 2) + 6) / 12,
            #
            #                (agent_x - (veh_neig[6][0] * 38 - 4) + 15) / 10,(agent_y - (veh_neig[6][1] * 23 + 14) + 1) / 7,
            #                (agent_vx - (veh_neig[6][2] * 21 - 14) - 3) / 5,(agent_vy - (veh_neig[6][3] * 12 - 2) - 1) / 3,
            #
            #                (agent_x - (veh_neig_landmark[0][0] * 39 - 5) + 14) / 29,(agent_y - (veh_neig_landmark[0][1] * 38 - 3) + 15) / 30,
            #                (agent_vx - (veh_neig_landmark[0][2] * 31 - 16) + 21) / 35,(agent_vy - (veh_neig_landmark[0][3] * 21 - 10) + 5) / 16,
            #
            #                (agent_x - (veh_neig_landmark[1][0] * 39 - 5) + 15) / 30,(agent_y - (veh_neig_landmark[1][1] * 38 - 3) + 15) / 29,
            #                (agent_vx - (veh_neig_landmark[1][2] * 31 - 16) + 14) / 25,(agent_vy - (veh_neig_landmark[1][3] * 21 - 10) + 7) / 17]).reshape([1, -1])

            # obs_old = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # 从这里接着改！
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
                           # 加的这两个是额外的，在后面训练判别器的train的时候需要去掉


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

            # obs_usefor_reward = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # print('可用于转换的obs是什么形态呢???????????????????????????????:', np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]], veh_neig[0][[0,1,2,3]] )
            # print('转换前的obs是什么形态呢???????????????????????????????:',np.array([abs(np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]] - veh_neig[0][[0,1,2,3]]) for ii in range(3)]))
            # print('转换后的obs是什么形态呢???????????????????????????????:', np.shape(obs),obs)
            # print('之前的obs是什么形态呢???????????????????????????????:', np.shape(obs_old),obs_old)

            des = np.zeros(2)

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new+37) / 59, (d_y_new+4) / 27])  # 和终点的x和y方向的归一化距离

            # print('agent.state.p_last_vx:',agent.state.p_last_vx,'agent.state.p_last_vy',agent.state.p_last_vy)
            # print('obs[0]:', np.shape(obs[0]))
            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]),np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))  # [[0,1,2,3,4,5,6]]  这里的a是下一个时刻的obs
            ini_steps = np.array([agent.state.ini_step])

            # 因为这个函数处理的是当前步的状态，所以都是trj_go_step
            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            current_agent_step = int(trj_go_step)

            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = current_agent_step - i
                # 如果当前步数合法，使用 getattr 获取属性值
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)  # 追加到列表前端
                    # obs_data.append(step_data)  # 追加到列表末尾
                    # print('step_data:', current_step, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))  # 追加到列表前端
                    # obs_data.append(np.zeros([46]))
                    # print('np.zeros([46]:',np.zeros([46]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同a：', np.shape(a))  # (46,)
            # 测试原来的a和obs_data_row的内容形式相不相同a： (46,)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同obs_data_row：', np.shape(obs_data_lstm))  # (21, 46)
            # 测试原来的a和obs_data_row的内容形式相不相同obs_data_row： (21, 46)
            # print('测试原来的a和obs_data_row的格式相不相同：', np.shape(a), np.shape(obs_data_row))
                # else:  # 当前时刻的前i步，已经没有了，那就赋值为0，adjusted_ob_data已经设置好了

            # print('有效的a:',np.shape(a),a)
            # print('有效的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)
        else:
            a = np.zeros([57])  # 原来是24
            if agent.collide == True:
                collide_label = True  # 这个车之前已经撞了
            else:
                collide_label = False  # 没有这个车
            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)

            obs_usefor_reward = np.zeros([45]).reshape([1, -1])

        # print('get_obs输出的数据的shape：','obs_data_lstm:',np.shape(obs_data_lstm),
        #       'obs_usefor_reward:',np.shape(obs_usefor_reward),'a:',np.shape(a))
        # obs_data_lstm: (21, 46) obs_usefor_reward: (1, 46) a: (46,)
        return obs_data_lstm, obs_usefor_reward, a, collide_label
        # return np.concatenate((agent.state.p_pos,obs))

    def observation_AVTEST_nowstep(self, agent, action_n_agent, AV_inf, world):  # HV刚进入交叉口的时候，AV已经在交叉口，需要更新HV的iniobs
        trj_go_step = action_n_agent[3]  # 轨迹走的步数
        if agent.state.p_pos[0] != 0:
        # if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            # 加上agent的step!,然后记录每个step的obs，然后再根据当前的step汇总
            if action_n_agent[2] >= action_n_agent[4]:  #  and action_n_agent[2] <= 183:
                # 环境当前的步数>=轨迹开始时的步数，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                # print('world.landmarks:',np.shape(world.landmarks)) # (30, 185, 7)
                landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]    # (30, 7) 当前步的landmark
                # print('True_if_shape_landmarks_veh:',np.shape(landmarks_veh))  # (30, 7)
                landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]  # (30, 5) 可能小于30
                # print('True_if_shape_landmarks_veh_use:',np.shape(landmarks_veh_use))  # (30, 5)
            # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
            else:  # 这个agent不开始前进，所以没有landmarks，30个全部为0  没有这种情况其实
                landmarks_veh = [np.zeros([7]) for _ in range(30)]
                landmarks_veh_use = [other[:5] for other in landmarks_veh]
                print('这种情况不会有', action_n_agent[2], action_n_agent[4])
                # print('True_else_shape_landmarks_veh:', np.shape(landmarks_veh))  # (30, 7)
                # print('True_else_shape_landmarks_veh_use:', np.shape(landmarks_veh_use))  # (30, 5)

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0]!=0)]  # x,y,vx,vy,deg
            # 把AV的信息加入vehs_agent 对于不同的测试场景，这个要个性化处理，比如对于98场景，AV作为agent0，所以按照如下方式append，agent.id为0
            vehs_agent.append(np.hstack((AV_inf[0], AV_inf[1], AV_inf[2], AV_inf[3], AV_inf[4], int(1))))
            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            print('作为agent的vehs:',np.shape(vehs_agent), vehs_agent)  # (7, 6)
            # vehs = vehs_agent + landmarks_veh_use
            # print('vehs:', np.shape(vehs), vehs)

            veh_same = []  # 车辆同方向最近的一辆车
            veh_left_agents = []  # 车辆左侧的三辆车 agent
            veh_right_agents = []  # 车辆右侧的三辆车 agent
            veh_left_landmark = []  # 车辆左侧的最近一辆车 landmark
            veh_right_landmark = []  # 车辆右侧侧的最近一辆车 landmark

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4)) ** 2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14)) ** 2) for ve in
                                        vehs_agent])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5)) ** 2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3)) ** 2)
                                           for ve_landmark in
                                           landmarks_veh_use])  # 两辆车在这一帧的距离

            # distances_agent = np.array([np.linalg.norm([agent.state.p_pos[0]*38 - 4 - ve[0]*38 - 4,
            #                                       agent.state.p_pos[1] * 23 + 14 - ve[1]* 23 + 14]) for ve in vehs_agent]) # 两辆车在这一帧的距离
            # # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # distances_landmark = np.array([np.linalg.norm([agent.state.p_pos[0] * 38 - 4 - ve_landmark[0] * 39 - 5,
            #                                             agent.state.p_pos[1] * 23 + 14 - ve_landmark[1] * 38 - 3]) for ve_landmark in
            #                             landmarks_veh_use])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # 重新写，agent和landmark，分开找
            for ii in range(len(vehs_agent)):  # vehs_agent (7, 5)
                if vehs_agent[ii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
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
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
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

                    print('agent.id:', agent.id, 'agent_x:', agent.state.p_pos[0], 'agent_y:', agent.state.p_pos[1],
                          'agent_x_real:', agent.state.p_pos[0] * 38 - 4, 'agent_y_real:',
                          agent.state.p_pos[1] * 23 + 14,
                          'heading_angle_last_1:', agent.state.heading_angle_last1, 'heading_angle_last_1_real:', angle,
                          'jiaohu_id:', vehs_agent[ii][5], 'jiaohu_x:', vehs_agent[ii][0], 'jiaohu_y:',
                          vehs_agent[ii][1],
                          'jiaohu_x_real:', vehs_agent[ii][0] * 38 - 4, 'jiaohu_y_real:', vehs_agent[ii][1] * 23 + 14,
                          'b[0]:', b[0], 'b[1]:', b[1], 'cross:', cross, 'angle_jiaodu:', angle_jiaodu)
                    # print()

                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到同方向前方的车辆
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有同方向前方的车辆：', vehs_agent[ii][5], veh_same)
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向左前方的车辆
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向左前方的车辆：', vehs_agent[ii][5], veh_left_agents)
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向右前方的车辆
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向右前方的车辆：', vehs_agent[ii][5], veh_right_agents)

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []
            # print('veh_same：',veh_same)
            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                    print('agent.id:', agent.id, '有同方向前方的车辆：', veh_same, veh_neig)
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))

            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    # print('修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(), distances_old[0])
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    print('sorted_veh:', sorted_veh)
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig)
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    # 视野里不止三辆车，那就把最近的三辆车存起来
                    # 找到最近的三辆车
                    # 按照距离升序排序，然后选择前三辆车
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    # 取前三辆车
                    top_3_veh = sorted_veh[:3, :]
                    # 将结果存储为 veh_new
                    veh_new = top_3_veh.copy()
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig, '视野内超过3辆')
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            # print('veh_neig:',np.shape(veh_neig),veh_neig)
            for iii in range(len(landmarks_veh_use)):  # vehs_landmark (n, 5)
                if landmarks_veh_use[iii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1

                    # if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                    #     angle = agent.state.heading_angle_last1 * 191 - 1 + 360
                    # else:
                    #     angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
                    cross = np.cross((b[0], b[1]),
                                     ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                      (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到左前方的landmark
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到右前方的landmark
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []
            # 找到距离agent最近的左右侧视野的landmark，先找左侧
            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            # 再找右侧
            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            # obs = np.array([ (agent_x-(veh_neig[0][0] * 38 - 4)+12)/27, (agent_y-(veh_neig[0][1] * 23 + 14)+14)/18,
            #                  (agent_vx-(veh_neig[0][2] * 21 - 14)+7)/14, (agent_vy-(veh_neig[0][3] * 12 - 2)+5)/6,
            #
            #                  (agent_x-(veh_neig[1][0] * 38 - 4)+15)/30, (agent_y-(veh_neig[1][1] * 23 + 14)+7)/14,
            #                  (agent_vx-(veh_neig[1][2] * 21 - 14)+18)/36, (agent_vy-(veh_neig[1][3] * 12 - 2)+5)/7,
            #
            #                (agent_x - (veh_neig[2][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[2][1] * 23 + 14) + 7) / 14,
            #                (agent_vx - (veh_neig[2][2] * 21 - 14) + 14) / 29,(agent_vy - (veh_neig[2][3] * 12 - 2) + 1) / 2,
            #
            #                (agent_x - (veh_neig[3][0] * 38 - 4) + 14) / 14,(agent_y - (veh_neig[3][1] * 23 + 14) + 7) / 2,
            #                (agent_vx - (veh_neig[3][2] * 21 - 14) - 4) / 9, (agent_vy - (veh_neig[3][3] * 12 - 2) - 0) / 1,
            #
            #                (agent_x - (veh_neig[4][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[4][1] * 23 + 14) + 15) / 24,
            #                (agent_vx - (veh_neig[4][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[4][3] * 12 - 2) + 9) / 15,
            #
            #                (agent_x - (veh_neig[5][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[5][1] * 23 + 14) + 9) / 17,
            #                (agent_vx - (veh_neig[5][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[5][3] * 12 - 2) + 6) / 12,
            #
            #                (agent_x - (veh_neig[6][0] * 38 - 4) + 15) / 10,(agent_y - (veh_neig[6][1] * 23 + 14) + 1) / 7,
            #                (agent_vx - (veh_neig[6][2] * 21 - 14) - 3) / 5,(agent_vy - (veh_neig[6][3] * 12 - 2) - 1) / 3,
            #
            #                (agent_x - (veh_neig_landmark[0][0] * 39 - 5) + 14) / 29,(agent_y - (veh_neig_landmark[0][1] * 38 - 3) + 15) / 30,
            #                (agent_vx - (veh_neig_landmark[0][2] * 31 - 16) + 21) / 35,(agent_vy - (veh_neig_landmark[0][3] * 21 - 10) + 5) / 16,
            #
            #                (agent_x - (veh_neig_landmark[1][0] * 39 - 5) + 15) / 30,(agent_y - (veh_neig_landmark[1][1] * 38 - 3) + 15) / 29,
            #                (agent_vx - (veh_neig_landmark[1][2] * 31 - 16) + 14) / 25,(agent_vy - (veh_neig_landmark[1][3] * 21 - 10) + 7) / 17]).reshape([1, -1])

            # obs_old = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # 从这里接着改！
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

                            ((agent_x - (veh_neig_landmark[0][0] * 39 - 5)) + 14) / 29 if veh_neig_landmark[0][
                                                                                              0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[0][1] * 38 - 3)) + 15) / 30 if veh_neig_landmark[0][
                                                                                              1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[0][2] * 31 - 16)) + 21) / 35 if veh_neig_landmark[0][
                                                                                                2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[0][3] * 21 - 10)) + 5) / 16 if veh_neig_landmark[0][
                                                                                               3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[1][0] * 39 - 5)) + 15) / 30 if veh_neig_landmark[1][
                                                                                              0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[1][1] * 38 - 3)) + 15) / 29 if veh_neig_landmark[1][
                                                                                              1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[1][2] * 31 - 16)) + 14) / 25 if veh_neig_landmark[1][
                                                                                                2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[1][3] * 21 - 10)) + 7) / 17 if veh_neig_landmark[1][
                                                                                               3] != 0 else 0,
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
            # 加的这两个是额外的，在后面训练判别器的train的时候需要去掉

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
                 veh_neig_landmark[0][2] * 31 - 16, veh_neig_landmark[0][3] * 21 - 10,
                 veh_neig_landmark[0][4] * 360 - 90,
                 veh_neig_landmark[1][0] * 39 - 5, veh_neig_landmark[1][1] * 38 - 3,
                 veh_neig_landmark[1][2] * 31 - 16, veh_neig_landmark[1][3] * 21 - 10,
                 veh_neig_landmark[1][4] * 360 - 90,
                 ]).reshape([1, -1])

            # obs_usefor_reward = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # print('可用于转换的obs是什么形态呢???????????????????????????????:', np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]], veh_neig[0][[0,1,2,3]] )
            # print('转换前的obs是什么形态呢???????????????????????????????:',np.array([abs(np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]] - veh_neig[0][[0,1,2,3]]) for ii in range(3)]))
            # print('转换后的obs是什么形态呢???????????????????????????????:', np.shape(obs),obs)
            # print('之前的obs是什么形态呢???????????????????????????????:', np.shape(obs_old),obs_old)

            des = np.zeros(2)

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new + 37) / 59, (d_y_new + 4) / 27])  # 和终点的x和y方向的归一化距离

            # print('agent.state.p_last_vx:',agent.state.p_last_vx,'agent.state.p_last_vy',agent.state.p_last_vy)
            # print('obs[0]:', np.shape(obs[0]))
            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]), np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))  # [[0,1,2,3,4,5,6]]  这里的a是下一个时刻的obs
            ini_steps = np.array([agent.state.ini_step])

            # 因为这个函数处理的是当前步的状态，所以都是trj_go_step
            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            current_agent_step = int(trj_go_step)

            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = current_agent_step - i
                # 如果当前步数合法，使用 getattr 获取属性值
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)  # 追加到列表前端
                    # obs_data.append(step_data)  # 追加到列表末尾
                    # print('step_data:', current_step, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))  # 追加到列表前端
                    # obs_data.append(np.zeros([46]))
                    # print('np.zeros([46]:',np.zeros([46]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同a：', np.shape(a))  # (46,)
            # 测试原来的a和obs_data_row的内容形式相不相同a： (46,)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同obs_data_row：', np.shape(obs_data_lstm))  # (21, 46)
            # 测试原来的a和obs_data_row的内容形式相不相同obs_data_row： (21, 46)
            # print('测试原来的a和obs_data_row的格式相不相同：', np.shape(a), np.shape(obs_data_row))
                # else:  # 当前时刻的前i步，已经没有了，那就赋值为0，adjusted_ob_data已经设置好了

            # print('有效的a:',np.shape(a),a)
            # print('有效的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)
        else:
            a = np.zeros([57])  # 原来是24
            setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)

            obs_usefor_reward = np.zeros([45]).reshape([1, -1])

        # print('get_obs输出的数据的shape：','obs_data_lstm:',np.shape(obs_data_lstm),
        #       'obs_usefor_reward:',np.shape(obs_usefor_reward),'a:',np.shape(a))
        # obs_data_lstm: (21, 46) obs_usefor_reward: (1, 46) a: (46,)
        return obs_data_lstm, obs_usefor_reward, a
        # return np.concatenate((agent.state.p_pos,obs))

    def observation_AVTEST_nextstep(self, agent, action_n_agent, reset_infos, world):
        trj_go_step = action_n_agent[3]  # 轨迹走的步数
        if agent.state.p_pos[0] != 0:
        # if agent.state.p_pos[0] != 0 or agent.state.p_pos[1] != 0 or agent.state.p_vel[0] != 0 or agent.state.p_vel[1] != 0:
            # 加上agent的step!,然后记录每个step的obs，然后再根据当前的step汇总
            t = world.time
            # world.landmarks = np.array(all_vehss[aa]['landmarks'])  # [30,185,7] 归一化之后的数据
            # print('reset_infos:',reset_infos)
            if reset_infos == [True]: # 这个环境刚重置过，需要获得初始时刻的obs
                # print('action_n_agent:',action_n_agent)
                # action_n_agent：agent.state.p_pos, np.array([0]),
                # np.array([0]),
                # np.array([ini_steps[agent.id][0]]), ini_obs[agent.id]
                if action_n_agent[2] == action_n_agent[4]:  #  and action_n_agent[2] <= 183:
                    # 环境当前的步数==轨迹开始时的步数，且环境当前的步长<=183，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                    # print('world.landmarks:',np.shape(world.landmarks)) # (30, 185, 7)
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2]),:]    # (30, 7)
                    # print('True_if_shape_landmarks_veh:',np.shape(landmarks_veh))  # (30, 7)
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]  # (30, 5) 可能小于30
                    # print('True_if_shape_landmarks_veh_use:',np.shape(landmarks_veh_use))  # (30, 5)
                # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
                else:  # 这个agent不开始前进，所以没有landmarks，30个全部为0  没有这种情况其实
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]
                    print('重置时这种情况不会有',action_n_agent[2], action_n_agent[4])
                    # print('True_else_shape_landmarks_veh:', np.shape(landmarks_veh))  # (30, 7)
                    # print('True_else_shape_landmarks_veh_use:', np.shape(landmarks_veh_use))  # (30, 5)
            else:  # 没有重置，但这个agent也是有效的，说明开始前进了
                # (self.actions[k][i], np.array([env_go_step]),
                # np.array([self.trj_GO_STEP[i][k]]),
                # np.array([self.ini_steps[k][i][0]]),
                # self.ini_obs[k][i])
                if action_n_agent[2] >= action_n_agent[4]: #  and action_n_agent[2] <= 183:
                    # 环境当前的步数>=轨迹开始时的步数，且环境当前的步长<=183，且这个轨迹的初始状态不为0，也就是这个轨迹有效
                    landmarks_veh = world.landmarks[:,int(action_n_agent[2])+1,:]    # [14,7]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh if other[0]!=0]
                # print('agent在：', step_now, '时刻的landmarks_veh_use:', np.shape(landmarks_veh_use), landmarks_veh_use)
                else:  # 这种情况也不会有
                    a = [np.zeros([1, 5]) for _ in range(30)]
                    landmarks_veh = [np.zeros([7]) for _ in range(30)]
                    landmarks_veh_use = [other[:5] for other in landmarks_veh]
                    print('这种情况也不会有',action_n_agent[2],action_n_agent[4])

            vehs_agent = [np.hstack((other.state.p_pos, other.state.p_vel, other.state.heading_angle_last1, other.id))
                          for other in world.agents
                          if (other != agent) and (other.state.p_pos[0] != 0)]  # x,y,vx,vy,deg
            if len(action_n_agent)>62:
                AV_inf = action_n_agent[62:]
                print('step中的AV_inf:', np.shape(AV_inf), AV_inf)
                vehs_agent.append(np.hstack((AV_inf[0], AV_inf[1], AV_inf[2], AV_inf[3], AV_inf[4], int(1))))
            # print('作为landmark的vehs:',np.shape(landmarks_veh_use), landmarks_veh_use) # (30, 5)
            print('作为agent的vehs:',np.shape(vehs_agent), vehs_agent)  # (7, 6)
            # vehs = vehs_agent + landmarks_veh_use
            # print('vehs:', np.shape(vehs), vehs)

            veh_same = []  # 车辆同方向最近的一辆车
            veh_left_agents = []  # 车辆左侧的三辆车 agent
            veh_right_agents = []  # 车辆右侧的三辆车 agent
            veh_left_landmark = []  # 车辆左侧的最近一辆车 landmark
            veh_right_landmark = []  # 车辆右侧侧的最近一辆车 landmark

            distances_agent = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve[0] * 38 - 4)) ** 2 +
                                                ((agent.state.p_pos[1] * 23 + 14) - (ve[1] * 23 + 14)) ** 2) for ve in
                                        vehs_agent])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            distances_landmark = np.array([np.sqrt(((agent.state.p_pos[0] * 38 - 4) - (ve_landmark[0] * 39 - 5)) ** 2 +
                                                   ((agent.state.p_pos[1] * 23 + 14) - (ve_landmark[1] * 38 - 3)) ** 2)
                                           for ve_landmark in
                                           landmarks_veh_use])  # 两辆车在这一帧的距离

            # distances_agent = np.array([np.linalg.norm([agent.state.p_pos[0]*38 - 4 - ve[0]*38 - 4,
            #                                       agent.state.p_pos[1] * 23 + 14 - ve[1]* 23 + 14]) for ve in vehs_agent]) # 两辆车在这一帧的距离
            # # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # distances_landmark = np.array([np.linalg.norm([agent.state.p_pos[0] * 38 - 4 - ve_landmark[0] * 39 - 5,
            #                                             agent.state.p_pos[1] * 23 + 14 - ve_landmark[1] * 38 - 3]) for ve_landmark in
            #                             landmarks_veh_use])  # 两辆车在这一帧的距离
            # print('修改之前的dinstances!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old.shape, distances_old)
            # 重新写，agent和landmark，分开找
            for ii in range(len(vehs_agent)):  # vehs_agent (7, 5)
                if vehs_agent[ii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(vehs_agent[ii][0] * 38 - 4) - (agent.state.p_pos[0] * 38 - 4),
                                  (vehs_agent[ii][1] * 23 + 14) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
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
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
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

                    print('agent.id:', agent.id, 'agent_x:', agent.state.p_pos[0], 'agent_y:', agent.state.p_pos[1],
                          'agent_x_real:', agent.state.p_pos[0] * 38 - 4, 'agent_y_real:',
                          agent.state.p_pos[1] * 23 + 14,
                          'heading_angle_last_1:', agent.state.heading_angle_last1, 'heading_angle_last_1_real:', angle,
                          'jiaohu_id:', vehs_agent[ii][5], 'jiaohu_x:', vehs_agent[ii][0], 'jiaohu_y:',
                          vehs_agent[ii][1],
                          'jiaohu_x_real:', vehs_agent[ii][0] * 38 - 4, 'jiaohu_y_real:', vehs_agent[ii][1] * 23 + 14,
                          'b[0]:', b[0], 'b[1]:', b[1], 'cross:', cross, 'angle_jiaodu:', angle_jiaodu)
                    # print()

                    if (agent_direction == jiaohu_agent_direction) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到同方向前方的车辆
                        veh_same.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有同方向前方的车辆：', vehs_agent[ii][5], veh_same)
                    if (agent_direction != jiaohu_agent_direction) and (cross >= 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向左前方的车辆
                        veh_left_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向左前方的车辆：', vehs_agent[ii][5], veh_left_agents)
                    if (agent_direction != jiaohu_agent_direction) and (cross < 0) \
                            and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到异向右前方的车辆
                        veh_right_agents.append([vehs_agent[ii], distances_agent[ii]])
                        print('agent.id:', agent.id, '有异向右前方的车辆：', vehs_agent[ii][5], veh_right_agents)

            veh_same = np.array(veh_same, dtype=object)
            veh_left_agents = np.array(veh_left_agents, dtype=object)
            veh_right_agents = np.array(veh_right_agents, dtype=object)

            veh_neig = []
            # print('veh_same：',veh_same)
            if len(veh_same) > 0:
                if veh_same[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig.append(veh_same[np.argmin(veh_same[:, 1])][0][:5])
                    print('agent.id:', agent.id, '有同方向前方的车辆：', veh_same, veh_neig)
                else:
                    veh_neig.append(np.zeros(5))
            else:
                veh_neig.append(np.zeros(5))

            for veh_ in [veh_left_agents, veh_right_agents]:
                if 0 < len(veh_) <= 3:
                    # print('修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(), distances_old[0])
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    print('sorted_veh:', sorted_veh)
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(sorted_veh):
                            if sorted_veh[i, 1] <= 15:
                                veh_neig.append(sorted_veh[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig)
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                elif len(veh_) > 3:
                    # 视野里不止三辆车，那就把最近的三辆车存起来
                    # 找到最近的三辆车
                    # 按照距离升序排序，然后选择前三辆车
                    sorted_veh = veh_[veh_[:, 1].argsort()]
                    # 取前三辆车
                    top_3_veh = sorted_veh[:3, :]
                    # 将结果存储为 veh_new
                    veh_new = top_3_veh.copy()
                    for i in range(3):  # 假设你希望至少有三个元素
                        if i < len(veh_new):
                            if veh_new[i, 1] <= 15:
                                veh_neig.append(veh_new[i][0][:5])
                                print('agent.id:', agent.id, '有异方向的车辆：', veh_neig, '视野内超过3辆')
                            else:
                                veh_neig.append(np.zeros(5))
                        else:
                            veh_neig.append(np.zeros(5))
                else:
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))
                    veh_neig.append(np.zeros(5))

            # print('veh_neig:',np.shape(veh_neig),veh_neig)
            for iii in range(len(landmarks_veh_use)):  # vehs_landmark (n, 5)
                if landmarks_veh_use[iii][0] > 0:  # 说明这辆车在交叉口范围内，是有效的，需要进一步处理

                    a = np.array([(landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                  (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)])

                    if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                        angle = (agent.state.heading_angle_last1 * 191 - 1) + 360
                    elif agent.state.heading_angle_last1 * 191 - 1 >= 270:
                        angle = (agent.state.heading_angle_last1 * 191 - 1) - 360
                    else:
                        angle = agent.state.heading_angle_last1 * 191 - 1

                    # if agent.state.heading_angle_last1 * 191 - 1 < -90:  # 把上一时刻的angle转化为-90-270度
                    #     angle = agent.state.heading_angle_last1 * 191 - 1 + 360
                    # else:
                    #     angle = agent.state.heading_angle_last1 * 191 - 1

                    b = np.zeros(2)

                    if 0 <= angle < 90:  # tan>0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == 90:
                        b = np.array([0, 2])
                    elif 90 < angle <= 180:  # tan<0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif 180 < angle < 270:  # tan>0
                        b = np.array([-1, -1 * math.tan(math.radians(angle))])
                    elif angle == 270:  # 负无穷
                        b = np.array([0, -2])
                    elif 270 < angle <= 360:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif -90 < angle < 0:  # tan<0
                        b = np.array([1, math.tan(math.radians(angle))])
                    elif angle == -90:
                        b = np.array([0, -2])

                    # b = np.array([(agent.state.p_vel[0] * 22) - 3, (agent.state.p_vel[1] * 21) - 2])
                    Lb = np.sqrt(b.dot(b))
                    La = np.sqrt(a.dot(a))

                    cos_angle = np.dot(a, b) / (La * Lb)
                    # print('b明明有呀？？',b, b[0],b[1])
                    cross = np.cross((b[0], b[1]),
                                     ((landmarks_veh_use[iii][0] * 39 - 5) - (agent.state.p_pos[0] * 38 - 4),
                                      (landmarks_veh_use[iii][1] * 38 - 3) - (agent.state.p_pos[1] * 23 + 14)))
                    angle_hudu = np.arccos(cos_angle)
                    angle_jiaodu = angle_hudu * (360 / (2 * np.pi))

                    if agent.id <= 2:
                        agent_direction = 'left'
                    else:
                        agent_direction = 'straight'

                    if (cross >= 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到左前方的landmark
                        veh_left_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])
                    if (cross < 0) and (angle_jiaodu >= 0) and (angle_jiaodu <= 90):  # 找到右前方的landmark
                        veh_right_landmark.append([landmarks_veh_use[iii], distances_landmark[iii]])

            veh_left_landmark = np.array(veh_left_landmark, dtype=object)
            veh_right_landmark = np.array(veh_right_landmark, dtype=object)
            veh_neig_landmark = []
            # 找到距离agent最近的左右侧视野的landmark，先找左侧
            if len(veh_left_landmark) > 0:
                if veh_left_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_left_landmark[np.argmin(veh_left_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            # 再找右侧
            if len(veh_right_landmark) > 0:
                if veh_right_landmark[:, 1].min() <= 15:
                    # print('满足条件的修改之后的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', veh_[:, 1].min(), veh_[:, 1])
                    # print('满足条件的修改之前的dinstances计算distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', distances_old[0].min(),
                    #       distances_old[0])
                    veh_neig_landmark.append(veh_right_landmark[np.argmin(veh_right_landmark[:, 1])][0][:5])
                else:
                    veh_neig_landmark.append(np.zeros(5))
            else:
                veh_neig_landmark.append(np.zeros(5))

            agent_x = (agent.state.p_pos[0] * 38) - 4
            agent_y = (agent.state.p_pos[1] * 23) + 14
            agent_vx = (agent.state.p_vel[0] * 21) - 14
            agent_vy = (agent.state.p_vel[1] * 12) - 2

            # obs = np.array([ (agent_x-(veh_neig[0][0] * 38 - 4)+12)/27, (agent_y-(veh_neig[0][1] * 23 + 14)+14)/18,
            #                  (agent_vx-(veh_neig[0][2] * 21 - 14)+7)/14, (agent_vy-(veh_neig[0][3] * 12 - 2)+5)/6,
            #
            #                  (agent_x-(veh_neig[1][0] * 38 - 4)+15)/30, (agent_y-(veh_neig[1][1] * 23 + 14)+7)/14,
            #                  (agent_vx-(veh_neig[1][2] * 21 - 14)+18)/36, (agent_vy-(veh_neig[1][3] * 12 - 2)+5)/7,
            #
            #                (agent_x - (veh_neig[2][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[2][1] * 23 + 14) + 7) / 14,
            #                (agent_vx - (veh_neig[2][2] * 21 - 14) + 14) / 29,(agent_vy - (veh_neig[2][3] * 12 - 2) + 1) / 2,
            #
            #                (agent_x - (veh_neig[3][0] * 38 - 4) + 14) / 14,(agent_y - (veh_neig[3][1] * 23 + 14) + 7) / 2,
            #                (agent_vx - (veh_neig[3][2] * 21 - 14) - 4) / 9, (agent_vy - (veh_neig[3][3] * 12 - 2) - 0) / 1,
            #
            #                (agent_x - (veh_neig[4][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[4][1] * 23 + 14) + 15) / 24,
            #                (agent_vx - (veh_neig[4][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[4][3] * 12 - 2) + 9) / 15,
            #
            #                (agent_x - (veh_neig[5][0] * 38 - 4) + 15) / 30,(agent_y - (veh_neig[5][1] * 23 + 14) + 9) / 17,
            #                (agent_vx - (veh_neig[5][2] * 21 - 14) + 13) / 26,(agent_vy - (veh_neig[5][3] * 12 - 2) + 6) / 12,
            #
            #                (agent_x - (veh_neig[6][0] * 38 - 4) + 15) / 10,(agent_y - (veh_neig[6][1] * 23 + 14) + 1) / 7,
            #                (agent_vx - (veh_neig[6][2] * 21 - 14) - 3) / 5,(agent_vy - (veh_neig[6][3] * 12 - 2) - 1) / 3,
            #
            #                (agent_x - (veh_neig_landmark[0][0] * 39 - 5) + 14) / 29,(agent_y - (veh_neig_landmark[0][1] * 38 - 3) + 15) / 30,
            #                (agent_vx - (veh_neig_landmark[0][2] * 31 - 16) + 21) / 35,(agent_vy - (veh_neig_landmark[0][3] * 21 - 10) + 5) / 16,
            #
            #                (agent_x - (veh_neig_landmark[1][0] * 39 - 5) + 15) / 30,(agent_y - (veh_neig_landmark[1][1] * 38 - 3) + 15) / 29,
            #                (agent_vx - (veh_neig_landmark[1][2] * 31 - 16) + 14) / 25,(agent_vy - (veh_neig_landmark[1][3] * 21 - 10) + 7) / 17]).reshape([1, -1])

            # obs_old = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # 从这里接着改！
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

                            ((agent_x - (veh_neig_landmark[0][0] * 39 - 5)) + 14) / 29 if veh_neig_landmark[0][
                                                                                              0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[0][1] * 38 - 3)) + 15) / 30 if veh_neig_landmark[0][
                                                                                              1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[0][2] * 31 - 16)) + 21) / 35 if veh_neig_landmark[0][
                                                                                                2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[0][3] * 21 - 10)) + 5) / 16 if veh_neig_landmark[0][
                                                                                               3] != 0 else 0,

                            ((agent_x - (veh_neig_landmark[1][0] * 39 - 5)) + 15) / 30 if veh_neig_landmark[1][
                                                                                              0] != 0 else 0,
                            ((agent_y - (veh_neig_landmark[1][1] * 38 - 3)) + 15) / 29 if veh_neig_landmark[1][
                                                                                              1] != 0 else 0,
                            ((agent_vx - (veh_neig_landmark[1][2] * 31 - 16)) + 14) / 25 if veh_neig_landmark[1][
                                                                                                2] != 0 else 0,
                            ((agent_vy - (veh_neig_landmark[1][3] * 21 - 10)) + 7) / 17 if veh_neig_landmark[1][
                                                                                               3] != 0 else 0,
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
            # 加的这两个是额外的，在后面训练判别器的train的时候需要去掉

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
                 veh_neig_landmark[0][2] * 31 - 16, veh_neig_landmark[0][3] * 21 - 10,
                 veh_neig_landmark[0][4] * 360 - 90,
                 veh_neig_landmark[1][0] * 39 - 5, veh_neig_landmark[1][1] * 38 - 3,
                 veh_neig_landmark[1][2] * 31 - 16, veh_neig_landmark[1][3] * 21 - 10,
                 veh_neig_landmark[1][4] * 360 - 90,
                 ]).reshape([1, -1])

            # obs_usefor_reward = np.array(
            #     [np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0, 1, 2, 3]] - veh_neig[ii][[0, 1, 2, 3]]
            #      for ii in range(2)]).reshape([1, -1])
            # print('可用于转换的obs是什么形态呢???????????????????????????????:', np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]], veh_neig[0][[0,1,2,3]] )
            # print('转换前的obs是什么形态呢???????????????????????????????:',np.array([abs(np.concatenate((agent.state.p_pos, agent.state.p_vel))[[0,1,2,3]] - veh_neig[0][[0,1,2,3]]) for ii in range(3)]))
            # print('转换后的obs是什么形态呢???????????????????????????????:', np.shape(obs),obs)
            # print('之前的obs是什么形态呢???????????????????????????????:', np.shape(obs_old),obs_old)

            des = np.zeros(2)

            p_pos_x_new = agent.state.p_pos[0] * 38 - 4
            p_pos_y_new = agent.state.p_pos[1] * 23 + 14

            des_x_always_ = agent.state.p_des[0] * 38 - 4
            des_y_always_ = agent.state.p_des[1] * 23 + 14

            d_x_new = des_x_always_ - p_pos_x_new
            d_y_new = des_y_always_ - p_pos_y_new

            des = np.array([(d_x_new + 37) / 59, (d_y_new + 4) / 27])  # 和终点的x和y方向的归一化距离

            # print('agent.state.p_last_vx:',agent.state.p_last_vx,'agent.state.p_last_vy',agent.state.p_last_vy)
            # print('obs[0]:', np.shape(obs[0]))
            a = np.concatenate((agent.state.p_pos, agent.state.p_vel, des, np.array([agent.state.heading_angle_last1]),
                                np.array([agent.state.p_dis]), np.array([agent.state.p_last_vx]),
                                np.array([agent.state.p_last_vy]),
                                obs[0]))  # [[0,1,2,3,4,5,6]]  这里的a是下一个时刻的obs
            ini_steps = np.array([agent.state.ini_step])
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
                current_agent_step = int(trj_go_step)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
                current_agent_step = int(trj_go_step) + 1

            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = current_agent_step - i
                # 如果当前步数合法，使用 getattr 获取属性值
                if current_step >= 0:
                    step_data = getattr(agent.state, f"ob_trj_step_{int(current_step)}")
                    obs_data.insert(0, step_data)  # 追加到列表前端
                    # obs_data.append(step_data)  # 追加到列表末尾
                    # print('step_data:', current_step, step_data)
                else:
                    obs_data.insert(0, np.zeros([57]))  # 追加到列表前端
                    # obs_data.append(np.zeros([46]))
                    # print('np.zeros([46]:',np.zeros([46]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同a：', np.shape(a))  # (46,)
            # 测试原来的a和obs_data_row的内容形式相不相同a： (46,)
            # print('current_agent_step:',current_agent_step,'测试原来的a和obs_data_row的内容形式相不相同obs_data_row：', np.shape(obs_data_lstm))  # (21, 46)
            # 测试原来的a和obs_data_row的内容形式相不相同obs_data_row： (21, 46)
            # print('测试原来的a和obs_data_row的格式相不相同：', np.shape(a), np.shape(obs_data_row))
                # else:  # 当前时刻的前i步，已经没有了，那就赋值为0，adjusted_ob_data已经设置好了

            # print('有效的a:',np.shape(a),a)
            # print('有效的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)
        else:
            a = np.zeros([57])  # 原来是24
            if reset_infos == [True]:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)}", a)
            else:
                setattr(agent.state, f"ob_trj_step_{int(trj_go_step)+1}", a)
            # 假设 trj_go_step 是当前的时刻
            obs_data = []

            # 遍历前 20 个时刻
            for i in range(0, 21):
                # 计算当前时刻的步数
                current_step = int(trj_go_step) - i
                obs_data.append(np.zeros([57]))

            # 将 trj_data 拼接成21行
            obs_data_lstm = np.vstack(obs_data)
            # 将 trj_data 拼接成一行
            # obs_data_row = np.concatenate(obs_data)

            obs_usefor_reward = np.zeros([45]).reshape([1, -1])
            # ini_steps = np.zeros([1])
            # print('0的a:', np.shape(a), a)
            # print('0的obs_usefor_reward:', np.shape(obs_usefor_reward), obs_usefor_reward)

        # print('get_obs输出的数据的shape：','obs_data_lstm:',np.shape(obs_data_lstm),
        #       'obs_usefor_reward:',np.shape(obs_usefor_reward),'a:',np.shape(a))
        # obs_data_lstm: (21, 46) obs_usefor_reward: (1, 46) a: (46,)
        return obs_data_lstm, obs_usefor_reward, a
        # return np.concatenate((agent.state.p_pos,obs))

    def done(self, agent, world):
        # print('运行这个函数了吗？')
        if world.time >= 229:  # 这里其实没用到，因为发现time并没有被更新
            return True
        # elif agent.collide == True:
        #     # print('超出交叉口边界的位置:',agent.state.p_pos[0]*38-4, agent.state.p_pos[1]*23+14)
        #     return True
        elif agent.end_label == True:
            return True
        elif agent.state.step >= 230:
            return True
        else:
            return False
# %%%
# a = []np.array
# for agent in world.agents:
#     a.append(np.concatenate((np.concatenate((agent.state.p_pos, agent.state.p_vel)),obs)))
