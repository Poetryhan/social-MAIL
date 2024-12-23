'''针对一个场景的最佳模型，读取利己利他值，我们学习到的是一个θ，越接近于0，越利己
   并将利己利他值绘制为一个视频'''


import numpy as np
import pandas as pd
from tqdm import tqdm
from irl.render import makeModel, render, get_dis, render_discrimination, render_discrimination_expert
from irl.mack.kfac_discriminator_airl import Discriminator
import pickle as pkl
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from shapely import geometry
from scipy.spatial import distance
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from utils.DataReader import read_tracks_all, read_tracks_meta, read_light
# from loguru import logger
import glob
import os
import argparse
import pickle
import cv2
import glob
from PIL import Image
from matplotlib.patches import Wedge
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


scenario_test = 79
scenario_test_east = str(scenario_test) + 'west_left'
# 读取判别器
num_agents = 8
num_left = 3
num_straight = 5
# mids=['0100']
mids = ['0310']
model_best = 11
print('mids:', len(mids))
# 上一行的print结果如下 mids: ['0001', '0100', '0200', '0300', '0400', '0489']

mid = mids[0]
# % rendering
env_id = 'trj_intersection_4'
model = makeModel(env_id)


#%% plot_reward

path = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight' \
         r'\multi-agent-irl\irl\mack\multi-agent-trj\logger\airl\trj_intersection_4\decentralized' \
         r'\v31\l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0\seed-13\m_0' + mid
path_d = r'D:/Study/同济大学/博三/面向自动驾驶测试的仿真/sinD_nvn_xuguan/ATT-social-iniobs/MA_Intersection_straight' \
     r'/multi-agent-irl/irl/mack/multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
     r'/v31/l-0.001-b-100-d-0.1-c-500-l2-0.1-iter-10-r-0.0/seed-13/'

scenario_name = "sample_trajss_" + str(scenario_test)
path_pkl = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan' \
                      r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight' \
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f_expert = open(path_pkl, 'rb')
all_model_expert_trj = pickle.load(f_expert)
one_model_expert_trj = all_model_expert_trj[scenario_test]
sample_trajs = render_discrimination_expert(path, path_d, model, env_id, mid, one_model_expert_trj)

# 指定文件夹路径
folder_path = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
              fr'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\社会倾向\专门生成生成轨迹的数据\expert\{scenario_test_east}'
# 将 scenario 加入文件夹路径中

# 创建文件夹（如果不存在）
os.makedirs(folder_path, exist_ok=True)
# scenario 名称
scenario_name = "sample_trajss_" + str(scenario_test)

# 构建保存 pkl 文件的完整路径
pkl_file_path = os.path.join(folder_path, f"{scenario_name}.pkl")

# 将数据 sample_trajss 存入 pkl 文件
with open(pkl_file_path, 'wb') as f:
    pkl.dump(sample_trajs, f)

all_expert_trj = sample_trajs

# 根据sample_trajs绘制出利己性利他性变化视频
scenario_trj_ob = all_expert_trj[0]['ob']
# scenario_trj_att_weight = all_expert_trj[0]['all_attention_weight']  # shape为（8, 230，21,10,10）
scenario_trj_ac = all_expert_trj[0]['ac']
scenario_trj_att_socialpre = all_expert_trj[0]['all_social']  # shape为（8, 230，1）
print('scenario_trj_ob:',type(scenario_trj_ob),len(scenario_trj_ob))
# print('scenario_trj_att_weight:', type(scenario_trj_att_weight), len(scenario_trj_att_weight))
print('scenario_trj_ac:', type(scenario_trj_ac), len(scenario_trj_ac))
print('scenario_trj_att_socialpre:', type(scenario_trj_att_socialpre), len(scenario_trj_att_socialpre))

left_scenario_use = pd.DataFrame()  # 存储该场景下有用的左转车轨迹数据
straight_scenario_use = pd.DataFrame()  # 存储该场景下有用的直行车轨迹数据

# 提取左转车
for left_i in range(3):
    left_trj_i_array = scenario_trj_ob[left_i][:,0:57]
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

    # left_weight_i_array = scenario_trj_att_weight[left_i]
    # print('left_weight_i_array:', type(left_weight_i_array), len(left_weight_i_array))
    # bool_index = np.zeros(left_weight_i_array.shape[0], dtype=bool)
    # bool_index[left_trj_i_use.index] = True
    # # 使用布尔索引提取对应的第一维度
    # left_weight_i_use_array = left_weight_i_array[bool_index]
    # print('left_weight_i_use_array:',type(left_weight_i_use_array), left_weight_i_use_array)
    # left_weight_i_use_array_2 = left_weight_i_use_array[:, 20, 0]
    # print('left_weight_i_use_array_2:', type(left_weight_i_use_array_2), left_weight_i_use_array_2)
    # left_weight_i_use = pd.DataFrame(left_weight_i_use_array_2)
    # left_weight_i_use.index = left_trj_i_use.index
    # print('left_weight_i_use:', type(left_weight_i_use), left_weight_i_use)
    # left_weight_i_use = left_weight_i_use_df.loc[left_trj_i_use.index][20][0]  #  10个参数，对应着主体agent对自身以及其他9个交互对象的关注度，10个参数的和等于1

    left_socialpre_i_array = scenario_trj_att_socialpre[left_i]
    print('left_socialpre_i_array:', type(left_socialpre_i_array), len(left_socialpre_i_array))
    left_socialpre_i = pd.DataFrame(left_socialpre_i_array)
    print('left_socialpre_i:', type(left_socialpre_i), left_socialpre_i.shape, left_socialpre_i)
    left_socialpre_i_use = left_socialpre_i.loc[left_trj_i_use.index][0]  # 只有一列是delat_angle
    print('left_socialpre_i_use:', type(left_socialpre_i_use), left_socialpre_i_use.shape, left_socialpre_i_use)

    # left_i_use_0 = pd.concat([left_trj_i_use, left_weight_i_use], axis=1)  # 横向拼接
    left_i_use = pd.concat([left_trj_i_use, left_socialpre_i_use], axis=1)  # 横向拼接

    # 计算 socialpre_liji socialpre_lita列
    left_i_use['socialpre_liji'] = np.cos(left_socialpre_i_use)  # v31版本训练反了，v5以及之后要调整过来
    left_i_use['socialpre_lita'] = np.sin(left_socialpre_i_use)

    print('left_i_use:',type(left_i_use),left_i_use.shape,left_i_use)

    left_i_use = left_i_use.assign(id=left_i)
    # 将索引转换为time列，并删除原始索引
    left_i_use_reset = left_i_use.reset_index()

    # 将列名 'index' 更改为 'time'，位于第一列
    left_i_use_reset = left_i_use_reset.rename(columns={'index': 'time'})

    left_scenario_use = pd.concat([left_scenario_use, left_i_use_reset],axis=0)  # 纵向拼接
    print('left_scenario_use:', type(left_scenario_use), left_scenario_use.shape, left_scenario_use)


left_scenario_use['direction'] = 'left'
left_scenario_use['width'] = 1.8
left_scenario_use['length'] = 4.6
left_scenario_use['model_id'] = mid

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
                    'socialpre','socialpre_liji','socialpre_lita',
                    'id', 'direction', 'width', 'length', 'model_id']  # 九个交互对象的state，10+1*4+3*4+3*4+2*4

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
    straight_trj_i_array = scenario_trj_ob[straight_i][:,0:57]
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

    # straight_weight_i_array = scenario_trj_att_weight[straight_i]
    # print('straight_weight_i_array:', type(straight_weight_i_array), len(straight_weight_i_array))
    # bool_index = np.zeros(straight_weight_i_array.shape[0], dtype=bool)
    # bool_index[straight_trj_i_use.index] = True
    # # 使用布尔索引提取对应的第一维度
    # straight_weight_i_use_array = straight_weight_i_array[bool_index]
    # print('straight_weight_i_use_array:', type(straight_weight_i_use_array), straight_weight_i_use_array)
    # straight_weight_i_use_array_2 = straight_weight_i_use_array[:, 20, 0]
    # print('straight_weight_i_use_array_2:', type(straight_weight_i_use_array_2), straight_weight_i_use_array_2)
    # straight_weight_i_use = pd.DataFrame(straight_weight_i_use_array_2)
    # straight_weight_i_use.index = straight_trj_i_use.index
    # print('straight_weight_i_use:', type(straight_weight_i_use), straight_weight_i_use)
    # straight_weight_i_use = straight_weight_i_use_df.loc[straight_trj_i_use.index][20][0]  #  10个参数，对应着主体agent对自身以及其他9个交互对象的关注度，10个参数的和等于1

    straight_socialpre_i_array = scenario_trj_att_socialpre[straight_i]
    print('straight_socialpre_i_array:', type(straight_socialpre_i_array), len(straight_socialpre_i_array))
    straight_socialpre_i = pd.DataFrame(straight_socialpre_i_array)
    print('straight_socialpre_i:', type(straight_socialpre_i), straight_socialpre_i.shape, straight_socialpre_i)
    straight_socialpre_i_use = straight_socialpre_i.loc[straight_trj_i_use.index][0]  # 只有一列是delat_angle
    print('straight_socialpre_i_use:', type(straight_socialpre_i_use), straight_socialpre_i_use.shape, straight_socialpre_i_use)

    # straight_i_use_0 = pd.concat([straight_trj_i_use, straight_weight_i_use], axis=1)  # 横向拼接
    straight_i_use = pd.concat([straight_trj_i_use, straight_socialpre_i_use], axis=1)  # 横向拼接

    # 计算 socialpre_liji socialpre_lita列
    straight_i_use['socialpre_liji'] = np.cos(straight_socialpre_i_use)  # v31版本训练反了，v5以及之后要调整过来
    straight_i_use['socialpre_lita'] = np.sin(straight_socialpre_i_use)

    print('straight_i_use:', type(straight_i_use), straight_i_use.shape, straight_i_use)

    straight_i_use = straight_i_use.assign(id=straight_i)

    # 将索引转换为time列，并删除原始索引
    straight_i_use_reset = straight_i_use.reset_index()

    # 将列名 'index' 更改为 'time'，位于第一列
    straight_i_use_reset = straight_i_use_reset.rename(columns={'index': 'time'})

    straight_scenario_use = pd.concat([straight_scenario_use, straight_i_use_reset], axis=0)  # 纵向拼接
    print('straight_scenario_use:', type(straight_scenario_use), straight_scenario_use.shape, straight_scenario_use)

straight_scenario_use['direction'] = 'straight'
straight_scenario_use['width'] = 1.8
straight_scenario_use['length'] = 4.2
straight_scenario_use['model_id'] = mid

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
                    'same_heading_last','left1_heading_last','left2_heading_last','left3_heading_last',
                    'right1_heading_last','right2_heading_last','right3_heading_last',
                    'landmark1_heading_last','landmark2_heading_last','min_distance_to_lane', 'last_delta_angle',
                    'angle_now_real','heading_angle_last1_real',
                    'socialpre','socialpre_liji','socialpre_lita',
                    'id', 'direction', 'width', 'length', 'model_id']  # 九个交互对象的state，10+1*4+3*4+3*4+2*4

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
scenario_data_file_path = f'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                          f'/MA_Intersection_straight/results_evaluate/v31/有注意力的视频/社会倾向' \
                          f'/专门生成生成轨迹的数据/expert/{scenario_test_east}/scenario{scenario_test}_data.csv'
scenario_data.to_csv(scenario_data_file_path)

# 根据每一辆车，画出角度图，然后拼接成视频

def plot_top_view_ani_with_lidar_label(trj_in, frame_id_in, veh_id):
    # this function plots one single frame of the top view video
    image_size = 100
    fig, ax = plt.subplots(figsize=(6, 6))  # 创建图形和轴对象
    center = (image_size // 2, image_size // 2)
    radius = 30  # 半径为3，放大为30像素

    # 绘制圆形
    circle = plt.Circle(center, radius, color='black', alpha=0.2)
    ax.add_patch(circle)

    # 绘制0度线段
    ax.plot([center[0], center[0] + radius], [center[1], center[1]], color='blue', linewidth=2)

    # 绘制指示方向
    radians = trj_in['socialpre'][frame_id_in]

    if radians < 0:
        start_angle = np.degrees(radians)
        end_angle = 0
    else:
        start_angle = 0
        end_angle = np.degrees(radians)

    # 绘制扇形
    wedge = Wedge(center, radius, start_angle, end_angle, color='red', alpha=0.5)
    ax.add_patch(wedge)

    # 加速指示方向的线段
    accel_end_point = (center[0] + (radius) * np.cos(radians),
                       center[1] + (radius) * np.sin(radians))
    ax.plot([center[0], accel_end_point[0]], [center[1], accel_end_point[1]], color='green', linewidth=2)

    # 设置轴范围和纵横比
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    # 保存图像
    trj_save_name = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                    r'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\社会倾向\figure_save' \
                    r'\temp_top_view_figure\top_scenario_label_' + str(veh_id) + '_frame_' + \
                    str(frame_id_in) + '_social_pre.jpg'
    plt.savefig(trj_save_name)
    plt.close(fig)  # 关闭图形对象，释放内存

def top_view_video_generation(start_frame_num, veh_id):
    # this function experts one top view video based on top view figures from one segment
    img_array = []
    for num in range(start_frame_num,
                     start_frame_num + len(os.listdir(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                                      r'\MA_Intersection_straight\results_evaluate\v31'
                                                      r'\有注意力的视频\社会倾向\figure_save\temp_top_view_figure/'))):
        image_filename = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                         r'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\社会倾向' \
                         r'\figure_save/temp_top_view_figure/' \
                         + 'top_scenario_label_' + str(veh_id) + \
                         '_frame_' + str(num) + '_social_pre.jpg'
        # print(image_filename)
        img = Image.open(image_filename)  # Image loaded successfully.
        img = np.array(img)
        # img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # img
    video_save_name = fr'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                      fr'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\社会倾向' \
                      fr'\figure_save\top_view_video\expert\{scenario_test_east}\'_scenario_label_{veh_id}.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('top view video made success')
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                      r'\MA_Intersection_straight\results_evaluate\v31\有注意力的视频\社会倾向\figure_save'
                                      r'\temp_top_view_figure\\', "*.jpg"))
    for f in filelist:
        os.remove(f)


vehs = [name[1] for name in scenario_data.groupby(['id'])]
for veh in vehs:
    veh.index = range(len(veh))
    veh_id = veh['id'][0]
    start_frame_num = 0
    end_frame_num = len(veh) - 1
    for frame_id in range(start_frame_num, int(end_frame_num) + 1):  # 遍历一个场景内每一帧
        # 生成图片
        plot_top_view_ani_with_lidar_label(veh, frame_id, veh_id)
    # ---------- video generation ----------

    top_view_video_generation(start_frame_num, veh_id)  # 生成视频的函数