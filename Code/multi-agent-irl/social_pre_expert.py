'''
Strategies SVO for Expert Scenarios
'''
import pandas as pd
from irl.render import makeModel, render, get_dis, render_discrimination_nogenerate, render_discrimination_expert
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import cv2
import glob
from PIL import Image
from matplotlib.patches import Wedge


scenario_test = 8
scenario_testfile_name = str(scenario_test) + 'west_left_expert'


num_agents = 8
num_left = 3
num_straight = 5
mids = ['0150']  # best_model
mid = mids[0]
env_id = 'trj_intersection_4'
training_label = False
model = makeModel(env_id, scenario_test, training_label)
model_best = 3  # best_model_number

# model obtained by training
path = r'Code\multi-agent-irl\irl\mack\multi-agent-trj\logger\airl\trj_intersection_4\decentralized' \
       r'\s-90\seed-13\m_0' + mid
path_d = r'Code/multi-agent-irl/irl/mack/multi-agent-trj/logger/airl/trj_intersection_4/decentralized' \
       r'/s-90/seed-13/'

# expert_trj path
path_pkl = r'\Data\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl'
f_expert = open(path_pkl, 'rb')
all_model_expert_trj = pickle.load(f_expert)
one_model_expert_trj = all_model_expert_trj[scenario_test]
sample_trajs = render_discrimination_expert(path, path_d, model, env_id, mid, one_model_expert_trj, scenario_test, training_label)

# save path
root_SVO_save = r'Code\results_evaluate\s-90\attention_video\social_pre'
os.makedirs(root_SVO_save, exist_ok=True)

folder_path = os.path.join(root_SVO_save, 'trj_data', 'expert', scenario_testfile_name)
os.makedirs(folder_path, exist_ok=True)

scenario_name = "sample_trajsspre_withSVO_" + str(scenario_testfile_name)

pkl_file_path = os.path.join(folder_path, f"{scenario_name}.pkl")

with open(pkl_file_path, 'wb') as f:
    pkl.dump(sample_trajs, f)

all_expert_trj = sample_trajs


scenario_trj_ob = all_expert_trj[0]['ob']
scenario_trj_ac = all_expert_trj[0]['ac']
scenario_trj_att_socialpre = all_expert_trj[0]['all_social']

left_scenario_use = pd.DataFrame()
straight_scenario_use = pd.DataFrame()

for left_i in range(3):
    left_trj_i_array = scenario_trj_ob[left_i]
    left_trj_i = pd.DataFrame(left_trj_i_array)
    left_trj_i_use = left_trj_i[left_trj_i.iloc[:, 0] != 0]

    left_ac_i_array = scenario_trj_ac[left_i]
    left_ac_i = pd.DataFrame(left_ac_i_array)
    left_ac_i_use = left_ac_i.loc[left_trj_i_use.index][1]
    left_trj_i_use['angle_now_real'] = left_trj_i_use.iloc[:, 6]*191-1 + ((2.8*(left_ac_i_use+1))/2) - 0.3
    left_trj_i_use['heading_angle_last1_real'] = left_trj_i_use.iloc[:, 6] * 191 - 1

    left_socialpre_i_array = scenario_trj_att_socialpre[left_i]
    left_socialpre_i = pd.DataFrame(left_socialpre_i_array)
    left_socialpre_i_use = left_socialpre_i.loc[left_trj_i_use.index][0]

    left_i_use_0 = left_trj_i_use
    left_i_use = pd.concat([left_i_use_0, left_socialpre_i_use], axis=1)

    left_i_use['socialpre_liji'] = np.cos(left_socialpre_i_use)
    left_i_use['socialpre_lita'] = np.sin(left_socialpre_i_use)

    left_i_use = left_i_use.assign(id=left_i)
    left_i_use_reset = left_i_use.reset_index()

    left_i_use_reset = left_i_use_reset.rename(columns={'index': 'time'})

    left_scenario_use = pd.concat([left_scenario_use, left_i_use_reset],axis=0)


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
                    'landmark1_heading_last','landmark2_heading_last','min_distance_to_lane', 'last_delta_angle','ini_step',
                    'angle_now_real','heading_angle_last1_real',
                    'socialpre','socialpre_liji','socialpre_lita',
                    'id', 'direction', 'width', 'length', 'model_id']

left_scenario_use['x_real'] = left_scenario_use['x'] * 38 - 4
left_scenario_use['y_real'] = left_scenario_use['y'] * 23 + 14


for straight_i in range(3,8):
    straight_trj_i_array = scenario_trj_ob[straight_i]
    straight_trj_i = pd.DataFrame(straight_trj_i_array)
    straight_trj_i_use = straight_trj_i[straight_trj_i.iloc[:, 0] != 0]

    straight_ac_i_array = scenario_trj_ac[straight_i]
    straight_ac_i = pd.DataFrame(straight_ac_i_array)
    straight_ac_i_use = straight_ac_i.loc[straight_trj_i_use.index][1]
    straight_trj_i_use['angle_now_real'] = straight_trj_i_use.iloc[:, 6] * 191 - 1 + ((2.4*(straight_ac_i_use+1))/2) - 1.2
    straight_trj_i_use['heading_angle_last1_real'] = straight_trj_i_use.iloc[:, 6] * 191 - 1

    straight_socialpre_i_array = scenario_trj_att_socialpre[straight_i]
    straight_socialpre_i = pd.DataFrame(straight_socialpre_i_array)
    straight_socialpre_i_use = straight_socialpre_i.loc[straight_trj_i_use.index][0]

    straight_i_use_0 = straight_trj_i_use
    straight_i_use = pd.concat([straight_i_use_0, straight_socialpre_i_use], axis=1)

    straight_i_use['socialpre_liji'] = np.cos(straight_socialpre_i_use)
    straight_i_use['socialpre_lita'] = np.sin(straight_socialpre_i_use)

    straight_i_use = straight_i_use.assign(id=straight_i)
    straight_i_use_reset = straight_i_use.reset_index()

    straight_i_use_reset = straight_i_use_reset.rename(columns={'index': 'time'})

    straight_scenario_use = pd.concat([straight_scenario_use, straight_i_use_reset], axis=0)

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
                    'landmark1_heading_last','landmark2_heading_last','min_distance_to_lane', 'last_delta_angle','ini_step',
                    'angle_now_real','heading_angle_last1_real',
                    'socialpre','socialpre_liji','socialpre_lita',
                    'id', 'direction', 'width', 'length', 'model_id']

straight_scenario_use['x_real'] = straight_scenario_use['x'] * 38 - 4
straight_scenario_use['y_real'] = straight_scenario_use['y'] * 23 + 14


scenario_data = pd.concat([left_scenario_use,straight_scenario_use], axis=0)

scenario_data_file_path = os.path.join(
    folder_path,
    f'scenario{scenario_test}_expert_pre_data.csv'
)

scenario_data.to_csv(scenario_data_file_path)

# SVO video
def plot_top_view_ani_with_lidar_label(trj_in, frame_id_in, veh_id):
    # this function plots one single frame of the top view video
    image_size = 100
    fig, ax = plt.subplots(figsize=(6, 6))
    center = (image_size // 2, image_size // 2)
    radius = 30

    circle = plt.Circle(center, radius, color='black', alpha=0.2)
    ax.add_patch(circle)

    ax.plot([center[0], center[0] + radius], [center[1], center[1]], color='blue', linewidth=2)

    radians = trj_in['socialpre'][frame_id_in]

    if radians < 0:
        start_angle = np.degrees(radians)
        end_angle = 0
    else:
        start_angle = 0
        end_angle = np.degrees(radians)

    wedge = Wedge(center, radius, start_angle, end_angle, color='red', alpha=0.5)
    ax.add_patch(wedge)

    accel_end_point = (center[0] + (radius) * np.cos(radians),
                       center[1] + (radius) * np.sin(radians))
    ax.plot([center[0], accel_end_point[0]], [center[1], accel_end_point[1]], color='green', linewidth=2)

    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')

    trj_save_name = os.path.join(
        root_SVO_save,
        'figure_save',
        'temp_top_view_figure'
    )
    os.makedirs(trj_save_name, exist_ok=True)

    trj_save_name_2 = os.path.join(
        trj_save_name,
        f'top_scenario_label_{veh_id}_frame_{frame_id_in}_social_pre.jpg'
    )
    plt.savefig(trj_save_name_2)
    plt.close(fig)

def top_view_video_generation(start_frame_num, veh_id):
    # this function experts one top view video based on top view figures from one segment
    img_array = []

    temp_top_view_figure_path = os.path.join(root_SVO_save, 'figure_save', 'temp_top_view_figure')
    os.makedirs(temp_top_view_figure_path, exist_ok=True)

    for num in range(start_frame_num,
                     start_frame_num + len(os.listdir(temp_top_view_figure_path))):

        image_filename = os.path.join(
            root_SVO_save,
            'figure_save',
            'temp_top_view_figure',
            f'top_scenario_label_{veh_id}_frame_{num}_social_pre.jpg'
        )
        img = Image.open(image_filename)  # Image loaded successfully.
        img = np.array(img)
        # img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # img

    video_save_path = os.path.join(
        root_SVO_save,
        'figure_save',
        'top_view_video', 'expert',
        scenario_testfile_name
    )
    os.makedirs(video_save_path, exist_ok=True)

    video_save_name = os.path.join(
        video_save_path,
        f'_scenario_label_{veh_id}.avi'
    )
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('top view video made success')
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join(root_SVO_save, 'figure_save', 'temp_top_view_figure', "*.jpg"))

    for f in filelist:
        os.remove(f)

vehs = [name[1] for name in scenario_data.groupby(['id'])]
for veh in vehs:
    veh.index = range(len(veh))
    veh_id = veh['id'][0]
    start_frame_num = 0
    end_frame_num = len(veh) - 1
    for frame_id in range(start_frame_num, int(end_frame_num) + 1):
        plot_top_view_ani_with_lidar_label(veh, frame_id, veh_id)
    # ---------- video generation ----------

    top_view_video_generation(start_frame_num, veh_id)

    filelist = glob.glob(os.path.join(root_SVO_save, 'figure_save', 'temp_top_view_figure', "*.jpg"))
    for f in filelist:
        os.remove(f)
