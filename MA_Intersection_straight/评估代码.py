'''
1.构建每个模型下所有场景下的生成的加速度、delta_angle指标的分布
2.构建每个模型下所有场景下的专家的加速度、delta_angle指标的分布
3.计算每个模型下KL散度
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import matplotlib.mlab as mlab
import os
from sklearn.neighbors import KernelDensity
# import sympy
import time,datetime
import warnings
import math
from tqdm import tqdm
# import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
np.random.seed(1)

# 图像上显示中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']


# 先把所有测试场景的数据组合起来

'''
1. 读取每一个model所有测试场景的位置rmse，统计出每一个model的所有场景的rmse之和
2. 读取每一个model所有测试场景的速度、加速度、航向角，计算出和真实的数据的速度、加速度、航向角的KL散度
3. 遍历每一个model，找到轨迹rmse最小的model编号，速度、加速度、航向角KL散度之和最小的model编号，rmse和速度、加速度、航向角KL散度之和最小的model编号
4. 根据这个最小的model，提取出这个model下所有的测试场景数据，包括期望加速度数据，画出期望加速度的数据分布、速度、加速度、航向角、GT的分布
'''

# 计算点数据的分布
def KDE_density(data):
    sigma_std = np.std(data)
    n_len = len(data)
    h_bandwidth = 1.06 * sigma_std * (n_len) ** (-1/5)
    func_gaussian = KernelDensity(kernel='gaussian', bandwidth=h_bandwidth, algorithm='auto').fit(data)
    x_ax = np.linspace(np.min(data), np.max(data), 5000).reshape(-1,1)
    log_density=func_gaussian.score_samples(x_ax).reshape(-1,1)
    density_results = np.exp(log_density).reshape(-1,1)
    return x_ax,density_results

# KL_value(interaction_inf, expert_data, generate_data, target)
def KL_value(interaction_inf, expert_data, generate_data, target, evaluate_indicator,direction):
    # sns.set_style('darkgrid',{"xtick.major.size":10,"ytick.major.size":10})
    # start=time.time()
    epsilon = 10 ** (-3)
    root = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight\results_evaluate' \
           r'\v13\训练集-评价结果\各model参数分布图\概率密度图\%s' % (target)
    KL_div = list()
    KL_div_dic = []
    KL_expert_list, KL_generate_list = list(), list()
    if evaluate_indicator == 'desried_acc':
        expert_data_value = expert_data[target].dropna().values.reshape(-1, 1)  # 一辆车的某一个参数的数值
        generate_data_value = generate_data[target].dropna().values.reshape(-1, 1)  # 所有generate的某一个参数的数值
        data_parameter_expert = expert_data_value.copy()
        data_parameter_generate = generate_data_value.copy()

        # 计算个体参数分布与整体分布
        x_a0, ds0 = KDE_density(data_parameter_expert)
        x_a1, ds1 = KDE_density(data_parameter_generate)

        # 绘制概率密度图
        # 一辆车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_expert.min()-1,data_parameter_expert.max()+1,int((data_parameter_expert.max()-data_parameter_expert.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a0, ds0)
        plt.title('data_parameter_expert_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        # print('str(interaction_inf) + str(target)):', str(interaction_inf) + '_' + str(target))
        plt.savefig(root + '\%s_expert.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # 所有车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_generate.min()-1,data_parameter_generate.max()+1,int((data_parameter_generate.max()-data_parameter_generate.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a1, ds1)
        plt.title('data_parameter_generate_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        plt.savefig(root + '\%s_generate.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # KL散度计算
        KL_target_parameter = scipy.stats.entropy(ds1, ds0)
        # KL_target_parameter_generate = scipy.stats.entropy(ds0, ds0)
        # print(interaction_inf, 'KL_target_parameter_KL:', KL_target_parameter, KL_target_parameter[0])
        # print(interaction_inf,'KL_target_parameter_generateKL:',KL_target_parameter_generate, KL_target_parameter_generate[0])

        KL_div.append(KL_target_parameter)
        # KL_div.append(KL_target_parameter)

    elif evaluate_indicator == 'regular':
        expert_data_value = expert_data['expert_' + target].dropna().values.reshape(-1, 1)  # 一辆车的某一个参数的数值
        generate_data_value = generate_data['generate_' + target].dropna().values.reshape(-1, 1)  # 所有generate的某一个参数的数值
        # print(expert_data_value)
        # print(generate_data_value)
        data_parameter_expert = expert_data_value.copy()
        data_parameter_generate = generate_data_value.copy()

        # 计算个体参数分布与整体分布
        x_a0, ds0 = KDE_density(data_parameter_expert)
        x_a1, ds1 = KDE_density(data_parameter_generate)

        # 绘制概率密度图
        # 一辆车的目标参数
        fig=plt.figure()
        ax=plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_expert.min()-1,data_parameter_expert.max()+1,int((data_parameter_expert.max()-data_parameter_expert.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a0, ds0)
        plt.title('data_parameter_expert_%s' % (str(interaction_inf) +'_'+str(target)))
        #figpath=os.path.join(root,'3'+target)
        figpath=os.path.join(root, target)
        # print('str(interaction_inf) + str(target)):',str(interaction_inf) +'_'+str(target))
        plt.savefig(root+'\%s_expert.png'%(str(interaction_inf) +'_'+str(target)))

        # 所有车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_generate.min()-1,data_parameter_generate.max()+1,int((data_parameter_generate.max()-data_parameter_generate.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a1, ds1)
        plt.title('data_parameter_generate_%s' % (str(interaction_inf) +'_'+str(target)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        plt.savefig(root+'\%s_generate.png'%(str(interaction_inf) +'_'+str(target)))

        # KL散度计算
        KL_target_parameter = scipy.stats.entropy(ds1, ds0)
        # KL_target_parameter_generate = scipy.stats.entropy(ds0, ds0)
        # print(interaction_inf,'KL_target_parameter_KL:',KL_target_parameter, KL_target_parameter[0])
        # print(interaction_inf,'KL_target_parameter_generateKL:',KL_target_parameter_generate, KL_target_parameter_generate[0])

        KL_div.append(KL_target_parameter)
        # KL_div.append(KL_target_parameter)

    # print('KL_div',KL_div)
    return (KL_div[0])


def jensen_shannon_distance(interaction_inf, expert_data, generate_data, target, evaluate_indicator,direction):
    # 先计算expert_data, generate_data的平均分布
    root = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight\results_evaluate' \
           r'\v13\训练集-评价结果\各model参数分布图\JS概率密度图\%s' % (target)
    KL_div = list()
    KL_div_dic = []
    KL_expert_list, KL_generate_list = list(), list()
    js_distance = None
    if evaluate_indicator == 'desried_acc':
        expert_data_value = expert_data[target+'_y'].dropna().values.reshape(-1, 1)  # 所有expert的某一个参数的数值
        generate_data_value = generate_data[target+'_x'].dropna().values.reshape(-1, 1)  # 所有generate的某一个参数的数值
        data_parameter_expert = expert_data_value.copy()
        data_parameter_generate = generate_data_value.copy()

        # 计算个体参数分布与整体分布
        x_a_expert, ds_expert = KDE_density(data_parameter_expert)
        x_a_generate, ds_generate = KDE_density(data_parameter_generate)
        m = (ds_expert + ds_generate) / 2.0  # 平均分布

        # 绘制概率密度图
        # 一辆车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_expert.min()-1,data_parameter_expert.max()+1,int((data_parameter_expert.max()-data_parameter_expert.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_expert, ds_expert)
        plt.title('data_parameter_expert_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        # print('str(interaction_inf) + str(target)):', str(interaction_inf) + '_' + str(target))
        plt.savefig(root + '\%s_expert.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # 所有车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_generate.min()-1,data_parameter_generate.max()+1,int((data_parameter_generate.max()-data_parameter_generate.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_generate, ds_generate)
        plt.title('data_parameter_generate_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        plt.savefig(root + '\%s_generate.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # KL散度计算
        KL_target_expert_m = scipy.stats.entropy(ds_expert, m)
        KL_target_generate_m = scipy.stats.entropy(ds_generate, m)

        # Calculate JS distance
        js_distance = np.sqrt((KL_target_expert_m + KL_target_generate_m) / 2.0)

    elif evaluate_indicator == 'regular':
        expert_data_value = expert_data['expert_' + target].dropna().values.reshape(-1, 1)  # 一辆车的某一个参数的数值
        generate_data_value = generate_data['generate_' + target].dropna().values.reshape(-1, 1)  # 所有generate的某一个参数的数值
        # print(expert_data_value)
        # print(generate_data_value)
        data_parameter_expert = expert_data_value.copy()
        data_parameter_generate = generate_data_value.copy()

        # 计算个体参数分布与整体分布
        x_a_expert, ds_expert = KDE_density(data_parameter_expert)
        x_a_generate, ds_generate = KDE_density(data_parameter_generate)
        m = (ds_expert + ds_generate) / 2.0  # 平均分布

        # 绘制概率密度图
        # 一辆车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_expert.min()-1,data_parameter_expert.max()+1,int((data_parameter_expert.max()-data_parameter_expert.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_expert, ds_expert)
        plt.title('data_parameter_expert_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        # print('str(interaction_inf) + str(target)):', str(interaction_inf) + '_' + str(target))
        plt.savefig(root + '\%s_expert.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # 所有车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_generate.min()-1,data_parameter_generate.max()+1,int((data_parameter_generate.max()-data_parameter_generate.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_generate, ds_generate)
        plt.title('data_parameter_generate_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        plt.savefig(root + '\%s_generate.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # KL散度计算
        KL_target_expert_m = scipy.stats.entropy(ds_expert, m)
        KL_target_generate_m = scipy.stats.entropy(ds_generate, m)

        # Calculate JS distance
        js_distance = np.sqrt((KL_target_expert_m + KL_target_generate_m) / 2.0)

    elif evaluate_indicator == 'ave_gt':
        expert_data_value = expert_data[target].dropna().values.reshape(-1, 1)  # 一辆车的某一个参数的数值
        generate_data_value = generate_data[target].dropna().values.reshape(-1, 1)  # 所有generate的某一个参数的数值
        # print(expert_data_value)
        # print(generate_data_value)
        data_parameter_expert = expert_data_value.copy()
        data_parameter_generate = generate_data_value.copy()

        # 计算个体参数分布与整体分布
        x_a_expert, ds_expert = KDE_density(data_parameter_expert)
        x_a_generate, ds_generate = KDE_density(data_parameter_generate)
        m = (ds_expert + ds_generate) / 2.0  # 平均分布

        # 绘制概率密度图
        # 一辆车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_expert.min()-1,data_parameter_expert.max()+1,int((data_parameter_expert.max()-data_parameter_expert.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_expert, ds_expert)
        plt.title('data_parameter_expert_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        # print('str(interaction_inf) + str(target)):', str(interaction_inf) + '_' + str(target))
        plt.savefig(root + '\%s_expert.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # 所有车的目标参数
        fig = plt.figure()
        ax = plt.subplot(111)
        # my_x_ticks = np.arange(data_parameter_generate.min()-1,data_parameter_generate.max()+1,int((data_parameter_generate.max()-data_parameter_generate.min())/10))
        # plt.xticks(my_x_ticks)
        ax.plot(x_a_generate, ds_generate)
        plt.title('data_parameter_generate_%s' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))
        # figpath=os.path.join(root,'3'+target)
        figpath = os.path.join(root, target)
        plt.savefig(root + '\%s_generate.png' % (str(interaction_inf) + '_' + str(target) + '_' + str(direction)))

        # KL散度计算
        KL_target_expert_m = scipy.stats.entropy(ds_expert, m)
        KL_target_generate_m = scipy.stats.entropy(ds_generate, m)

        # Calculate JS distance
        js_distance = np.sqrt((KL_target_expert_m + KL_target_generate_m) / 2.0)

    # print('KL_div',KL_div)
    return js_distance


# 画出分布直方图
def Distribution_histogram(datagenerate, dataexpert, feature_name, model_name, root, evaluate_indicator, direction):
    if evaluate_indicator == 'desried_acc':
        data_generate = datagenerate[feature_name+'_x']
        data_expert = dataexpert[feature_name+'_y']
        sns.set_style("white")
        plt.figure(dpi=120)

        sns.distplot(data_expert,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "expert",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='g',
                     label="expert",
                     axlabel=feature_name,  # 设置x轴标题
                     )

        sns.distplot(data_generate,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "generate",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='b',  # #098154
                     label="generate",
                     axlabel=feature_name,  # 设置x轴标题
                     )
        # plt.xlim(-400,200)
        plt.legend()  # 显示中文图例plt.legend(prop=chinese)。如果是英文图例，括号里面不加东西
        plt.savefig(root + '\%s.png' % ('_' + str(model_name) + "_" + str(feature_name) + '_' + str(direction)))
        plt.clf()


    elif evaluate_indicator == 'regular':
        data_generate = datagenerate['generate_' + feature_name]
        data_expert = dataexpert['expert_' + feature_name]
        sns.set_style("white")
        plt.figure(dpi=120)

        sns.distplot(data_expert,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "expert",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='g',
                     label="expert",
                     axlabel=feature_name,  # 设置x轴标题
                     )

        sns.distplot(data_generate,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "generate",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='b',  # #098154
                     label="generate",
                     axlabel=feature_name,  # 设置x轴标题
                     )
        # plt.xlim(-400,200)
        plt.legend()  # 显示中文图例plt.legend(prop=chinese)。如果是英文图例，括号里面不加东西
        plt.savefig(root + '\%s.png' % ('_' + str(model_name) + "_" + str(feature_name)))
        plt.clf()

    elif evaluate_indicator == 'ave_gt':
        data_generate = datagenerate[feature_name]
        data_expert = dataexpert[feature_name]
        sns.set_style("white")
        plt.figure(dpi=120)

        sns.distplot(data_expert,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "expert",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='g',
                     label="expert",
                     axlabel=feature_name,  # 设置x轴标题
                     )

        sns.distplot(data_generate,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'label': "generate",  # 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='b',  # #098154
                     label="generate",
                     axlabel=feature_name,  # 设置x轴标题
                     )
        # plt.xlim(-400,200)
        plt.legend()  # 显示中文图例plt.legend(prop=chinese)。如果是英文图例，括号里面不加东西
        plt.savefig(root + '\%s.png' % ('_' + str(model_name) + "_" + str(feature_name)))
        plt.clf()

    return

# 计算一个场景内左转车和直行车谁先通过冲突点，先通过冲突点的是抢行
# def interaction_type_judge(one_scenario):



def Cal_GT(one_model):  # 输入给模型的是一个model_id的所有场景的数据
    # 先按照场景group，再分别计算同一个group下两辆车的期望加速度，如果过了冲突点，就为None
    all_test_scenario = [name[1] for name in one_model.groupby(['phase_id'])]
    all_scenario_data = pd.DataFrame()  # 存储这个model下所有场景的计算完GT的数据
    for one_scenario in tqdm(all_test_scenario, position=0):
        one_scenario['generate_inter_agent_id'] = None
        one_scenario['generate_inter_agent_x'] = None
        one_scenario['generate_inter_agent_y'] = None
        one_scenario['generate_inter_agent_acc'] = None
        one_scenario['generate_inter_agent_steering'] = None
        one_scenario['generate_inter_agent_v'] = None
        one_scenario['generate_inter_agent_headingnow'] = None
        one_scenario['generate_GT'] = None
        one_scenario['expert_inter_agent_id'] = None
        one_scenario['expert_inter_agent_x'] = None
        one_scenario['expert_inter_agent_y'] = None
        one_scenario['expert_inter_agent_acc'] = None
        one_scenario['expert_inter_agent_steering'] = None
        one_scenario['expert_inter_agent_v'] = None
        one_scenario['expert_inter_agent_headingnow'] = None
        one_scenario['expert_GT'] = None
        one_scenario['Time'] = None
        one_scenario['interaction'] = None
        # 找到每一辆左转车和直行车
        left_vehs = one_scenario[one_scenario['direction'] == 'left']
        straight_vehs = one_scenario[one_scenario['direction'] == 'straight']
        left_vehs.index = range(len(left_vehs))
        straight_vehs.index = range(len(straight_vehs))
        data_type = ['generate', 'expert']
        veh_length = 4.5
        veh_width = 2
        # 下面两个dataframe用来存储有了time属性的这个场景的每一辆左转车和直行车
        vehs_straight_new = pd.DataFrame()
        vehs_left_new = pd.DataFrame()
        # 给每一条左转车和直行车赋予时间time
        every_vehs_left = [name[1] for name in left_vehs.groupby(['agent_id'])]  # 万一有多辆车的时候，也可以用这个评估
        every_vehs_straight = [name[1] for name in straight_vehs.groupby(['agent_id'])]

        # 先给左转车赋予时间time
        for one_veh_left in every_vehs_left:
            one_veh_left.index = range(len(one_veh_left))
            one_veh_left['time'] = range(len(one_veh_left))
            vehs_left_new = pd.concat([vehs_left_new,one_veh_left],axis=0)
        # 再给直行车赋予时间time
        for one_veh_straight in every_vehs_straight:
            one_veh_straight.index = range(len(one_veh_straight))
            one_veh_straight['time'] = range(len(one_veh_straight))
            vehs_straight_new = pd.concat([vehs_straight_new, one_veh_straight],axis=0)

        # 计算完time的这个场景的每一辆左转车和直行车
        every_vehs_left_new = [name[1] for name in vehs_left_new.groupby(['agent_id'])]  # 万一有多辆车的时候，也可以用这个评估
        every_vehs_straight_new = [name[1] for name in vehs_straight_new.groupby(['agent_id'])]

        # 下面两个dataframe是用来存放计算完GT的这个场景的每一辆左转车和直行车的数据
        vehs_left_GT = pd.DataFrame()
        vehs_straight_GT = pd.DataFrame()

        # 给左转车计算TTC
        for one_veh_left in every_vehs_left_new:
            for type in data_type:
                for i in range(len(one_veh_left)):
                    pos_x_i = one_veh_left[type+'_x'][i]
                    pos_y_i = one_veh_left[type+'_y'][i]
                    heading_angle_last1_i = one_veh_left[type+'_angle'][i]  # -90~180
                    if pos_x_i != -4:
                        time_i = one_veh_left['time'][i]
                        # 这里只考虑还没到交点时的车辆的TTC
                        potential_interaction_vehs = vehs_straight_new[(abs(vehs_straight_new['time'] - time_i) == 0)
                                                                             &(vehs_straight_new[type+'_x']>=pos_x_i)]  # 用于存放agent k时刻的潜在交互车辆k时刻的数据  # 时刻相同则是和agent同时出现的车辆。125代表0.125s

                        # 判断这辆交互车是否在agent的视野前方，且距离小于20m
                        potential_interaction_vehs.index = range(len(potential_interaction_vehs))
                        if len(potential_interaction_vehs) > 0:
                            interaction_vehs = pd.DataFrame()
                            for jj in range(len(potential_interaction_vehs)):
                                a = np.array([potential_interaction_vehs[type+'_x'][jj] - pos_x_i,
                                              potential_interaction_vehs[type+'_y'][jj] - pos_y_i])
                                b = np.zeros(2)
                                if 0 <= heading_angle_last1_i < 90:  # tan>0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == 90:
                                    b = np.array([0, 2])
                                elif 90 < heading_angle_last1_i <= 180:  # tan<0
                                    b = np.array([-1, -1 * math.tan(np.radians(heading_angle_last1_i))])
                                elif 180 < heading_angle_last1_i < 270:  # tan>0
                                    b = np.array([-1, -1 * math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == 270:  # 负无穷
                                    b = np.array([0, -2])
                                elif 270 < heading_angle_last1_i <= 360:  # tan<0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif -90 < heading_angle_last1_i < 0:  # tan<0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == -90:
                                    b = np.array([0, -2])

                                Lb = np.sqrt(b.dot(b))
                                La = np.sqrt(a.dot(a))

                                cos_angle = np.dot(a, b) / (La * Lb)
                                cross = np.cross((b[0], b[1]),
                                                 (potential_interaction_vehs[type+'_x'][jj] - pos_x_i,
                                                  potential_interaction_vehs[type+'_y'][jj] - pos_y_i))
                                angle_hudu = np.arccos(cos_angle)
                                angle_jiaodu = angle_hudu * 360 / 2 / np.pi

                                dis = np.sqrt((potential_interaction_vehs[type+'_x'][jj] - pos_x_i) ** 2
                                              + (potential_interaction_vehs[type+'_y'][jj] - pos_y_i) ** 2)

                                if (angle_jiaodu >= 0) and (angle_jiaodu <= 90) and dis <= 20:
                                    one = potential_interaction_vehs[potential_interaction_vehs.index == jj]
                                    interaction_vehs = pd.concat([interaction_vehs, one], axis=0)

                                if len(interaction_vehs) > 0:
                                    # 有这个交互对象
                                    one_veh_left[type+'_inter_agent_id'][i] = interaction_vehs['agent_id']
                                    one_veh_left[type+'_inter_agent_x'][i] = interaction_vehs[type+'_x']
                                    one_veh_left[type+'_inter_agent_y'][i] = interaction_vehs[type+'_y']
                                    one_veh_left[type+'_inter_agent_acc'][i] = interaction_vehs[type+'_acc']
                                    one_veh_left[type+'_inter_agent_steering'][i] = interaction_vehs[type+'_yaw']
                                    one_veh_left[type+'_inter_agent_v'][i] = interaction_vehs[type+'_v']
                                    one_veh_left[type+'_inter_agent_headingnow'][i] = interaction_vehs[type+'_angle_now']

                                    # 计算和这个车辆的GT
                                    left_x = pos_x_i
                                    left_y = pos_y_i
                                    straight_x = one_veh_left[type+'_inter_agent_x'].iloc[i].item()
                                    straight_y = one_veh_left[type+'_inter_agent_y'].iloc[i].item()
                                    left_heading_now = one_veh_left[type+'_angle_now'].iloc[i].item()  # 角度
                                    straight_heading_now = one_veh_left[type+'_inter_agent_headingnow'].iloc[i].item()  # 角度
                                    left_v = one_veh_left[type+'_v'].iloc[i].item()
                                    straight_v = one_veh_left[type+'_inter_agent_v'].iloc[i].item()

                                    # 两辆车的k，斜率
                                    a_left = math.tan(np.radians(left_heading_now))  # 斜率a_left
                                    a_straight = math.tan(np.radians(straight_heading_now))  # 斜率a_straight

                                    # 两辆车的b
                                    b_left = (left_y) - a_left * (left_x)
                                    # print('straight_y:',straight_y,'a_straight:',a_straight,'straight_x:',straight_x)
                                    b_straight = (straight_y) - a_straight * (straight_x)
                                    # print('a_left:',a_left,'a_straight:',a_straight,'b_left:',b_left,'b_straight:',b_straight)

                                    # 两车的交点
                                    # 计算两直线的交点
                                    jiaodianx = None
                                    jiaodiany = None
                                    if a_left == a_straight:
                                        continue
                                    else:
                                        jiaodianx = (b_straight - b_left) / (a_left - a_straight)  # 真实的坐标
                                        jiaodiany = a_left * jiaodianx + b_left
                                        # print('jiaodianx_expert:', jiaodianx_expert, 'straight_x_expert', straight_x_expert,
                                        #       'jiaodiany_expert', jiaodiany_expert, 'left_y_expert:', left_y_expert)

                                        # print('jiaodianx:',jiaodianx.values,'straight_x:',straight_x.values)
                                        if jiaodianx > straight_x + 0.5 * veh_length:
                                        # 两车不会相撞
                                            GT_value = None
                                            # print('这两辆车不会相撞')
                                        elif straight_x < jiaodianx <= straight_x + 0.5 * veh_length:
                                            agent_dis = np.sqrt((pos_x_i - jiaodianx) ** 2 + (pos_y_i - jiaodiany) ** 2)
                                            inter_straight_dis = 0.5 * veh_length - np.sqrt((straight_x - jiaodianx) ** 2 + (straight_y - jiaodiany) ** 2)

                                            # 当前agent先通过交叉口
                                            time_A = abs(
                                                (inter_straight_dis - 0.5 * veh_width) / straight_v - (agent_dis + 0.5 * veh_length + 0.5*veh_width) / left_v)

                                            # 左前方交互车辆先通过冲突区
                                            time_B = abs(
                                                (agent_dis - 0.5 * veh_length - 0.5 * veh_width) / left_v - (inter_straight_dis + 0.5 * veh_width) / straight_v)

                                            GT_value = min(time_A, time_B)
                                            one_veh_left[type+'_GT'][i] = GT_value

                                        else:
                                            agent_dis = np.sqrt((left_x - jiaodianx) ** 2 + (left_y - jiaodiany) ** 2)

                                            inter_straight_dis = np.sqrt(
                                                (straight_x - jiaodianx) ** 2 + (straight_y - jiaodiany) ** 2)

                                            # 当前agent先通过交叉口
                                            time_A = abs(
                                                (inter_straight_dis - 0.5 * veh_length - 0.5 *veh_width) / straight_v - (agent_dis + 0.5 * veh_length + 0.5 *veh_width) / left_v)

                                            # 左前方交互车辆先通过冲突区
                                            time_B = abs(
                                                (agent_dis - 0.5 * veh_length - 0.5 *veh_width) / left_v - (inter_straight_dis + 0.5 * veh_length + 0.5 *veh_width) / straight_v)

                                            GT_value = min(time_A, time_B)
                                            one_veh_left[type+'_GT'][i] = GT_value

            vehs_left_GT = pd.concat([vehs_left_GT,one_veh_left],axis=0)
        # 给直行车计算TTC
        for one_veh_straight in every_vehs_straight_new:
            for type in data_type:
                for i in range(len(one_veh_straight)):
                    pos_x_i = one_veh_straight[type + '_x'][i]
                    pos_y_i = one_veh_straight[type + '_y'][i]
                    heading_angle_last1_i = one_veh_straight[type + '_angle'][i]  # -90~180
                    if pos_x_i != -4:
                        time_i = one_veh_straight['time'][i]
                        # 这里只考虑还没到交点时的车辆的TTC
                        potential_interaction_vehs = vehs_left_new[
                            (abs(vehs_left_new['time'] - time_i) == 0)
                            & (vehs_left_new[
                                   type + '_y'] <= pos_y_i)]  # 用于存放agent k时刻的潜在交互车辆k时刻的数据  # 时刻相同则是和agent同时出现的车辆。125代表0.125s

                        # 判断这辆交互车是否在agent的视野前方，且距离小于20m
                        potential_interaction_vehs.index = range(len(potential_interaction_vehs))
                        if len(potential_interaction_vehs) > 0:
                            interaction_vehs = pd.DataFrame()
                            for jj in range(len(potential_interaction_vehs)):
                                a = np.array([potential_interaction_vehs[type + '_x'][jj] - pos_x_i,
                                              potential_interaction_vehs[type + '_y'][jj] - pos_y_i])
                                b = np.zeros(2)
                                if 0 <= heading_angle_last1_i < 90:  # tan>0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == 90:
                                    b = np.array([0, 2])
                                elif 90 < heading_angle_last1_i <= 180:  # tan<0
                                    b = np.array([-1, -1 * math.tan(np.radians(heading_angle_last1_i))])
                                elif 180 < heading_angle_last1_i < 270:  # tan>0
                                    b = np.array([-1, -1 * math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == 270:  # 负无穷
                                    b = np.array([0, -2])
                                elif 270 < heading_angle_last1_i <= 360:  # tan<0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif -90 < heading_angle_last1_i < 0:  # tan<0
                                    b = np.array([1, math.tan(np.radians(heading_angle_last1_i))])
                                elif heading_angle_last1_i == -90:
                                    b = np.array([0, -2])

                                Lb = np.sqrt(b.dot(b))
                                La = np.sqrt(a.dot(a))

                                cos_angle = np.dot(a, b) / (La * Lb)
                                cross = np.cross((b[0], b[1]),
                                                 (potential_interaction_vehs[type + '_x'][jj] - pos_x_i,
                                                  potential_interaction_vehs[type + '_y'][jj] - pos_y_i))
                                angle_hudu = np.arccos(cos_angle)
                                angle_jiaodu = angle_hudu * 360 / 2 / np.pi

                                dis = np.sqrt((potential_interaction_vehs[type + '_x'][jj] - pos_x_i) ** 2
                                              + (potential_interaction_vehs[type + '_y'][jj] - pos_y_i) ** 2)

                                if (angle_jiaodu >= 0) and (angle_jiaodu <= 90) and dis <= 20:
                                    one = potential_interaction_vehs[potential_interaction_vehs.index == jj]
                                    interaction_vehs = pd.concat([interaction_vehs, one], axis=0)

                                if len(interaction_vehs) > 0:
                                    # 有这个交互对象
                                    one_veh_straight[type + '_inter_agent_id'][i] = interaction_vehs['agent_id']
                                    one_veh_straight[type + '_inter_agent_x'][i] = interaction_vehs[type + '_x']
                                    one_veh_straight[type + '_inter_agent_y'][i] = interaction_vehs[type + '_y']
                                    one_veh_straight[type + '_inter_agent_acc'][i] = interaction_vehs[type + '_acc']
                                    one_veh_straight[type + '_inter_agent_steering'][i] = interaction_vehs[
                                        type + '_yaw']
                                    one_veh_straight[type + '_inter_agent_v'][i] = interaction_vehs[type + '_v']
                                    one_veh_straight[type + '_inter_agent_headingnow'][i] = interaction_vehs[
                                        type + '_angle_now']

                                    # 计算和这个车辆的GT
                                    straight_x = pos_x_i
                                    straight_y = pos_y_i
                                    try:
                                        left_x = one_veh_straight[type + '_inter_agent_x'][i].item()
                                    except:
                                        left_x = one_veh_straight[type + '_inter_agent_x'][i]
                                        # print('left_x',left_x)
                                    try:
                                        left_y = one_veh_straight[type + '_inter_agent_y'][i].item()
                                    except:
                                        left_y = one_veh_straight[type + '_inter_agent_y'][i]
                                    try:
                                        straight_heading_now = one_veh_straight[type + '_angle_now'][i].item()  # 角度
                                    except:
                                        straight_heading_now = one_veh_straight[type + '_angle_now'][i]
                                    try:
                                        left_heading_now = one_veh_straight[type + '_inter_agent_headingnow'][i].item()  # 角度
                                    except:
                                        left_heading_now = one_veh_straight[type + '_inter_agent_headingnow'][i]
                                    try:
                                        straight_v = one_veh_straight[type + '_v'][i].item()
                                    except:
                                        straight_v = one_veh_straight[type + '_v'][i]
                                    try:
                                        left_v = one_veh_straight[type + '_inter_agent_v'][i].item()
                                    except:

                                        left_v = one_veh_straight[type + '_inter_agent_v'][i]

                                    # 两辆车的k，斜率
                                    a_straight = math.tan(np.radians(straight_heading_now))  # 斜率a_straight
                                    a_left = math.tan(np.radians(left_heading_now))  # 斜率a_left

                                    # 两辆车的b
                                    b_straight = (straight_y) - a_straight * (straight_x)
                                    b_left = (left_y) - a_left * (left_x)

                                    # 两车的交点
                                    # 计算两直线的交点
                                    jiaodianx = None
                                    jiaodiany = None
                                    if a_straight == a_left:
                                        continue
                                    else:
                                        jiaodianx = (b_left - b_straight) / (a_straight - a_left)  # 真实的坐标
                                        jiaodiany = a_straight * jiaodianx + b_straight
                                        # print('jiaodianx_expert:', jiaodianx_expert, 'left_x_expert', left_x_expert,
                                        #       'jiaodiany_expert', jiaodiany_expert, 'straight_y_expert:', straight_y_expert)

                                        if jiaodiany < left_y - 0.5 * veh_length:
                                            # 两车不会相撞
                                            GT_value = None
                                            # print('这辆车是直行车，左前方车辆的交互, 有，但是不会交互情况5')
                                        elif left_y - 0.5 * veh_length <= jiaodiany < left_y:
                                            # print('这辆车是直行车，左前方车辆的交互, 有，会交互情况6')
                                            agent_dis = np.sqrt(
                                                (straight_x - jiaodianx) ** 2 + (straight_y - jiaodiany) ** 2)

                                            inter_left_dis = 0.5 * veh_length - np.sqrt(
                                                (left_x - jiaodianx) ** 2 + (left_y - jiaodiany) ** 2)

                                            # 当前agent先通过交叉口
                                            time_A = abs(
                                                (inter_left_dis - 0.5 * veh_width) / left_v - (agent_dis + 0.5 * veh_length + 0.5*veh_width) / straight_v)

                                            # 左前方交互车辆先通过冲突区
                                            time_B = abs(
                                                (agent_dis - 0.5 * veh_length - 0.5 *veh_width) / straight_v - (inter_left_dis + 0.5 * veh_width) / left_v)

                                            GT_value = min(time_A, time_B)
                                            one_veh_straight[type+'_GT'][i] = GT_value

                                        else:
                                            # print('这辆车是直行车，左前方车辆的交互, 有，会交互情况7')
                                            agent_dis = np.sqrt((straight_x - jiaodianx) ** 2 + (straight_y - jiaodiany) ** 2)

                                            left_dis = np.sqrt((left_x - jiaodianx) ** 2 + (left_y - jiaodiany) ** 2)

                                            # 当前agent先通过交叉口
                                            time_A = abs(
                                                (left_dis - 0.5 * veh_length - 0.5 *veh_width) / left_v - (agent_dis + 0.5 * veh_length + 0.5*veh_width) / straight_v)

                                            # 左前方交互车辆先通过冲突区
                                            time_B = abs(
                                                (agent_dis - 0.5 * veh_length - 0.5 *veh_width) / straight_v - (left_dis + 0.5 * veh_length + 0.5*veh_width) / left_v)

                                            GT_value = min(time_A, time_B)
                                            one_veh_straight[type+'_GT'][i] = GT_value

            vehs_straight_GT = pd.concat([vehs_straight_GT, one_veh_straight], axis=0)

        # 存放计算完GT的这个场景的所有轨迹的数据
        one_scenario_new = pd.concat([vehs_left_GT,vehs_straight_GT],axis=0)
        # 判断左转车辆是让行还是抢行
        all_scenario_data = pd.concat([all_scenario_data,one_scenario_new],axis=0)
    return all_scenario_data  # 范围新的这个场景的数据



if __name__ == '__main__':
    '''
    step1 读取每一个model所有测试场景的位置rmse，统计出每一个model的所有场景的rmse之和
    '''
    filedir_pos_rmse = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                       r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\轨迹位置的rmse\\'  # 轨迹位置的rmse

    file_namelist_pos_rmse = os.listdir(filedir_pos_rmse)

    root_pos_rmse = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                    r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\各model所有test场景评估参数折线图\POS_RMSE'  # 存储位置

    # 主程序
    all_interaction_scenario_save = pd.DataFrame()  # 用于存放文件内所有场景的数据（有超车、跟驰字段信息）
    result_df = pd.DataFrame()  # 用于存放每一个model下所有场景的评估结果

    for filename in tqdm(file_namelist_pos_rmse, position=0):
        filepath = filedir_pos_rmse + filename
        current_file_all_trj = pd.read_csv(filepath) # 一个场景的数据

        scenario_id = str(filename[:3])
        current_file_all_trj['phase_id'] = scenario_id
        # print('每一个场景的数据：',current_file_all_trj.columns,current_file_all_trj)
        all_interaction_scenario_save = pd.concat([all_interaction_scenario_save,current_file_all_trj],axis=0)

    all_interaction_scenario_save.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                         r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果'
                                         r'\all_interaction_scenario_save.csv')

    # 对每一个model下所有的场景的pos rmse相加
    column_sum = all_interaction_scenario_save.sum()
    # 计算每一列的均值
    column_mean = all_interaction_scenario_save.mean()
    # 将和和均值添加到 DataFrame 的最后两行
    all_interaction_scenario_save.loc['Sum'] = column_sum
    all_interaction_scenario_save.loc['Mean'] = column_mean

    # 创建新的 DataFrame，每一列是每个 model，只保留和和均值两行
    # print('all_interaction_scenario_save:',all_interaction_scenario_save.columns,all_interaction_scenario_save)
    result_df = pd.DataFrame(columns=all_interaction_scenario_save.columns)

    result_df.loc['pos_rmse_sum'] = column_sum
    result_df.loc['pos_rmse_mean'] = column_mean
    # result_df.columns = all_interaction_scenario_save.columns

    print('result_df:', result_df.columns, result_df)

    # 画折线图
    result_df.loc['pos_rmse_sum', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('Position RMSE Sum by Model')
    plt.xlabel('Model')
    plt.ylabel('Position RMSE Sum')
    plt.savefig(root_pos_rmse+'\_Position RMSE Sum by Model.png')
    plt.close()

    # 画折线图
    result_df.loc['pos_rmse_mean', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('Position RMSE Mean by Model')
    plt.xlabel('Model')
    plt.ylabel('Position RMSE Mean')
    plt.savefig(root_pos_rmse + '\_Position RMSE Mean by Model.png')
    plt.close()

    '''
    step2.2 读取每一个model所有测试场景的desired_acc，计算出和真实的数据的desired_acc的KL散度，并统计出谁先抢行
    '''
    filedir_desried_acc = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                          r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\专家和生成轨迹desried_acc\\'  # 用于做分布图和计算KL散度的数据

    file_namelist_desried_acc = os.listdir(filedir_desried_acc)

    root_desried_acc = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                       r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\各model所有test场景评估参数折线图'  # 存储位置

    # 主程序
    all_desried_acc_scenario_save = pd.DataFrame()  # 用于存放文件内所有场景的数据（有超车、跟驰字段信息）

    for filename in tqdm(file_namelist_desried_acc, position=0):
        filepath = filedir_desried_acc + filename
        current_file_all_trj_desried_acc = pd.read_csv(filepath)  # 一个场景的数据

        scenario_id = str(filename[:3])
        current_file_all_trj_desried_acc['phase_id'] = scenario_id

        all_desried_acc_scenario_save = pd.concat([all_desried_acc_scenario_save, current_file_all_trj_desried_acc],
                                                  axis=0)

    all_desried_acc_scenario_save.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                         r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果'
                                         r'\all_desried_acc_scenario_save.csv')


    # 按照model_id分组数据
    # all_desried_acc_scenario_save.rename(columns={'dy/dt': 'dydt'}, inplace=True)
    all_model = [name[1] for name in all_desried_acc_scenario_save.groupby(['model_id'])]
    all_desried_acc_scenario_save_new = pd.DataFrame()  # 存放计算完谁抢行‘interaction’的数据
    no_jiaodian_inf = []  # 存放每个model下每个场景的专家和生成轨迹的左转车和直行车无交点的信息，model_id，scenario_id，类型（是expert没有，还是generate没有，还是both都没有）
    for one_model in tqdm(all_model, position=0):
        one_model.index = range(len(one_model))
        one_model_inf = 'model_' + str(int(one_model['model_id'].iloc[0].item()))
        # print('one_model_inf:',one_model_inf)

        one_model_generate0 = one_model[one_model['type'] == 'generate']
        one_model_generate = one_model_generate0[(one_model_generate0['generate_x'] != -4)]
        one_model_generate.index = range(len(one_model_generate))
        one_model_expert0 = one_model[one_model['type'] == 'expert']
        one_model_expert = one_model_expert0[(one_model_expert0['expert_x'] != -4)]
        one_model_expert.index = range(len(one_model_expert))
        # interaction_inf = one_model_generate['model_id'][0]

        # 动态交互参数dydt等 如果generate和expert的下面两个参数相同，时间点记录下来，统计完全相同的比例
        # left_dongtai_id  right_dongtai_id
        # 提取出left_id不为None，但是right_id为None的行
        filtered_generate_left = one_model_generate[one_model_generate['left_dongtai_id'].notna() & one_model_generate['right_dongtai_id'].isna()]
        filtered_expert_left = one_model_expert[one_model_expert['left_dongtai_id'].notna() & one_model_expert['right_dongtai_id'].isna()]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数

        common_rows_left = pd.merge(filtered_generate_left, filtered_expert_left,
                                    on=['agent_id', 'time_ms','left_dongtai_id','direction','phase_id'], how='inner')
        print('filtered_generate_left', filtered_generate_left, 'filtered_expert_left', filtered_expert_left,
              'common_rows_left', common_rows_left)
        # 计算相同的行数除以dataframe a提取出来的行数
        # common_ratio_left = len(common_rows_left) / len(filtered_expert_left)

        # 提取出left_id不为None，right_id也不为None的行
        filtered_generate_left_right = one_model_generate[one_model_generate['left_dongtai_id'].notna() & one_model_generate['right_dongtai_id'].notna()]
        filtered_expert_left_right = one_model_expert[one_model_expert['left_dongtai_id'].notna() & one_model_expert['right_dongtai_id'].notna()]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_left_right = pd.merge(filtered_generate_left_right, filtered_expert_left_right, on=['agent_id', 'time_ms','left_dongtai_id','right_dongtai_id','direction','phase_id'], how='inner')
        # 计算相同的行数除以dataframe a提取出来的行数
        # common_ratio_left_right = len(common_rows_left_right) / len(filtered_expert_left_right)

        # 提取出left_id为None，right_id也为None的行
        filtered_generate_ = one_model_generate[
            one_model_generate['left_dongtai_id'].isna() & one_model_generate['right_dongtai_id'].isna()]
        filtered_expert_ = one_model_expert[
            one_model_expert['left_dongtai_id'].isna() & one_model_expert['right_dongtai_id'].isna()]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_ = pd.merge(filtered_generate_, filtered_expert_,
                                          on=['agent_id', 'time_ms', 'direction','phase_id'], how='inner')
        # 计算相同的行数除以dataframe a提取出来的行数
        # common_ratio_ = len(common_rows_) / len(filtered_expert_)

        # 提取出left_id为None，right_id不为None的行
        filtered_generate_right = one_model_generate[
            one_model_generate['left_dongtai_id'].isna() & one_model_generate['right_dongtai_id'].notna()]
        filtered_expert_right = one_model_expert[
            one_model_expert['left_dongtai_id'].isna() & one_model_expert['right_dongtai_id'].notna()]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_right = pd.merge(filtered_generate_right, filtered_expert_right,
                                on=['agent_id', 'time_ms','right_dongtai_id','direction','phase_id'], how='inner')
        # 计算相同的行数除以dataframe a提取出来的行数
        # common_ratio_right = len(common_rows_right) / len(filtered_expert_right)

        same_row = len(common_rows_left) + len(common_rows_left_right) + len(common_rows_) + len(common_rows_right)
        all_expert_row = len(filtered_expert_left) + len(filtered_expert_left_right) + len(filtered_expert_) \
                         + len(filtered_expert_right)

        dongtai_id_sameratio = same_row/all_expert_row
        print('common_rows_left:', len(common_rows_left), len(filtered_expert_left) )
        print('common_rows_left_right:', len(common_rows_left_right), len(filtered_expert_left_right))
        print('common_rows_:', len(common_rows_), len(filtered_expert_))
        print('common_rows_right:', len(common_rows_right), len(filtered_expert_right))

        result_df.loc['dongtai_id_sameratio', one_model_inf] = dongtai_id_sameratio

        # 对上述提取出来的相同的行，计算JS距离
        js_data_df = pd.concat([common_rows_left,common_rows_left_right],axis=0)
        js_data_df = pd.concat([js_data_df, common_rows_], axis=0)
        js_data_df = pd.concat([js_data_df, common_rows_right], axis=0)


        # 合并之后，因为generate和expert的列名相同，所以合并公式的第一个dataframe的列名+'_x'(generate)，第二个dataframe的列名+'_y'(expert)
        target_parameter = ['left_ad_rush', 'left_ad_yield', 'left_ad_rush_delta', 'left_ad_yield_delta', 'left_dydt',
                            'right_ad_rush', 'right_ad_yield', 'right_ad_rush_delta', 'right_ad_yield_delta', 'right_dydt']

        # print('js_data_df:',js_data_df.columns)
        for j in range(len(target_parameter)):
            target = target_parameter[j]
            # print(expert_data[target],DATA_POINT_generate_flying_useful[target])
            # 分别计算左转车和直行车 expert和generate 相关desired_acc的KL散度
            expert_data_left = js_data_df[js_data_df['direction'] == 'left']
            generate_data_left = js_data_df[js_data_df['direction'] == 'left']

            expert_data_straight = js_data_df[js_data_df['direction'] == 'straight']
            generate_data_straight = js_data_df[js_data_df['direction'] == 'straight']

            expert_data_one_pos_left = expert_data_left[[target+'_y']]
            generate_data_one_pos_left = generate_data_left[[target+'_x']]

            expert_data_one_pos_straight = expert_data_straight[[target+'_y']]
            generate_data_one_pos_straight = generate_data_straight[[target+'_x']]

            if target != 'dydt':
                expert_data_one_pos_left2 = expert_data_one_pos_left[abs(expert_data_one_pos_left[target+'_y'])<=25]
                generate_data_one_pos_left2 = generate_data_one_pos_left[abs(generate_data_one_pos_left[target+'_x'])<=25]

                expert_data_one_pos_straight2 = expert_data_one_pos_straight[abs(expert_data_one_pos_straight[target+'_y'])<=25]
                generate_data_one_pos_straight2 = generate_data_one_pos_straight[abs(generate_data_one_pos_straight[target+'_x'])<=25]

            else:
                expert_data_one_pos_left2 = expert_data_one_pos_left
                generate_data_one_pos_left2 = generate_data_one_pos_left

                expert_data_one_pos_straight2 = expert_data_one_pos_straight
                generate_data_one_pos_straight2 = generate_data_one_pos_straight
            # expert_data_one_pos_new = expert_data_one_pos[expert_data_one_pos['expert_x'] != -4]
            # generate_data_one_pos_new = generate_data_one_pos[generate_data_one_pos['generate_x'] != -4]

            evaluate_indicator = 'desried_acc'
            if expert_data_one_pos_left2[target+'_y'].nunique() >= 2 and generate_data_one_pos_left2[target+'_x'].nunique() >= 2:
                print('expert_data_one_pos_left:', expert_data_one_pos_left2, 'generate_data_one_pos_left:',
                      generate_data_one_pos_left2)
                direction = 'left'

                jensen_shannon_target_parameter_left = jensen_shannon_distance(one_model_inf, expert_data_one_pos_left2, generate_data_one_pos_left2,
                                                    target, evaluate_indicator, direction)  # 直接得到两个分布的KL散度的差值

                result_df.loc[target + '_jensen_shannon_left', one_model_inf] = jensen_shannon_target_parameter_left
            else:
                result_df.loc[target + '_jensen_shannon_left', one_model_inf] = None
            if expert_data_one_pos_straight2[target+'_y'].nunique() >=2 and generate_data_one_pos_straight2[target+'_x'].nunique() >= 2:
                direction = 'straight'
                # print('expert_data_one_pos_straight2:', expert_data_one_pos_straight2, 'generate_data_one_pos_straight2:',
                #       generate_data_one_pos_straight2)
                jensen_shannon_target_parameter_straight = jensen_shannon_distance(one_model_inf, expert_data_one_pos_straight2,generate_data_one_pos_straight2, target, evaluate_indicator, direction)  # 直接得到两个分布的KL散度的差值

                result_df.loc[target + '_jensen_shannon_straight', one_model_inf] = jensen_shannon_target_parameter_straight
            else:
                result_df.loc[target + '_jensen_shannon_straight', one_model_inf] = None

            # 画出每个模型model所有场景的acc和steering的分布图
            root = r"D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs" \
                   r"\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\各model参数分布图\柱状图\\%s" % (target)

            if expert_data_one_pos_left2[target+'_y'].nunique() >= 2 and generate_data_one_pos_left2[target+'_x'].nunique() >= 2:
                direction = 'left'
                Distribution_histogram(generate_data_one_pos_left2, expert_data_one_pos_left2, target, one_model_inf,
                                       root, evaluate_indicator,direction)
            if expert_data_one_pos_straight2[target+'_y'].nunique() >= 2 and generate_data_one_pos_straight2[target+'_x'].nunique() >= 2:
                direction = 'straight'
                Distribution_histogram(generate_data_one_pos_straight2, expert_data_one_pos_straight2, target,
                                       one_model_inf, root, evaluate_indicator,direction)

        # 判断这个model内每个场景的每辆车每个轨迹点dongtai交互对象相同的车辆通过顺序（考虑到每个点对应的交互对象不一定一样，所以逐行处理）
        js_data_df_new = js_data_df.dropna(subset=['left_dongtai_id', 'right_dongtai_id']) # 把左右dongtai_id都没有的行删除
        all_trj_one_model = [name[1] for name in js_data_df_new.groupby(['phase_id','agent_id','direction'])]
        all_trj_one_model_new = pd.DataFrame()  # 存放计算完谁先行‘interaction’的数据

        len_model = len(js_data_df_new)  # 这个modle中所有场景的generate和expert left_neig和right_neig交互对象相同的时刻
        same_order = 0 # 记录每个轨迹点代表的车辆通行顺序
        for one_trj in all_trj_one_model:
            one_trj.index = range(len(one_trj))
            one_model_one_scenario_inf = one_trj['phase_id'].iloc[0]
            one_model_one_scenario_one_trj_inf = one_trj['agent_id'].iloc[0]

            for i in range(len(one_trj)):  # 遍历每一个轨迹点
                # 先处理generate
                # 左侧的neig和agent谁先通过冲突点
                one = one_trj[one_trj.index == i]
                one.index = range(len(one))
                one['generate_interaction_leftneig'] = None
                one['generate_interaction_rightneig'] = None
                one['expert_interaction_leftneig'] = None
                one['expert_interaction_rightneig'] = None
                if pd.isnull(one['left'+'_jiaodian_time_ms'+'_x'].iloc[0]) == False:
                    generate_agent_jiaodian_time_ms_leftneig = one['left'+'_jiaodian_time_ms'+'_x'].iloc[0].item()
                    generate_neig_jiaodian_time_ms_leftneig = one['left'+'_jiaodian_time_ms_neig'+'_x'].iloc[0].item()

                    if generate_agent_jiaodian_time_ms_leftneig < generate_neig_jiaodian_time_ms_leftneig:
                        # agent先通过冲突点
                        one['generate_interaction_leftneig'] = str(one_model_one_scenario_one_trj_inf) + '_' + str(one['left_dongtai_id'].iloc[0])
                    elif generate_agent_jiaodian_time_ms_leftneig > generate_neig_jiaodian_time_ms_leftneig:
                        # neig先通过冲突点
                        one['generate_interaction_leftneig'] = str(one['left_dongtai_id'].iloc[0]) + '_' + str(one_model_one_scenario_one_trj_inf)
                    else:
                        one['generate_interaction_leftneig'] = 'same_go'

                    # 再处理expert
                    # 左侧的neig和agent谁先通过冲突点
                    expert_agent_jiaodian_time_ms_leftneig = one['left' + '_jiaodian_time_ms' + '_y'].iloc[0].item()
                    expert_neig_jiaodian_time_ms_leftneig = one['left' + '_jiaodian_time_ms_neig' + '_y'].iloc[
                        0].item()
                    if expert_agent_jiaodian_time_ms_leftneig < expert_neig_jiaodian_time_ms_leftneig:
                        # agent先通过冲突点
                        one['expert_interaction_leftneig'] = str(one_model_one_scenario_one_trj_inf) + '_' + \
                                                             str(one['left_dongtai_id'].iloc[0])
                    elif expert_agent_jiaodian_time_ms_leftneig > expert_neig_jiaodian_time_ms_leftneig:
                        # neig先通过冲突点
                        one['expert_interaction_leftneig'] = str(one['left_dongtai_id'].iloc[
                                                                 0]) + '_' + str(one_model_one_scenario_one_trj_inf)
                    else:
                        one['expert_interaction_leftneig'] = 'same_go'

                if pd.isnull(one['right' + '_jiaodian_time_ms' + '_x'].iloc[0]) == False:
                    # 右侧的neig和agent谁先通过冲突点
                    # print('one:',one['right' + '_jiaodian_time_ms' + '_x'])
                    generate_agent_jiaodian_time_ms_rightneig = one['right' + '_jiaodian_time_ms' + '_x'].iloc[0].item()
                    generate_neig_jiaodian_time_ms_rightneig = one['right' + '_jiaodian_time_ms_neig' + '_x'].iloc[0].item()
                    if generate_agent_jiaodian_time_ms_rightneig < generate_neig_jiaodian_time_ms_rightneig:
                        # agent先通过冲突点
                        one['generate_interaction_rightneig'] = str(one_model_one_scenario_one_trj_inf) + '_' + \
                                                      str(one['right_dongtai_id'].iloc[0])
                    elif generate_agent_jiaodian_time_ms_rightneig > generate_neig_jiaodian_time_ms_rightneig:
                        # neig先通过冲突点
                        one['generate_interaction_rightneig'] = str(one['right_dongtai_id'].iloc[
                                                          0]) + '_' + str(one_model_one_scenario_one_trj_inf)
                    else:
                        one['generate_interaction_rightneig'] = 'same_go'

                    # 右侧的neig和agent谁先通过冲突点
                    expert_agent_jiaodian_time_ms_rightneig = one['right' + '_jiaodian_time_ms' + '_x'].iloc[0].item()
                    expert_neig_jiaodian_time_ms_rightneig = one['right' + '_jiaodian_time_ms_neig' + '_x'].iloc[0].item()
                    if expert_agent_jiaodian_time_ms_rightneig < expert_neig_jiaodian_time_ms_rightneig:
                        # agent先通过冲突点
                        one['expert_interaction_rightneig'] = str(one_model_one_scenario_one_trj_inf) + '_' + \
                                                      str(one['right_dongtai_id'].iloc[0])
                    elif expert_agent_jiaodian_time_ms_rightneig > expert_neig_jiaodian_time_ms_rightneig:
                        # neig先通过冲突点
                        one['expert_interaction_rightneig'] = str(one['right_dongtai_id'].iloc[
                                                          0]) + '_' + str(one_model_one_scenario_one_trj_inf)
                    else:
                        one['expert_interaction_rightneig'] = 'same_go'


                if one['generate_interaction_leftneig'][0] == one['generate_interaction_leftneig'][0] and one['generate_interaction_rightneig'][0] == one['generate_interaction_rightneig'][0]:
                    same_order = same_order + 1

                all_desried_acc_scenario_save_new = pd.concat([all_desried_acc_scenario_save_new, one],axis=0)

        try:
            order_sameratio = same_order / len_model
        except:
            order_sameratio = None
        result_df.loc['dongtai_order_sameratio', one_model_inf] = order_sameratio


        # GT分析。如果generate和expert的下面两个参数相同，并且都有GT，计算GT分布
        # left_interaction_agent_id  right_interaction_agent_id
        # 提取出left不为None，但是right为None的行
        filtered_generate_left_GT = one_model_generate[
            (one_model_generate['left_interaction_agent_id'].notna()) & (one_model_generate['right_interaction_agent_id'].isna())
            & (one_model_generate['GT_left'].notna())]
        filtered_expert_left_GT = one_model_expert[
            (one_model_expert['left_interaction_agent_id'].notna()) & (one_model_expert['right_interaction_agent_id'].isna())
            &(one_model_expert['GT_left'].notna())]

        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_left_GT = pd.merge(filtered_generate_left_GT, filtered_expert_left_GT,
                                    on=['agent_id', 'time_ms', 'left_interaction_agent_id', 'direction'], how='inner')


        # 提取出left不为None，right也不为None的行
        filtered_generate_left_right_GT = one_model_generate[
            (one_model_generate['left_interaction_agent_id'].notna()) & (one_model_generate['right_interaction_agent_id'].notna())
            &(one_model_generate['GT_left'].notna()) & (one_model_generate['GT_right'].notna())]
        filtered_expert_left_right_GT = one_model_expert[
            (one_model_expert['left_interaction_agent_id'].notna()) & (one_model_expert['right_interaction_agent_id'].notna())
            &(one_model_expert['GT_left'].notna()) & (one_model_expert['GT_right'].notna())]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_left_right_GT = pd.merge(filtered_generate_left_right_GT, filtered_expert_left_right_GT,
                                          on=['agent_id', 'time_ms', 'left_interaction_agent_id', 'right_interaction_agent_id','direction'],
                                          how='inner')

        # 提取出left为None，right也为None的行
        filtered_generate_GT = one_model_generate[
            (one_model_generate['left_dongtai_id'].isna()) & (one_model_generate['right_dongtai_id'].isna())
            &(one_model_generate['GT_left'].isna()) & (one_model_generate['GT_right'].isna())]
        filtered_expert_GT = one_model_expert[
            (one_model_expert['left_dongtai_id'].isna()) & (one_model_expert['right_dongtai_id'].isna())
            &(one_model_expert['GT_left'].isna()) & (one_model_expert['GT_right'].isna())]

        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_GT = pd.merge(filtered_generate_GT, filtered_expert_GT,
                                on=['agent_id', 'time_ms','direction'], how='inner')


        # 提取出left为None，right不为None的行
        filtered_generate_right_GT = one_model_generate[
            (one_model_generate['left_dongtai_id'].isna()) & (one_model_generate['right_dongtai_id'].notna())
            &(one_model_generate['GT_left'].isna()) & (one_model_generate['GT_right'].notna())]
        filtered_expert_right_GT = one_model_expert[
            (one_model_expert['left_dongtai_id'].isna()) & (one_model_expert['right_dongtai_id'].notna())
            &(one_model_expert['GT_left'].isna()) & (one_model_expert['GT_right'].notna())]
        # 对比两个dataframe提取出来的行，agent_id、time_ms相同的行数
        common_rows_right_GT = pd.merge(filtered_generate_right_GT, filtered_expert_right_GT,
                                     on=['agent_id', 'time_ms', 'right_interaction_agent_id','direction'], how='inner')


        # 对上述提取出来的相同的行，计算JS距离
        js_data_df_GT = pd.concat([common_rows_left_GT, common_rows_left_right_GT], axis=0)
        js_data_df_GT = pd.concat([js_data_df_GT, common_rows_GT], axis=0)
        js_data_df_GT = pd.concat([js_data_df_GT, common_rows_right_GT], axis=0)

        # 合并之后，因为generate和expert的列名相同，所以合并公式的第一个dataframe的列名+'_x'(generate)，第二个dataframe的列名+'_y'(expert)
        target_parameter_GT = ['GT_left', 'GT_right']

        for j in range(len(target_parameter_GT)):
            target_GT = target_parameter_GT[j]

            # print(expert_data[target],DATA_POINT_generate_flying_useful[target])
            # 分别计算左转车和直行车 expert和generate 相关desired_acc的KL散度
            expert_data_left_GT = js_data_df_GT[js_data_df_GT['direction'] == 'left']
            generate_data_left_GT = js_data_df_GT[js_data_df_GT['direction'] == 'left']

            expert_data_straight_GT = js_data_df_GT[js_data_df_GT['direction'] == 'straight']
            generate_data_straight_GT = js_data_df_GT[js_data_df_GT['direction'] == 'straight']

            expert_data_one_pos_left_GT = expert_data_left_GT[[target_GT + '_y']]
            generate_data_one_pos_left_GT = generate_data_left_GT[[target_GT + '_x']]

            expert_data_one_pos_straight_GT = expert_data_straight_GT[[target_GT + '_y']]
            generate_data_one_pos_straight_GT = generate_data_straight_GT[[target_GT + '_x']]


            expert_data_one_pos_left2_GT = expert_data_one_pos_left_GT[
                abs(expert_data_one_pos_left_GT[target_GT + '_y']) <= 25]
            generate_data_one_pos_left2_GT = generate_data_one_pos_left_GT[
                abs(generate_data_one_pos_left_GT[target_GT + '_x']) <= 25]

            expert_data_one_pos_straight2_GT = expert_data_one_pos_straight_GT[
                abs(expert_data_one_pos_straight_GT[target_GT + '_y']) <= 25]
            generate_data_one_pos_straight2_GT = generate_data_one_pos_straight_GT[
                abs(generate_data_one_pos_straight_GT[target_GT + '_x']) <= 25]


            evaluate_indicator = 'desried_acc'
            if expert_data_one_pos_left2_GT[target_GT + '_y'].nunique() >= 2 and generate_data_one_pos_left2_GT[target_GT + '_x'].nunique() >= 2:
                # print('expert_data_one_pos_left:', expert_data_one_pos_left2, 'generate_data_one_pos_left:',
                #       generate_data_one_pos_left2)
                direction = 'left'

                jensen_shannon_target_parameter_left_GT = jensen_shannon_distance(one_model_inf, expert_data_one_pos_left2_GT,
                                                                               generate_data_one_pos_left2_GT,
                                                                               target_GT, evaluate_indicator,
                                                                               direction)  # 直接得到两个分布的KL散度的差值

                result_df.loc[target_GT + '_jensen_shannon_left', one_model_inf] = jensen_shannon_target_parameter_left_GT
            else:
                result_df.loc[target_GT + '_jensen_shannon_left', one_model_inf] = None
            if expert_data_one_pos_straight2_GT[target_GT + '_y'].nunique() >= 2 and generate_data_one_pos_straight2_GT[target_GT + '_x'].nunique() >= 2:
                direction = 'straight'
                # print('expert_data_one_pos_straight2:', expert_data_one_pos_straight2, 'generate_data_one_pos_straight2:',
                #       generate_data_one_pos_straight2)
                jensen_shannon_target_parameter_straight_GT = jensen_shannon_distance(one_model_inf,
                                                                                   expert_data_one_pos_straight2_GT,
                                                                                   generate_data_one_pos_straight2_GT,
                                                                                   target_GT, evaluate_indicator,
                                                                                   direction)  # 直接得到两个分布的KL散度的差值

                result_df.loc[
                    target_GT + '_jensen_shannon_straight', one_model_inf] = jensen_shannon_target_parameter_straight_GT
            else:
                result_df.loc[
                    target_GT + '_jensen_shannon_straight', one_model_inf] = None

            # 画出每个模型model所有场景的acc和steering的分布图
            root = r"D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight" \
                   r"\results_evaluate\v13\训练集-评价结果\各model参数分布图\柱状图\\%s" % (target_GT)

            if expert_data_one_pos_left2_GT[target_GT+'_y'].nunique() >= 2 and generate_data_one_pos_left2_GT[target_GT+'_x'].nunique() >= 2:
                direction = 'left'
                Distribution_histogram(generate_data_one_pos_left2_GT, expert_data_one_pos_left2_GT, target_GT, one_model_inf,
                                       root, evaluate_indicator, direction)
            if expert_data_one_pos_straight2_GT[target_GT+'_y'].nunique() >= 2 and generate_data_one_pos_straight2_GT[target_GT+'_x'].nunique() >= 2:
                direction = 'straight'
                Distribution_histogram(generate_data_one_pos_straight2_GT, expert_data_one_pos_straight2_GT, target_GT,
                                       one_model_inf, root, evaluate_indicator, direction)


    # 计算新的GT_AVE
    for one_model in tqdm(all_model, position=0):
        one_model.index = range(len(one_model))
        one_model_inf = 'model_' + str(int(one_model['model_id'].iloc[0].item()))
        # print('one_model_inf:',one_model_inf)
        one_model_new = one_model
        one_model_generate0 = one_model[one_model['type'] == 'generate']
        one_model_generate = one_model_generate0[(one_model_generate0['generate_x'] != -4)&(one_model_generate0['GT_AVE'].notna())]
        one_model_generate.index = range(len(one_model_generate))
        one_model_expert0 = one_model[one_model['type'] == 'expert']
        one_model_expert = one_model_expert0[(one_model_expert0['expert_x'] != -4)&(one_model_expert0['GT_AVE'].notna())]
        one_model_expert.index = range(len(one_model_expert))
        # interaction_inf = one_model_generate['model_id'][0]

        # 合并之后，因为generate和expert的列名相同，所以合并公式的第一个dataframe的列名+'_x'(generate)，第二个dataframe的列名+'_y'(expert)
        target_parameter_GT = ['GT_AVE']

        for j in range(len(target_parameter_GT)):
            target = target_parameter_GT[j]
            interaction_inf = str(one_model_new['model_id'].iloc[0].item())
            expert_data_one_pos_new = one_model_expert[[target]]
            generate_data_one_pos_new = one_model_generate[[target]]
            expert_data_one_pos_new2 = expert_data_one_pos_new.dropna()
            expert_data_one_pos_new3 = expert_data_one_pos_new2[(expert_data_one_pos_new2[target] <= 10)]
            generate_data_one_pos_new2 = generate_data_one_pos_new.dropna()
            generate_data_one_pos_new3 = generate_data_one_pos_new2[(generate_data_one_pos_new2[target] <= 10)]
            evaluate_indicator = 'ave_gt'
            direction = None

            print('interaction_inf:', interaction_inf, 'target', target,'输入的专家数据：',expert_data_one_pos_new3,'生成数据；',generate_data_one_pos_new3)
            jensen_shannon_target_parameter = jensen_shannon_distance(interaction_inf, expert_data_one_pos_new3,
                                                                      generate_data_one_pos_new3,
                                                                      target, evaluate_indicator,
                                                                      direction)  # 直接得到两个分布的KL散度的差值

            result_df.loc[target + '_jensen_shannon', 'model_' + str(int(interaction_inf))] = jensen_shannon_target_parameter

            # 画出每个模型model所有场景的acc和steering的分布图
            root = r"D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight" \
                   r"\results_evaluate\v13\训练集-评价结果\各model参数分布图\柱状图\\%s" % (target)
            direction = None

            Distribution_histogram(generate_data_one_pos_new3, expert_data_one_pos_new3, target,
                                   interaction_inf, root, evaluate_indicator, direction)
            # except:
            #     print('interaction_inf:', interaction_inf, 'target', target, 'expert_data_one_pos_new:',
            #           expert_data_one_pos_new3, 'generate_data_one_pos_new:', generate_data_one_pos_new3)

    all_desried_acc_scenario_save_new.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                             r'\MA_Intersection_straight\results_evaluate'
                                     r'\v13\训练集-评价结果\有interaction类型的desried_acc数据\有interaction顺序的desried_acc数据.csv')

    '''
    step2.1 读取每一个model所有测试场景的速度、加速度、航向角，计算出和真实的数据的速度、加速度、航向角的KL散度
    '''
    filedir_KLdata = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight' \
                     r'\results_evaluate\v13\训练集-评价结果\用于做分布图和计算KL散度的数据\\'  # 用于做分布图和计算KL散度的数据

    file_namelist_KLdata = os.listdir(filedir_KLdata)

    root_KLdata = r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs' \
                  r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\各model所有test场景评估参数折线图'  # 存储位置

    # 主程序
    all_KLdata_scenario_save = pd.DataFrame()  # 用于存放文件内所有场景的数据（有超车、跟驰字段信息）

    for filename in tqdm(file_namelist_KLdata, position=0):
        filepath = filedir_KLdata + filename
        current_file_all_trj_KLdata = pd.read_csv(filepath) # 一个场景的数据

        scenario_id = str(filename[:3])
        current_file_all_trj_KLdata['phase_id'] = scenario_id

        all_KLdata_scenario_save = pd.concat([all_KLdata_scenario_save, current_file_all_trj_KLdata],axis=0)
    all_KLdata_scenario_save.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                                    r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\all_KLdata_scenario_save.csv')
    # 按照model_id分组数据
    all_model = [name[1] for name in all_KLdata_scenario_save.groupby(['model_id'])]
    for one_model in tqdm(all_model, position=0):
        one_model.index = range(len(one_model))
        # 计算每一个model下的期望加速度、GT
        # one_model_new = Cal_GT(one_model)  # 输入给模型的是一个model_id的所有场景的数据
        one_model_new = one_model
        target_parameter = ['acc', 'yaw', 'v', 'angle_now']
        for j in range(len(target_parameter)):
            target = target_parameter[j]
            interaction_inf = str(one_model_new['model_id'].iloc[0].item())
            # print(expert_data[target],DATA_POINT_generate_flying_useful[target])
            expert_data_one_pos = one_model_new[one_model_new['expert_x'] != -4]
            generate_data_one_pos = one_model_new[one_model_new['generate_x'] != -4]

            if target != 'GT_AVE':
                expert_data_one_pos_new = expert_data_one_pos[['expert_' + target]]
                generate_data_one_pos_new = generate_data_one_pos[['generate_' + target]]
                expert_data_one_pos_new2 = expert_data_one_pos_new.dropna()
                expert_data_one_pos_new3 = expert_data_one_pos_new2[expert_data_one_pos_new2['expert_' + target] != 'None']
                generate_data_one_pos_new2 = generate_data_one_pos_new.dropna()
                generate_data_one_pos_new3 = generate_data_one_pos_new2[generate_data_one_pos_new2['generate_' + target] != 'None']
                evaluate_indicator = 'regular'
                direction = None
                # try:
                jensen_shannon_target_parameter = jensen_shannon_distance(interaction_inf, expert_data_one_pos_new3, generate_data_one_pos_new3,
                                               target, evaluate_indicator,direction)  # 直接得到两个分布的KL散度的差值

                result_df.loc[target + '_jensen_shannon', 'model_'+str(int(float(interaction_inf)))] = jensen_shannon_target_parameter

                # 画出每个模型model所有场景的acc和steering的分布图
                root = r"D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight" \
                       r"\results_evaluate\v13\训练集-评价结果\各model参数分布图\柱状图\\%s" % (target)
                direction = None

                Distribution_histogram(generate_data_one_pos_new3, expert_data_one_pos_new3, target, interaction_inf, root, evaluate_indicator,direction)
                # except:
                #     print('interaction_inf:',interaction_inf,'target',target,'expert_data_one_pos_new:',expert_data_one_pos_new3,'generate_data_one_pos_new:',generate_data_one_pos_new3)
            else:
                expert_data_one_pos_new = expert_data_one_pos[[target]]
                generate_data_one_pos_new = generate_data_one_pos[[target]]
                expert_data_one_pos_new2 = expert_data_one_pos_new.dropna()
                expert_data_one_pos_new3 = expert_data_one_pos_new2[(expert_data_one_pos_new2[target] != 'None')&(expert_data_one_pos_new2[target] <= 10)]
                generate_data_one_pos_new2 = generate_data_one_pos_new.dropna()
                generate_data_one_pos_new3 = generate_data_one_pos_new2[(generate_data_one_pos_new2[target] != 'None')&(generate_data_one_pos_new2[target] <= 10)]
                evaluate_indicator = 'regular'
                direction = None
                try:
                    jensen_shannon_target_parameter = jensen_shannon_distance(interaction_inf, expert_data_one_pos_new3,
                                                                              generate_data_one_pos_new3,
                                                                              target, evaluate_indicator,
                                                                              direction)  # 直接得到两个分布的KL散度的差值

                    result_df.loc[target + '_jensen_shannon', 'model_' + str(
                        int(interaction_inf))] = jensen_shannon_target_parameter

                    # 画出每个模型model所有场景的acc和steering的分布图
                    root = r"D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs\MA_Intersection_straight" \
                           r"\results_evaluate\v13\训练集-评价结果\各model参数分布图\柱状图\\%s" % (target)
                    direction = None

                    Distribution_histogram(generate_data_one_pos_new3, expert_data_one_pos_new3, target,
                                           interaction_inf, root, evaluate_indicator, direction)
                except:
                    print('interaction_inf:', interaction_inf, 'target', target, 'expert_data_one_pos_new:',
                          expert_data_one_pos_new3, 'generate_data_one_pos_new:', generate_data_one_pos_new3)

    # 画折线图
    print('result_df:',result_df)
    result_df.loc['acc_jensen_shannon', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('acc_jensen_shannon by Model')
    plt.xlabel('Model')
    plt.ylabel('acc_jensen_shannon')
    plt.savefig(root_KLdata + r'\ACC_jensen_shannon\acc_jensen_shannon by Model.png')
    plt.close()

    # 画折线图
    result_df.loc['yaw_jensen_shannon', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('yaw_jensen_shannon by Model')
    plt.xlabel('Model')
    plt.ylabel('yaw_jensen_shannon')
    plt.savefig(root_KLdata + r'\YAW_jensen_shannon\yaw_jensen_shannon by Model.png')
    plt.close()

    # 画折线图
    result_df.loc['v_jensen_shannon', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('v_jensen_shannon by Model')
    plt.xlabel('Model')
    plt.ylabel('v_jensen_shannon')
    plt.savefig(root_KLdata + r'\V_jensen_shannon\v_jensen_shannon by Model.png')
    plt.close()

    # 画折线图
    result_df.loc['angle_now_jensen_shannon', result_df.columns[1:-1]].plot(kind='line', marker='o')

    # 添加标题和标签
    plt.title('angle_now_jensen_shannon by Model')
    plt.xlabel('Model')
    plt.ylabel('angle_now_jensen_shannon')
    plt.savefig(root_KLdata + r'\ANGLE_NOW_jensen_shannon\heading_now_jensen_shannon by Model.png')
    plt.close()

    # 画折线图 # 添加标题和标签 ['ad_rush','ad_yield', 'ad_rush_delta','ad_yield_delta', 'dydt']
    # 左侧neig
    result_df.loc['left_ad_rush_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_rush_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_rush_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\leftneig_ad_rush_jensen_shannon_left\left_ad_rush_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['left_ad_yield_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_yield_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_yield_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\leftneig_ad_yield_jensen_shannon_left\left_ad_yield_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['left_ad_rush_delta_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_rush_delta_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_rush_delta_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\leftneig_ad_rush_delta_jensen_shannon_left\left_ad_rush_delta_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['left_ad_yield_delta_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_yield_delta_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_yield_delta_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\leftneig_ad_yield_delta_jensen_shannon_left\left_ad_yield_delta_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['left_dydt_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_dydt_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('left_dydt_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\leftneig_dydt_jensen_shannon_left\left_dydt_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['left_ad_rush_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_rush_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_rush_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\leftneig_ad_rush_jensen_shannon_straight\left_ad_rush_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['left_ad_yield_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_yield_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_yield_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\leftneig_ad_yield_jensen_shannon_straight\left_ad_yield_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['left_ad_rush_delta_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_rush_delta_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_rush_delta_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\leftneig_ad_rush_delta_jensen_shannon_straight\left_ad_rush_delta_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['left_ad_yield_delta_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_ad_yield_delta_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('left_ad_yield_delta_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\leftneig_ad_yield_delta_jensen_shannon_straight\left_ad_yield_delta_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['left_dydt_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('left_dydt_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('left_dydt_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\leftneig_dydt_jensen_shannon_straight\left_dydt_jensen_shannon_straight by Model.png')
    plt.close()


    # 右侧neig
    result_df.loc['right_ad_rush_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_rush_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_rush_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\rightneig_ad_rush_jensen_shannon_left\right_ad_rush_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['right_ad_yield_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_yield_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_yield_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\rightneig_ad_yield_jensen_shannon_left\right_ad_yield_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['right_ad_rush_delta_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_rush_delta_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_rush_delta_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\rightneig_ad_rush_delta_jensen_shannon_left\right_ad_rush_delta_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['right_ad_yield_delta_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_yield_delta_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_yield_delta_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\rightneig_ad_yield_delta_jensen_shannon_left\right_ad_yield_delta_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['right_dydt_jensen_shannon_left', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_dydt_jensen_shannon_left by Model')
    plt.xlabel('Model')
    plt.ylabel('right_dydt_jensen_shannon_left')
    plt.savefig(root_KLdata + r'\rightneig_dydt_jensen_shannon_left\right_dydt_jensen_shannon_left by Model.png')
    plt.close()

    result_df.loc['right_ad_rush_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_rush_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_rush_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\rightneig_ad_rush_jensen_shannon_straight\right_ad_rush_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['right_ad_yield_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_yield_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_yield_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\rightneig_ad_yield_jensen_shannon_straight\right_ad_yield_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['right_ad_rush_delta_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_rush_delta_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_rush_delta_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\rightneig_ad_rush_delta_jensen_shannon_straight\right_ad_rush_delta_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['right_ad_yield_delta_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_ad_yield_delta_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('right_ad_yield_delta_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\rightneig_ad_yield_delta_jensen_shannon_straight\right_ad_yield_delta_jensen_shannon_straight by Model.png')
    plt.close()

    result_df.loc['right_dydt_jensen_shannon_straight', result_df.columns[1:-1]].plot(kind='line', marker='o')
    plt.title('right_dydt_jensen_shannon_straight by Model')
    plt.xlabel('Model')
    plt.ylabel('right_dydt_jensen_shannon_straight')
    plt.savefig(root_KLdata + r'\rightneig_dydt_jensen_shannon_straight\right_dydt_jensen_shannon_straight by Model.png')
    plt.close()

    '''
    step3 遍历每一个model，找到轨迹rmse最小的model编号，速度、加速度、航向角KL散度之和最小的model编号，rmse和速度、加速度、航向角KL散度之和最小的model编号
    '''

    result_df.loc['all_sum'] = result_df.sum()
    result_df.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                     r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\result_df.csv')

    # 找出和最小的列名
    result_df = result_df.drop(result_df.columns[[0, -1]], axis=1)
    min_model_name_all = result_df.loc['all_sum'].idxmin()
    min_model_name_pos_rmse = result_df.loc['pos_rmse_mean'].idxmin()
    min_model_name_action_KL = result_df.loc[['acc_jensen_shannon', 'yaw_jensen_shannon', 'v_jensen_shannon', 'angle_now_jensen_shannon']].sum().idxmin()
    min_model_name_pos_mean_and_action_KL = result_df.loc[['pos_rmse_mean', 'acc_jensen_shannon', 'yaw_jensen_shannon', 'v_jensen_shannon', 'angle_now_jensen_shannon']].sum().idxmin()
    min_model_name_left_desired_acc_KL = result_df.loc[['left_ad_rush_jensen_shannon_left', 'left_ad_yield_jensen_shannon_left', 'left_ad_rush_delta_jensen_shannon_left',
                                                        'left_ad_yield_delta_jensen_shannon_left', 'left_dydt_jensen_shannon_left',
                                                        'right_ad_rush_jensen_shannon_left', 'right_ad_yield_jensen_shannon_left', 'right_ad_rush_delta_jensen_shannon_left',
                                                        'right_ad_yield_delta_jensen_shannon_left', 'right_dydt_jensen_shannon_left']].sum().idxmin()
    min_model_name_straight_desired_acc_KL = result_df.loc[['left_ad_rush_jensen_shannon_straight', 'left_ad_yield_jensen_shannon_straight', 'left_ad_rush_delta_jensen_shannon_straight',
                                                        'left_ad_yield_delta_jensen_shannon_straight', 'left_dydt_jensen_shannon_straight',
                                                            'right_ad_rush_jensen_shannon_straight', 'right_ad_yield_jensen_shannon_straight', 'right_ad_rush_delta_jensen_shannon_straight',
                                                        'right_ad_yield_delta_jensen_shannon_straight', 'right_dydt_jensen_shannon_straight'
                                                            ]].sum().idxmin()

    min_model_name = [min_model_name_all, min_model_name_pos_rmse, min_model_name_action_KL, min_model_name_pos_mean_and_action_KL,
                      min_model_name_left_desired_acc_KL, min_model_name_straight_desired_acc_KL]
    min_model_name_df = pd.DataFrame(min_model_name)
    min_model_name_df.index = ['min_model_name_all','min_model_name_pos_rmse','min_model_name_action_jensen_shannon','min_model_name_pos_mean_and_action_jensen_shannon',
                               'min_model_name_left_desired_acc_jensen_shannon','min_model_name_straight_desired_acc_jensen_shannon']

    min_model_name_df.to_csv(r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan\ATT-social-iniobs'
                             r'\MA_Intersection_straight\results_evaluate\v13\训练集-评价结果\min_model_name_df.csv')


    '''
    step4 根据这个最小的model，提取出这个model下所有的测试场景数据，包括期望加速度数据，画出期望加速度的数据分布、速度、加速度、航向角、GT的分布
    step3已经把所有model下的图画出来了，人工寻找最合适的即可
    '''