## 查看训练过程的损失值



#
#
# import tensorflow as tf
#
# print(tf.__version__)
# print(tf.test.gpu_device_name())
# print(tf.config.experimental.set_visible_devices)
# # print('GPU:', tf.config.list_physical_devices('GPU'))
# # print('CPU:', tf.config.list_physical_devices(device_type='CPU'))
# # print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_gpu_available())
# # 输出可用的GPU数量
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# # 查询GPU设备
#
#
# import tensorflow as tf
# import timeit
#
#
# # 指定在cpu上运行
# def cpu_run():
#     with tf.device('/cpu:0'):
#         cpu_a = tf.random.normal([10000, 1000])
#         cpu_b = tf.random.normal([1000, 2000])
#         c = tf.matmul(cpu_a, cpu_b)
#     return c
#
#
# # 指定在gpu上运行
# def gpu_run():
#     with tf.device('/gpu:0'):
#         gpu_a = tf.random.normal([10000, 1000])
#         gpu_b = tf.random.normal([1000, 2000])
#         c = tf.matmul(gpu_a, gpu_b)
#     return c
#
#
# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print("cpu:", cpu_time, "  gpu:", gpu_time)
#
#
# import numpy as np
#
# # # 指定.npy文件的路径
# # file_path = r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\int_map.npy'
# #
# # # 使用NumPy加载.npy文件
# # loaded_data = np.load(file_path, allow_pickle= True)
# #
# # # 打印加载的数据
# # print(loaded_data)
# #
# # # 如果是多维数组，可以使用shape属性查看其形状
# # print(loaded_data.shape)
#
# import pickle
#
# path = r'C:\Users\60106\Desktop\code-MA-ARIL\MA_Intersection_straight\MA_Intersection\multi-agent-irl\multi-agent-trj\expert_trjs\intersection_131_str_5_4.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
#
# f = open(path, 'rb')
# data = pickle.load(f)
#
# print(data)
# print(len(data))
# print(data[0]['ob'])
#
# #
# # from tensorflow.python.client import device_lib
# # print(device_lib.list_local_devices())
