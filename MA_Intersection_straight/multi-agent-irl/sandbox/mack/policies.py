import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
import rl.common.tf_util as U
from rl.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div #, masked
# from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


# mask2 = np.load('C:/Users/uqjsun9/Documents/MA-AIRL-master/multi-agent-irl/mask.nyp', allow_pickle = True)
def masked(ac, X): # ac:一个 NumPy 数组，表示智能体的行动。它的形状应该是 (N, 2)，其中 N 是智能体的数量，每个智能体的行动是一个长度为 2 的一维数组。X：一个 NumPy 数组，表示相关的观察状态。它的形状应该是 (N, M)，其中 N 是智能体的数量，M 是观察状态的维度。
    # print('ac:',ac, np.shape(ac),'X:', np.shape(X),len(X))
    # # print(logits,X)
    # logits=np.array([[1.0, 2.0, 3.0, 4, 5, 6, 7, 8]])
    # X=np.array([[1, 0, 1]])
    # noise = np.random.uniform(size=np.shape(ac))
    for i in range(len(ac)):
        if X[i][0] == 0 or (abs(X[i][4]) <0.01 and abs(X[i][5]) < 0.01) :
            # # print(logits[0], mask[:,int(X[0][0])], np.NINF)
            ac[i] = np.zeros(2) #
        # if ac[i][1]< -0.01:
        #     ac[i][1] = 0.01
            # # print (ac,X)
       
    return ac

class CategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        # print('nbatch为：',nbatch)
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.n  # 原代码，但是报错 AttributeError: 'Box' object has no attribute 'n' 对于 Box 类型的连续空间具有有限数量的元素 n 是无效的，因此该属性不存在。如果您想要观察空间的离散值，则必须实现一种将空间量化为离散值的方法
        # nact = ac_space.shape[0]
        actions = tf.placeholder(tf.int32, (nbatch))
        # all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)  # 源代码
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        # print('logits:',pi,'labels:',actions)
        self.log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=actions)
        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step_log_prob(ob, acts):
            log_prob = sess.run(self.log_prob, {X: ob, actions: acts})
            return log_prob.reshape([-1, 1])

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step_log_prob = step_log_prob
        self.step = step
        self.value = value

class MaskedCategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)       
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = 15 #ac_space.n
        actions = tf.placeholder(tf.int32, (nbatch))
        all_ac_shape = (nbatch, (sum([15 for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        # mask2 = tf.constant(mask)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x: x)
            
           

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)


        self.log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=actions)
        v0 = vf[:, 0]

        # a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step_log_prob(ob, acts):
            log_prob = sess.run(self.log_prob, {X: ob, actions: acts})
            return log_prob.reshape([-1, 1])

        def step(ob, obs, a_v, *_args, **_kwargs):
            
            if a_v is not None:
                a, v = sess.run([pi, v0], {X: ob, X_v: obs, A_v: a_v})
                # # print(a, v)
            else:
                a, v = sess.run([pi, v0], {X: ob, X_v: obs})
                
            # # print(a) 
            # a = masked(a, ob)
            
                
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step_log_prob = step_log_prob
        self.step = step
        self.value = value


class GaussianPolicy(object):

    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):  # ob_space是一个agent的观察空间,ob_spaces是所有agent的观察空间

        # print('运行的是nvn的policy')

        nbatch = nenv * nsteps # 10, 10*36*5 # 根据 nbatch = nenv * nsteps 的计算，每个训练更新步骤中的 nbatch 个样本应该来自于nenv个不同的智能体，每个智能体与环境互动nsteps个时间步。然而，实际上样本如何分配给每个CPU线程取决于你的训练框架和设置。
        ob_shape = (nbatch, ob_space.shape[0] * nstack) # (10, 19) (10*36*5,19)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack) # (10, 19*67) (10*36*5,19*67)
        # # print('GaussianPolicy的ob_shape:',ob_shape,'GaussianPolicy的all_ob_shape:',all_ob_shape)
        nact = ac_space.shape[0] # 2
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack) # (10, 67*2-2)
        X = tf.placeholder(tf.float32, ob_shape)  # 一个agent的观察值
        X_v = tf.placeholder(tf.float32, all_ob_shape) # 所有agent的观察值
        A_v = tf.placeholder(tf.float32, all_ac_shape) # 创建 TensorFlow 占位符，用于接收观察状态 X、所有智能体观察状态 X_v 和其他所有智能体行动 A_v。

        # 策略网络的输出 pi 用于生成智能体的行动，而标准差 std 用于定义高斯策略的概率分布，决定了策略输出的随机性。
        # 通过控制标准差的大小，可以调整策略的探索程度。不同的策略网络可以具有不同的参数，因为它们都在不同的 TensorFlow 变量作用域下创建。

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse): # 定义 TensorFlow 变量作用域，用于创建策略网络的变量，这里包括神经网络的层和权重。
            h1 = fc(X, 'fc1', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h3 = fc(h2, 'fc3', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)  # 创建4个全连接层 h1, h2, h3, h4，用于构建策略网络的前向传播。这些层接受观察状态 X 作为输入，并生成潜在策略。
            # h5 = fc(h4, 'fc5', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            # h6 = fc(h5, 'fc6', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            # h7 = fc(h6, 'fc7', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            # h8 = fc(h7, 'fc8', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            # pi = fc(h2, 'pi', nact, act=lambda x: x, init_scale=0.01)
            pi = fc(h4, 'pi', nact, act=lambda x: x) # 创建一个全连接层 pi，用于生成策略的输出。这里的 nact 表示动作参数数量，pi 的输出将用于生成智能体的行动。

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse): # 再次进入相同的 TensorFlow 变量作用域，用于定义策略网络的标准差。标准差用于构建高斯策略的分布，决定了策略输出的随机性。
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0)) # 这行代码创建了表示高斯策略标准差的 TensorFlow 变量 logstd。shape=[nact] 指定了标准差的形状，这里的 nact 是行动数量，因此对于每个行动都有一个标准差。dtype=tf.float32 指定了变量的数据类型为浮点数。initializer=tf.constant_initializer(0.0) 表示将标准差初始化为0。
            logstd = tf.expand_dims(logstd, 0) # 这行代码通过 tf.expand_dims 在 logstd 上增加了一个维度，将其形状从 [nact] 变为 [1, nact]。这是为了与后面的操作兼容，因为策略输出需要一个标准差。
            std = tf.exp(logstd) # 这行代码通过取 logstd 的指数化来计算标准差 std，将标准差的值从对数空间转换为线性空间
            std = tf.tile(std, [nbatch, 1]) # 这行代码使用 tf.tile 复制标准差 std，使其具有与批次大小 nbatch 相匹配的形状。这是因为在高斯策略中，每个样本都有一个独立的标准差，而 nbatch 表示批次中的样本数量。

        # 这部分代码的作用是创建值函数网络，该网络用于估计状态的值或期望回报，以帮助智能体在强化学习任务中进行决策。值函数网络的输入通常包括观察状态和行动，用于更精确地估计状态的价值。
        # 与策略网络不同，值函数网络的输出是一个值，而不是行动的概率分布。这两个网络在强化学习中起着不同的作用，策略网络用于决定智能体的行动，而值函数网络用于评估状态的价值。
        with tf.variable_scope('value_{}'.format(name), reuse=reuse): # 定义 TensorFlow 变量作用域，用于创建值函数网络（Value Function Network），用于估计状态的价值或预测期望回报。
            if len(ob_spaces) > 1: # 如果agent的个数大于1
                Y = tf.concat([X_v, A_v], axis=1) # 如果有多个智能体，这行代码将观察状态 X_v 和行动 A_v 沿着 axis=1 的轴连接起来，形成一个扩展的输入 Y。这是因为值函数通常需要考虑智能体的状态和行动来估计状态值。[10,1273 + 132],axis=1 表示在第1维度（即列）上进行连接，
            else:
                Y = X_v

            h11 = fc(Y, 'fc11', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h12 = fc(h11, 'fc12', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h13 = fc(h12, 'fc13', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h14 = fc(h13, 'fc14', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh) # 创建4个全连接层 h11, h12, h13, h14，用于构建值函数网络的前向传播。这些层接受观察状态和行动作为输入，并生成值函数的估计。
            vf = fc(h14, 'v', 1, act=lambda x: x)

        # 这部分代码定义了策略网络和值函数网络的输出以及与这些网络相关的一些方法和属性。
        v0 = vf[:, 0] # 从值函数网络的输出 vf 中提取第一列，即值函数的估计值。这表示我们只关心单一数值的状态值，而不考虑多个状态值。
        # a0 = pi
        a0 = pi + tf.random.normal(tf.shape(pi), 0.0, 0.6) *std # 这一行代码定义了采样的行动 a0。它基于策略网络的输出 pi 生成，同时添加了一个高斯噪声以增加策略的探索性。tf.random.normal(tf.shape(pi), 0.0, 0.5) 创建了一个与策略 pi 相同形状的高斯噪声，然后与 pi 相加，从而得到采样的行动。
        # if a0[0] > 1:
        #     a0[0] = 1
        # elif a0[0] < -1:
        #     a0[0] = -1

        self.initial_state = []  # not stateful 这里初始化了 self.initial_state 属性为空列表，表示这个策略类不涉及状态信息的传递，所以这个属性保持为空。

        def step(ob, obs, a_v, *_args, **_kwargs): # 定义了一个名为 step 的方法，用于执行策略并返回智能体的行动和值。方法接受一些参数，包括当前agent的观察状态 ob、所有智能体的观察状态 obs，以及其他所有智能体的行动 a_v。
            if a_v is not None:
                # # print('policy里的输入的obs为:',np.shape(obs), obs)
                # 如果 a_v 不为 None，则通过 TensorFlow 会话 sess 运行策略网络，计算行动 a 和值函数估计 v，并将当前观察状态 ob、所有智能体的观察状态 obs 和所有智能体的行动 a_v 传递给 TensorFlow 图中的相应占位符。
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
                # # print('policy里的动作和值函数为:',a, v )
            else:
                # 如果 a_v 为 None，则仅计算行动 a 和值函数估计 v，并将当前观察状态 ob 和所有智能体的观察状态 obs 传递给 TensorFlow 图中的相应占位符。
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})

            # a = masked(a, ob)     # 最后，通过 masked 函数对行动 a 进行处理，该函数可能用于对行动进行进一步的处理。方法返回行动 a、值函数估计 v 以及一个空列表，作为“虚拟状态”。

            return a, v, []  # dummy state 虚拟状态

        def value(ob, a_v, *_args, **_kwargs): # 定义了一个名为 value 的方法，用于估算状态值。方法接受当前观察状态 ob 和所有智能体的行动 a_v。
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        # 接下来的代码段为对象的属性赋值，将策略网络和值函数网络的输出以及定义的方法与对象关联起来，以便后续可以在训练中使用这些属性和方法。
        #
        # self.X = X、self.X_v = X_v、self.A_v = A_v：输入占位符，分别表示观察状态、所有智能体的观察状态和所有智能体的行动。
        # self.pi = pi：策略网络的输出，表示智能体的行动。
        # self.a0 = a0：采样的行动，是基于策略输出并添加高斯噪声得到的行动。
        # self.vf = vf：值函数估计，表示状态值。
        # self.step（执行策略的方法）、self.value（估算状态值的方法）。
        # 这些属性和方法将在训练和执行智能体策略时使用。

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.a0 = a0
        self.vf = vf
        # self.std = std
        # self.logstd = logstd
        self.step = step
        self.value = value
        #self.step_log_prob =
        # self.mean_std = tf.concat([pi, std], axis=1)
        # 这部分代码的作用是定义了与策略网络和值函数网络相关的属性和方法，并将它们与对象关联起来，以便在训练中使用这些属性和方法来执行策略、估算状态值和进行价值评估。
        # 这些方法允许智能体与环境进行交互，并学习如何在给定观察状态下选择行动，以最大化累积奖励。同时，添加高斯噪声有助于增加策略的探索性，以更好地探索环境。


class MASKATTGaussianPolicy(object):

    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):  # ob_space是一个agent的观察空间,ob_spaces是所有agent的观察空间

        # print('运行的是nvn的policy')

        nbatch = nenv * nsteps # 10, 10*36*5 # 根据 nbatch = nenv * nsteps 的计算，每个训练更新步骤中的 nbatch 个样本应该来自于nenv个不同的智能体，每个智能体与环境互动nsteps个时间步。然而，实际上样本如何分配给每个CPU线程取决于你的训练框架和设置。
        ob_shape = (nbatch, ob_space.shape[0] * nstack) # (10, 19) (10*36*5,19)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack) # (10, 8*46*1)
        # # print('GaussianPolicy的ob_shape:',ob_shape,'GaussianPolicy的all_ob_shape:',all_ob_shape)
        nact = ac_space.shape[0] # 2
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack) # (10, 67*2-2)
        step_sizes = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        sequence_length_lstm = len(step_sizes)
        head_dim_lstm = 3

        X = tf.placeholder(tf.float32, ob_shape)  # 一个agent的观察值
        X_v = tf.placeholder(tf.float32, all_ob_shape) # 所有agent的观察值
        A_v = tf.placeholder(tf.float32, all_ac_shape) # 创建 TensorFlow 占位符，用于接收观察状态 X、所有智能体观察状态 X_v 和其他所有智能体行动 A_v。

        ob_shape_attention = (nbatch, 21, ob_space.shape[0] * nstack)  # (nbatch, 21, 46)
        X_attention = tf.placeholder(tf.float32, ob_shape_attention)
        ob_shape_LSTM = (nbatch, 21, 100 * nstack)  # (nbatch, 21, 64*1) # 这里的64是自注意力层的fc的参数

        all_ob_shape_LSTM_att = (nbatch, 21*8*ob_space.shape[0] * nstack)  # (10, 21*8, 46)
        X_v_LSTM_att = tf.placeholder(tf.float32, all_ob_shape_LSTM_att)  # 所有agent的观察值  (nbatch, 21*8, 46)
        # 策略网络的输出 pi 用于生成智能体的行动，而标准差 std 用于定义高斯策略的概率分布，决定了策略输出的随机性。
        # 通过控制标准差的大小，可以调整策略的探索程度。不同的策略网络可以具有不同的参数，因为它们都在不同的 TensorFlow 变量作用域下创建。

        mask_onetime_shape = (21, nbatch, 10, 10)  # (21, nbatch, 10, 10)
        Mask_onetime_all = tf.placeholder(tf.bool, mask_onetime_shape)

        mask_alltime_shape = (nbatch, 21, 21)  # (nbatch, 21, 21)
        Mask_alltime = tf.placeholder(tf.bool, mask_alltime_shape)


        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):  # reuse是再利用的意思
            # reuse=True 在这个作用域下，如果变量不存在，则尝试使用给定的名称创建变量；如果变量已经存在，则重用该变量。
            # reuse=False 在这个作用域下，尝试创建已存在的变量会引发错误。
            # 定义self_attention
            # X_LSTM的shape为(nbatch, 21, 46)
            # 定义注意力机制网络。每个注意力机制输入的是当前时刻obs_lstm的一个历史时刻（包括当前时刻）X_LSTM其中一行的一个时刻
            hidden_dim = 10

            # 策略网络
            def self_attention_gpt(inputs, mask_tensor, num_heads=1, head_dim=3, name_="self_attention"):
                # Linearly project the inputs into queries, keys, and values
                with tf.variable_scope(name_, reuse=tf.AUTO_REUSE):
                    # 单头注意力机制
                    num_units = 128
                    # print('inputs:', np.shape(inputs))  # (batch, seq, feature_num)
                    # queries = fc(inputs, 'fc_q', nh=num_units, init_scale=np.sqrt(0), act=tf.nn.tanh)
                    # keys = fc(inputs, 'fc_k', nh=num_units, init_scale=np.sqrt(0), act=tf.nn.tanh)
                    # values = fc(inputs, 'fc_v', nh=num_units, init_scale=np.sqrt(0), act=tf.nn.tanh)

                    queries = tf.layers.dense(inputs=inputs, units=num_units, name='fc_q')
                    keys = tf.layers.dense(inputs=inputs, units=num_units, name='fc_k')
                    values = tf.layers.dense(inputs=inputs, units=num_units, name='fc_v')

                    # print('queries:',np.shape(queries))  # (batch, seq, num_units)

                    # queries = tf.layers.dense(inputs, num_units, kernel_initializer=tf.compat.v1.keras.initializers.he_normal())  # activation=tf.nn.relu,
                    # print('queries的变量：', tf.compat.v1.trainable_variables())
                    # keys = tf.layers.dense(inputs, num_units, kernel_initializer=tf.compat.v1.keras.initializers.he_normal())
                    # print('keys的变量：', tf.compat.v1.trainable_variables())
                    # values = tf.layers.dense(inputs, num_units, kernel_initializer=tf.compat.v1.keras.initializers.he_normal())
                    # print('values的变量：', tf.compat.v1.trainable_variables())

                    # 定义掩码为 TensorFlow 常量张量
                    # mask_tensor = tf.constant(mask_array, dtype=tf.bool, name='mask')
                    # print('mask_tensor:', np.shape(mask_tensor), mask_tensor)  # (batch, 10, 10) 时间维度就是（batch, 21, 21)
                    padding_val = -2 ** 32
                    scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(
                        tf.cast(8, dtype=tf.float32))  # tf.to_float
                    # scores /= tf.sqrt(tf.cast(num_units, tf.float32))
                    scores_mask = tf.where(mask_tensor, scores, tf.ones_like(scores) * padding_val) / tf.sqrt(
                        tf.cast(num_units, tf.float32))  # 采用的是缩放的点积
                    # print('测试mask前后的scores,batch0:',scores[0],scores_mask[0])

                    # print('scores:', np.shape(scores))  # (batch,sequence,sequence)
                    # Apply Softmax to scores
                    attention_weights = tf.nn.softmax(scores_mask, axis=-1)  # （batch, sequence, sequence）
                    # print('scores_softmax:', np.shape(attention_weights),attention_weights)  # （batch, sequence, sequence）(10, 10, 10)
                    z_value = tf.matmul(attention_weights, values)
                    # print('z_value:', np.shape(z_value))  # (batch, sequence, num_units)  (10, 10, 128)

                    inputs_add = tf.layers.dense(inputs, num_units, activation=None)

                    # z_value_outputs = tf.layers.dense(z_value, inputs.get_shape().as_list()[-1], activation=None)  # (batch,10,features_num)
                    # print('z_value_outputs：',np.shape(z_value))  # (batch,seq,features_num(inputs.get_shape().as_list()[-1]))
                    add_z = tf.add(inputs_add, z_value)  # (batch,seq,features_num(inputs.get_shape().as_list()[-1]))
                    # tf.concat([stacked_attention_outputs, stacked_sub_inputs_new], axis=-1)
                    # print('add_z:', np.shape(add_z))  # (batch,seq,num_units)  (10, 10, 128)
                    # stacked_attention_outputs = LayerNormalization(epsilon=1e-6)(x + inputs_new)
                    # normalized_output = tf.contrib.layers.layer_norm(sub_output_, begin_norm_axis=1)  # 先归一化 begin_norm_axis=1对不同样本的同一对象进行归一化。
                    normalized_add_z = tf.contrib.layers.layer_norm(add_z, begin_norm_axis=2)  # 先归一化 begin_norm_axis=2对不同样本的不同时间分别进行归一化。
                    # print('normalized_add_z:', np.shape(normalized_add_z))  # (batch, 10, num_units)  (10, 10, 128)
                    # self-attention add & norm 后 接前馈层
                    # 定义第一层全连接层
                    hidden1 = tf.layers.dense(inputs=normalized_add_z, units=num_units,activation=tf.nn.relu, name='att_fc1')
                    # 定义第二层全连接层
                    hidden2 = tf.layers.dense(inputs=hidden1, units=num_units, activation=tf.nn.relu, name='att_fc2')
                    # 定义第三层全连接层
                    hidden3 = tf.layers.dense(inputs=hidden2, units=num_units, activation=tf.nn.relu, name='att_fc2')

                    feed_forward_output = tf.layers.dense(inputs=hidden3, units=normalized_add_z.get_shape().as_list()[-1],activation=None, name='att_fc2')
                    # print('feed_forward_output:', np.shape(feed_forward_output))  # (batch, 10, num_units)  (10, 10, 128)
                    # add 残差连接
                    feed_forward_output_add = tf.add(normalized_add_z, feed_forward_output)
                    # 归一化
                    normalized_feed_forward_output_add = tf.contrib.layers.layer_norm(feed_forward_output_add,
                                                                                      begin_norm_axis=2)
                    # print('z_value:',np.shape(z_value))  # (batch, sequence, num_units) (10, 10, 128)
                    # print('z_value:', np.shape(z_value))  # （batch, sequence, sequence）

                    # # Reshape for multi-heads
                    # queries = tf.reshape(queries, [-1, tf.shape(inputs)[1], num_heads, head_dim])
                    # keys = tf.reshape(keys, [-1, tf.shape(inputs)[1], num_heads, head_dim])
                    # values = tf.reshape(values, [-1, tf.shape(inputs)[1], num_heads, head_dim])

                    # # print('queries:', np.shape(queries))  # (batch,sequence,num_heads,head_dim)
                    # # print('keys:', np.shape(keys))  # (batch,sequence,num_heads,head_dim)
                    # # print('values:', np.shape(values))  # (batch,sequence,num_heads,head_dim)
                    # '''
                    # 这一部分通过 tf.reshape 操作，将 queries、keys、values 的形状进行变换，
                    # 其中 tf.shape(inputs)[1] 是序列的长度，num_heads 是多头注意力的头数，
                    # head_dim 是每个头的维度。这样做是为了让每个头都有自己的查询、键、值。
                    # '''
                    # scores = []  # 存放每个头的注意力得分
                    # z_values = []  # 存放每个头得到的z
                    # Calculate attention scores
                    # 将 query 和 key 进行转置，使得 num_heads 和 sequence 交换位置
                    # 对转置后的 query 和 key 进行矩阵相乘

                    # for head in range(num_heads):
                    #     query = queries[:, :, head, :]  # (batch, sequence, head_dim)
                    #     key = queries[:, :, head, :]  # (batch, sequence, head_dim)
                    #     value = values[:, :, head, :]  # (batch, sequence, head_dim)
                    #     score = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.to_float(head_dim))
                    #     # print('每个头的score:', np.shape(score))
                    #     # 这里计算了每个头的注意力得分，shape为（batch, sequence, sequence）
                    #     scores.append(score)
                    #     score_weight = tf.nn.softmax(score, axis=-1)  # (batch, sequence, sequence)
                    #     '''
                    #     这部分计算每一头注意力的分数。tf.matmul(query, key, transpose_b=True)
                    #     计算了 query 和 key 的点积，
                    #     transpose_b=True 表示将 key 进行转置。
                    #     最后除以 tf.sqrt(tf.to_float(head_dim)) 进行缩放，以确保数值稳定性。
                    #     score 的形状（shape）取决于输入 query 和 key 的形状，以及矩阵相乘的规则。
                    #     如果 query 的形状是 [batch_size, sequence_length, head_dim]，
                    #     而 key 的形状是 [batch_size, head_dim, sequence_length]，
                    #     那么 score 的形状将是
                    #     [batch_size, sequence_length, sequence_length]。
                    #     然后再softmax
                    #     '''
                    #     # Weighted sum of values using attention weights
                    #     z_value = tf.matmul(score_weight, value)  # (batch, sequence, head_dim)
                    #     z_values.append(z_value)
                    #
                    #     '''
                    #     使用注意力权重对 value 进行加权求和。tf.matmul(score_softmax, value)
                    #     相当于将每个头的注意力权重应用到对应的 value 上
                    #     z_value 的形状取决于 value 的形状。
                    #     z_value的shape为[batch_size, sequence_length, head_dim]
                    #     '''

                    # # print('scores:', np.shape(scores))  # (2,)
                    # scores_combined = tf.reduce_sum(tf.stack(scores, axis=-1), axis=-1)
                    # # print('scores_combined:', np.shape(scores_combined))  # （batch, sequence, sequence）
                    #
                    # # Apply Softmax to scores
                    # attention_weights = tf.nn.softmax(scores_combined, axis=-1)
                    # # print('scores_softmax:', np.shape(attention_weights))  # (?, ?, ?) （batch, sequence, sequence）

                    # '''
                    # 这一行使用 softmax 函数对分数进行归一化，得到注意力权重。
                    # axis=-1 表示在最后一个轴上进行 softmax 操作，即在每个头的和注意力得分的维度上进行 softmax。
                    # attention_weights的shape为【batch, sequence_length, sequence_length】
                    # '''
                    # # print('z_values:', np.shape(z_values))  # (2,)
                    # z_values_combined = tf.concat(z_values, axis=-1)
                    # # print('z_values_combined:', np.shape(z_values_combined))  # (batch, sequence, num_heads * head_dim)

                    # Linear transformation with a learnable weight matrix W
                    # W0 = tf.get_variable('W0_attention', shape=[num_heads * head_dim, head_dim],
                    #                     initializer=tf.contrib.layers.xavier_initializer())
                    #
                    # z_transformed = tf.tensordot(z_values_combined, W0, axes=[-1, 0])
                    # # print('z_transformed:', np.shape(z_transformed))  # (batch, sequence, head_dim)
                    # Linear projection to get the final output

                    # # print('output:', np.shape(output))
                    # '''
                    # 这一行进行最终的线性投影，将加权和的结果映射到最终的输出。tf.layers.dense(z_transformed, head_dim)
                    # 相当于对加权和进行线性变换。
                    # output 的形状是 [batch_size, sequence_length, head_dim]。
                    # '''
                return attention_weights, normalized_feed_forward_output_add


            attention_outputs = []  # 存储每个时刻的注意力机制输出
            sub_inputs_new = []  # 存储每个时刻的输入的子数据
            attention_weights = []  # 存储每个时刻的的注意力权重矩阵
            # 将一行数据中的每个时刻的数据都进行一遍

            # 使用 .get_shape() 获取张量的形状
            shape_ = X_attention.get_shape().as_list()
            # 提取出第二个维度的大小，即 21
            num_batch = shape_[0]
            num_time = shape_[1]
            num_features = 10  # 每个序列的参数的个数

            def process_one_time_step(i):
                # for i in range(num_time):
                X_ONE_TIME = X_attention[:, i, :]
                # print('X_ONE_TIME:',np.shape(X_ONE_TIME))  # (batch,46)  (10, 46)
                # 把X_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
                # 定义每个时间步长的特征数

                # 以下是tensorflow1.x版本下的代码
                sub_inputsRag_all = []
                # 遍历每个样本
                for j in range(num_batch):
                    # 创建一个零张量
                    sub_inputsRag_j = tf.zeros([0, num_features], dtype=tf.float32)
                    # 记录当前位置
                    current_pos = 0
                    # 遍历每个特征的长度
                    for k, step_size in enumerate(step_sizes):
                        # 截取当前时间步长的特征
                        feature_slice = X_ONE_TIME[j, current_pos: current_pos + step_size]
                        # # print('feature_slice:',np.shape(feature_slice))
                        # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
                        if feature_slice.shape[0] < num_features:
                            pad_size = num_features - feature_slice.shape[0]
                            feature_slice = tf.pad(feature_slice, paddings=[[0, pad_size]])
                        # 在垂直方向堆叠
                        # # print('tf.expand_dims(feature_slice, axis=0):',np.shape(tf.expand_dims(feature_slice, axis=0)))
                        sub_inputsRag_j = tf.concat([sub_inputsRag_j, tf.expand_dims(feature_slice, axis=0)], axis=0)
                        # 更新当前位置
                        current_pos += step_size
                    sub_inputsRag_all.append(sub_inputsRag_j)

                # 在垂直方向堆叠，形成 RaggedTensor
                sub_inputsRag = tf.stack(sub_inputsRag_all, axis=0)  # （nbatch, 10, 10)

                # print('sub_inputsRag:', np.shape(sub_inputsRag))  # （nbatch, 10, 10)  (10, 10, 10)

                Mask_onetime = Mask_onetime_all[i]
                # print('Mask_onetime:', np.shape(Mask_onetime))  # （nbatch, 10, 10)  (10, 10, 10)
                sub_attention_weights, sub_output_ = self_attention_gpt(sub_inputsRag, mask_tensor=Mask_onetime,
                                                                        num_heads=1, head_dim=head_dim_lstm,
                                                                        name_="self_attention_")

                return sub_output_, sub_inputsRag, sub_attention_weights

            # 使用 map_fn 并行计算
            attention_outputs, sub_inputs_new, attention_weights = tf.map_fn(process_one_time_step,
                                                                             tf.range(num_time),
                                                                             dtype=(tf.float32, tf.float32, tf.float32))
            # 把attention_outputs中的数据根据第二维给堆叠起来，形成shape为【batch_size, 21, sequence_length * head_dim】
            # print('attention_outputs:',np.shape(attention_outputs))  # (21,batch,seq,units) (21, 10, 10, 128)  # evaluate (21, 1, units)
            # print('sub_inputs_new:', np.shape(sub_inputs_new))  # (21,batch,seq,feature_num) (21, 10, 10, 10) # evaluate (21, 1, 10, 10)
            # print('attention_weights:',np.shape(attention_weights))  # (21,batch,seq,seq) (21, 10, 10, 10) # evaluate (21, 1, 10, 10)


            attention_outputs_reshape = tf.transpose(attention_outputs,perm=[1, 0, 2, 3])  # (batch,21,seq,units)
            # 将维度调整为（batch，21, units）
            attention_outputs_reshaped = tf.reshape(attention_outputs_reshape,[tf.shape(attention_outputs_reshape)[0], 21, -1])
            # print('attention_outputs_reshaped:', np.shape(attention_outputs_reshaped))  # （batch, 21, seq*units）
            attention_reshaped_inputs = tf.layers.dense(attention_outputs_reshaped, 21, activation=None)  # (batch,21,seq*units)
            # print('Mask_alltime:', np.shape(Mask_alltime))  #  (batch, 21, 21)
            # print('attention_reshaped_inputs:', np.shape(attention_reshaped_inputs))  #  (batch, 21, 21)
            sub_attention_weights_time, sub_output_time = self_attention_gpt(attention_outputs_reshaped,
                                                                             mask_tensor=Mask_alltime, num_heads=1,
                                                                             head_dim=head_dim_lstm,
                                                                             name_="self_attention_time")
            # print('sub_attention_weights_time:', np.shape(sub_attention_weights_time))  # (batch,time_seq,time_seq) (10, 21, 21)  # evaluate (21, 1, units)
            # print('sub_output_time:', np.shape(sub_output_time))  # (batch,time_seq,num_units)  (10, 21, 128) # evaluate ()


            # decoder:

            sub_output_time_reshaped = tf.reshape(sub_output_time,[tf.shape(attention_outputs_reshape)[0],-1])
            # print('sub_output_time_reshaped:',np.shape(sub_output_time_reshaped))  # (batch, 2688)
            # 预测未来1s内的轨迹点位置，decoder, 四层全连接层，这里的全连接层用的是fc，只能输入二维数据。也可以用tf.layers.dense，可以输入三维数据
            # 第一组预测  # 先试试输入三维数据
            model1_h1 = fc(sub_output_time_reshaped, 'model1_fc1_att_time', nh=128, init_scale=np.sqrt(0),
                           act=tf.nn.sigmoid)
            model1_h2 = fc(model1_h1, 'model1_fc2_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)
            model1_h3 = fc(model1_h2, 'model1_fc3_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)
            model1_h4 = fc(model1_h3, 'model1_fc4_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)  # 创建4个全连接层 h1, h2, h3, h4，用于构建策略网络的前向传播。这些层接受观察状态 X 作为输入，并生成潜在策略。
            model1_h5 = fc(model1_h4, 'model1_fc5_att_time', nh=128, init_scale=np.sqrt(0),act=tf.nn.tanh)  # 回归头
            pi = fc(model1_h5, 'pi', nact, act=lambda x: x) # 创建5个全连接层 h1, h2, h3, h4, h5，用于构建策略网络的前向传播。这些层接受观察状态 X 作为输入，并生成潜在策略。

            # self-attention接LSTM
            # X_LSTM = tf.reshape(normalized_stacked_attention_outputs, ob_shape_LSTM)  # 一个agent的包含历史时刻的观察值
            # # X_LSTM = stacked_attention_outputs
            # # 把stacked_attention_outputs输入lstm中，对时间序列进行分析，stacked_attention_outputs的shape为【batch，21，sequence_length * head_dim】
            # # 定义 LSTM 层
            # # inputs = tf.unstack(X_LSTM, axis=1)
            # lstm_layers = []
            # num_lstm_layers = 4  # 4层LSTM
            # lstm_units = 128  # 每一层128个LSTM单元
            # # 定义 LSTM 单元
            #
            # def lstm_cell():
            #     return tf.contrib.rnn.BasicLSTMCell(lstm_units)
            #
            # stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            #     [lstm_cell() for _ in range(num_lstm_layers)])
            # outputs, _ = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=X_LSTM, dtype=tf.float32)
            #
            # # 将 LSTM 输出 reshape 为二维张量
            # # print('outputs:',np.shape(outputs))  # (batch, 21, 128)
            # lstm_flat = tf.reshape(outputs[:, -1, :], [nbatch, -1])
            #
            # # 输出层，将输出限制在[-1, 1]范围内
            # pi = tf.tanh(tf.layers.dense(lstm_flat, ac_space.shape[0], activation=None))


        with tf.variable_scope('policy_{}'.format(name), reuse=reuse): # 再次进入相同的 TensorFlow 变量作用域，用于定义策略网络的标准差。标准差用于构建高斯策略的分布，决定了策略输出的随机性。
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0)) # 这行代码创建了表示高斯策略标准差的 TensorFlow 变量 logstd。shape=[nact] 指定了标准差的形状，这里的 nact 是行动数量，因此对于每个行动都有一个标准差。dtype=tf.float32 指定了变量的数据类型为浮点数。initializer=tf.constant_initializer(0.0) 表示将标准差初始化为0。
            logstd = tf.expand_dims(logstd, 0) # 这行代码通过 tf.expand_dims 在 logstd 上增加了一个维度，将其形状从 [nact] 变为 [1, nact]。这是为了与后面的操作兼容，因为策略输出需要一个标准差。
            std = tf.exp(logstd) # 这行代码通过取 logstd 的指数化来计算标准差 std，将标准差的值从对数空间转换为线性空间
            std = tf.tile(std, [nbatch, 1]) # 这行代码使用 tf.tile 复制标准差 std，使其具有与批次大小 nbatch 相匹配的形状。这是因为在高斯策略中，每个样本都有一个独立的标准差，而 nbatch 表示批次中的样本数量。

        # 这部分代码的作用是创建值函数网络，该网络用于估计状态的值或期望回报，以帮助智能体在强化学习任务中进行决策。值函数网络的输入通常包括观察状态和行动，用于更精确地估计状态的价值。
        # 与策略网络不同，值函数网络的输出是一个值，而不是行动的概率分布。这两个网络在强化学习中起着不同的作用，策略网络用于决定智能体的行动，而值函数网络用于评估状态的价值。
        with tf.variable_scope('value_{}'.format(name), reuse=reuse): # 定义 TensorFlow 变量作用域，用于创建值函数网络（Value Function Network），用于估计状态的价值或预测期望回报。
            if len(ob_spaces) > 1: # 如果agent的个数大于1
                Y = tf.concat([X_v_LSTM_att, A_v], axis=1) # 如果有多个智能体，这行代码将观察状态 X_v 和行动 A_v 沿着 axis=1 的轴连接起来，形成一个扩展的输入 Y。这是因为值函数通常需要考虑智能体的状态和行动来估计状态值。[10,1273 + 132],axis=1 表示在第1维度（即列）上进行连接，
            else:
                Y = X_v_LSTM_att

            h11 = fc(Y, 'fc11', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h12 = fc(h11, 'fc12', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h13 = fc(h12, 'fc13', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h14 = fc(h13, 'fc14', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh) # 创建4个全连接层 h11, h12, h13, h14，用于构建值函数网络的前向传播。这些层接受观察状态和行动作为输入，并生成值函数的估计。
            vf = fc(h14, 'v', 1, act=lambda x: x)

        # 这部分代码定义了策略网络和值函数网络的输出以及与这些网络相关的一些方法和属性。
        v0 = vf[:, 0] # 从值函数网络的输出 vf 中提取第一列，即值函数的估计值。这表示我们只关心单一数值的状态值，而不考虑多个状态值。
        # a0 = pi
        a0 = pi + tf.random_normal(tf.shape(pi), 0.0, 0.6) *std # tf.random.normal 这一行代码定义了采样的行动 a0。它基于策略网络的输出 pi 生成，同时添加了一个高斯噪声以增加策略的探索性。tf.random.normal(tf.shape(pi), 0.0, 0.5) 创建了一个与策略 pi 相同形状的高斯噪声，然后与 pi 相加，从而得到采样的行动。
        attention_weights_output_spatial = attention_weights
        attention_weights_output_temporal = sub_attention_weights_time
        # if a0[0] > 1:
        #     a0[0] = 1
        # elif a0[0] < -1:
        #     a0[0] = -1

        self.initial_state = []  # not stateful 这里初始化了 self.initial_state 属性为空列表，表示这个策略类不涉及状态信息的传递，所以这个属性保持为空。
        # ob_lstm[k], ob[k], obs, a_v
        def step(ob_attention, ob, obs, a_v, is_training, mask_atime, mask_times, *_args, **_kwargs): # 定义了一个名为 step 的方法，用于执行策略并返回智能体的行动和值。方法接受一些参数，包括当前agent的观察状态 ob、所有智能体的观察状态 obs，以及其他所有智能体的行动 a_v。
            if is_training == True:
                obs_flat = obs.reshape(ob_attention.shape[0], -1)
            else:
                obs_flat = obs.reshape(1, -1)
            if a_v is not None:
                # print('nbatch:',nbatch,'policy里的输入的ob_attention为:',np.shape(ob_attention))  #  (batch, 21, 46)
                # print('nbatch:', nbatch, 'policy里的输入的ob为:', np.shape(ob))  # (batch, 46)
                # print('nbatch:', nbatch, 'policy里的输入的obs为:', np.shape(obs), np.shape(obs_flat))  # (batch, 21*8, 46)  (batch, 21*8*46)
                # print('nbatch:', nbatch, 'policy里的输入的a_v为:', np.shape(a_v))  # (batch, 14)
                # print('nbatch:', nbatch, 'policy里的输入的mask_atime为:', np.shape(mask_atime))  # (21, batch, 10, 10)
                # print('nbatch:', nbatch, 'policy里的输入的mask_times为:', np.shape(mask_times))  # (batch, 21, 21)
                # 如果 a_v 不为 None，则通过 TensorFlow 会话 sess 运行策略网络，计算行动 a 和值函数估计 v，并将当前观察状态 ob、所有智能体的观察状态 obs 和所有智能体的行动 a_v 传递给 TensorFlow 图中的相应占位符。
                a, v, att_weights_output_spatial, att_weights_output_temporal = sess.run([a0, v0, attention_weights_output_spatial, attention_weights_output_temporal], {X_attention: ob_attention, X_v_LSTM_att: obs_flat,
                                                                                         A_v: a_v, Mask_onetime_all: mask_atime, Mask_alltime: mask_times})
                # # print('policy里的动作和值函数为:',a, v )
            else:
                # 如果 a_v 为 None，则仅计算行动 a 和值函数估计 v，并将当前观察状态 ob 和所有智能体的观察状态 obs 传递给 TensorFlow 图中的相应占位符。
                a, v, att_weights_output_spatial, att_weights_output_temporal = sess.run([a0, v0, attention_weights_output_spatial, attention_weights_output_temporal], {X_attention: ob_attention, X_v_LSTM_att: obs_flat})

            # a = masked(a, ob)     # 最后，通过 masked 函数对行动 a 进行处理，该函数可能用于对行动进行进一步的处理。方法返回行动 a、值函数估计 v 以及一个空列表，作为“虚拟状态”。

            return a, v, [], att_weights_output_spatial, att_weights_output_temporal  # dummy state 虚拟状态

        def value(ob, a_v, *_args, **_kwargs): # 定义了一个名为 value 的方法，用于估算状态值。方法接受当前观察状态 ob 和所有智能体的行动 a_v。
            ob_flat = ob.reshape(10, -1)
            if a_v is not None:
                return sess.run(v0, {X_v_LSTM_att: ob_flat, A_v: a_v})
            else:
                return sess.run(v0, {X_v_LSTM_att: ob_flat})

        # 接下来的代码段为对象的属性赋值，将策略网络和值函数网络的输出以及定义的方法与对象关联起来，以便后续可以在训练中使用这些属性和方法。
        #
        # self.X = X、self.X_v = X_v、self.A_v = A_v：输入占位符，分别表示观察状态、所有智能体的观察状态和所有智能体的行动。
        # self.pi = pi：策略网络的输出，表示智能体的行动。
        # self.a0 = a0：采样的行动，是基于策略输出并添加高斯噪声得到的行动。
        # self.vf = vf：值函数估计，表示状态值。
        # self.step（执行策略的方法）、self.value（估算状态值的方法）。
        # 这些属性和方法将在训练和执行智能体策略时使用。
        self.X_attention = X_attention
        # self.X_LSTM = X_LSTM
        self.X_v_LSTM_att = X_v_LSTM_att
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.a0 = a0
        self.vf = vf
        # self.std = std
        # self.logstd = logstd
        self.step = step
        self.value = value
        self.attention_weights_output_spatial = attention_weights_output_spatial
        self.attention_weights_output_temporal = attention_weights_output_temporal
        # self.attention_weights_output = attention_weights_output
        self.Mask_onetime_all = Mask_onetime_all
        self.Mask_alltime = Mask_alltime
        #self.step_log_prob =
        # self.mean_std = tf.concat([pi, std], axis=1)
        # 这部分代码的作用是定义了与策略网络和值函数网络相关的属性和方法，并将它们与对象关联起来，以便在训练中使用这些属性和方法来执行策略、估算状态值和进行价值评估。
        # 这些方法允许智能体与环境进行交互，并学习如何在给定观察状态下选择行动，以最大化累积奖励。同时，添加高斯噪声有助于增加策略的探索性，以更好地探索环境。

class LSTMGaussianPolicy(object):

    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):  # ob_space是一个agent的观察空间,ob_spaces是所有agent的观察空间

        # print('运行的是nvn的policy')

        nbatch = nenv * nsteps # 10, 10*36*5 # 根据 nbatch = nenv * nsteps 的计算，每个训练更新步骤中的 nbatch 个样本应该来自于nenv个不同的智能体，每个智能体与环境互动nsteps个时间步。然而，实际上样本如何分配给每个CPU线程取决于你的训练框架和设置。
        ob_shape = (nbatch, ob_space.shape[0] * nstack) # (10, 19) (10*36*5,19)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack) # (10, 8*46*1)
        # # print('GaussianPolicy的ob_shape:',ob_shape,'GaussianPolicy的all_ob_shape:',all_ob_shape)
        nact = ac_space.shape[0] # 2
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack) # (10, 67*2-2)
        step_sizes = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        sequence_length_lstm = len(step_sizes)
        head_dim_lstm = 3

        X = tf.placeholder(tf.float32, ob_shape)  # 一个agent的观察值
        X_v = tf.placeholder(tf.float32, all_ob_shape) # 所有agent的观察值
        A_v = tf.placeholder(tf.float32, all_ac_shape) # 创建 TensorFlow 占位符，用于接收观察状态 X、所有智能体观察状态 X_v 和其他所有智能体行动 A_v。

        ob_shape_attention = (nbatch, 21, ob_space.shape[0] * nstack)  # (nbatch, 21, 46)
        X_attention = tf.placeholder(tf.float32, ob_shape_attention)
        ob_shape_LSTM = (nbatch, 21, 100 * nstack)  # (nbatch, 21, 64*1) # 这里的64是自注意力层的fc的参数

        all_ob_shape_LSTM_att = (nbatch, 21*8*ob_space.shape[0] * nstack)  # (10, 21*8, 46)
        X_v_LSTM_att = tf.placeholder(tf.float32, all_ob_shape_LSTM_att)  # 所有agent的观察值  (nbatch, 21*8, 46)
        # 策略网络的输出 pi 用于生成智能体的行动，而标准差 std 用于定义高斯策略的概率分布，决定了策略输出的随机性。
        # 通过控制标准差的大小，可以调整策略的探索程度。不同的策略网络可以具有不同的参数，因为它们都在不同的 TensorFlow 变量作用域下创建。



        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):  # reuse是再利用的意思
            # reuse=True 在这个作用域下，如果变量不存在，则尝试使用给定的名称创建变量；如果变量已经存在，则重用该变量。
            # reuse=False 在这个作用域下，尝试创建已存在的变量会引发错误。
            # 定义self_attention
            # X_LSTM的shape为(nbatch, 21, 46)
            # 定义注意力机制网络。每个注意力机制输入的是当前时刻obs_lstm的一个历史时刻（包括当前时刻）X_LSTM其中一行的一个时刻
            hidden_dim = 10

            # 策略网络
            def self_attention_gpt(inputs, num_heads=1, head_dim=3, name_="self_attention"):
                # Linearly project the inputs into queries, keys, and values
                with tf.variable_scope(name_, reuse=tf.AUTO_REUSE):

                    # queries = tf.layers.dense(inputs, num_heads * head_dim, activation=None)
                    # print('queries的变量：', tf.compat.v1.trainable_variables())
                    # keys = tf.layers.dense(inputs, num_heads * head_dim, activation=None)
                    # print('keys的变量：', tf.compat.v1.trainable_variables())
                    # values = tf.layers.dense(inputs, num_heads * head_dim, activation=None)
                    # print('values的变量：', tf.compat.v1.trainable_variables())
                    print('inputs:',np.shape(inputs)) # (10, 10, 10)
                    queries = fc(inputs, 'fc_q', nh=64, init_scale=np.sqrt(0), act=tf.nn.tanh)
                    keys = fc(inputs, 'fc_k', nh=64, init_scale=np.sqrt(0), act=tf.nn.tanh)
                    values = fc(inputs, 'fc_v', nh=10, init_scale=np.sqrt(0), act=tf.nn.tanh)

                    # print('queries:', np.shape(queries))  # (batch,sequence,num_features)
                    # print('keys:', np.shape(keys))  # (batch,sequence,num_features)
                    # print('values:', np.shape(values))  # (batch,sequence,num_features)
                    scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(8, dtype=tf.float32))  # tf.to_float
                    # print('scores:', np.shape(scores))  # (batch,sequence,num_features)
                    # Apply Softmax to scores
                    attention_weights = tf.nn.softmax(scores, axis=-1)  # （batch, sequence, sequence）
                    # print('scores_softmax:', np.shape(attention_weights))  # (?, ?, ?) （batch, sequence, sequence）
                    z_value = tf.matmul(attention_weights, values)
                    print('z_value:',np.shape(z_value))  # (10, 10, 100)
                    # print('z_value:', np.shape(z_value))  # （batch, sequence, sequence）

                    # # Reshape for multi-heads
                    # queries = tf.reshape(queries, [-1, tf.shape(inputs)[1], num_heads, head_dim])
                    # keys = tf.reshape(keys, [-1, tf.shape(inputs)[1], num_heads, head_dim])
                    # values = tf.reshape(values, [-1, tf.shape(inputs)[1], num_heads, head_dim])

                    # # print('queries:', np.shape(queries))  # (batch,sequence,num_heads,head_dim)
                    # # print('keys:', np.shape(keys))  # (batch,sequence,num_heads,head_dim)
                    # # print('values:', np.shape(values))  # (batch,sequence,num_heads,head_dim)
                    # '''
                    # 这一部分通过 tf.reshape 操作，将 queries、keys、values 的形状进行变换，
                    # 其中 tf.shape(inputs)[1] 是序列的长度，num_heads 是多头注意力的头数，
                    # head_dim 是每个头的维度。这样做是为了让每个头都有自己的查询、键、值。
                    # '''
                    # scores = []  # 存放每个头的注意力得分
                    # z_values = []  # 存放每个头得到的z
                    # Calculate attention scores
                    # 将 query 和 key 进行转置，使得 num_heads 和 sequence 交换位置
                    # 对转置后的 query 和 key 进行矩阵相乘

                    # for head in range(num_heads):
                    #     query = queries[:, :, head, :]  # (batch, sequence, head_dim)
                    #     key = queries[:, :, head, :]  # (batch, sequence, head_dim)
                    #     value = values[:, :, head, :]  # (batch, sequence, head_dim)
                    #     score = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.to_float(head_dim))
                    #     # print('每个头的score:', np.shape(score))
                    #     # 这里计算了每个头的注意力得分，shape为（batch, sequence, sequence）
                    #     scores.append(score)
                    #     score_weight = tf.nn.softmax(score, axis=-1)  # (batch, sequence, sequence)
                    #     '''
                    #     这部分计算每一头注意力的分数。tf.matmul(query, key, transpose_b=True)
                    #     计算了 query 和 key 的点积，
                    #     transpose_b=True 表示将 key 进行转置。
                    #     最后除以 tf.sqrt(tf.to_float(head_dim)) 进行缩放，以确保数值稳定性。
                    #     score 的形状（shape）取决于输入 query 和 key 的形状，以及矩阵相乘的规则。
                    #     如果 query 的形状是 [batch_size, sequence_length, head_dim]，
                    #     而 key 的形状是 [batch_size, head_dim, sequence_length]，
                    #     那么 score 的形状将是
                    #     [batch_size, sequence_length, sequence_length]。
                    #     然后再softmax
                    #     '''
                    #     # Weighted sum of values using attention weights
                    #     z_value = tf.matmul(score_weight, value)  # (batch, sequence, head_dim)
                    #     z_values.append(z_value)
                    #
                    #     '''
                    #     使用注意力权重对 value 进行加权求和。tf.matmul(score_softmax, value)
                    #     相当于将每个头的注意力权重应用到对应的 value 上
                    #     z_value 的形状取决于 value 的形状。
                    #     z_value的shape为[batch_size, sequence_length, head_dim]
                    #     '''

                    # # print('scores:', np.shape(scores))  # (2,)
                    # scores_combined = tf.reduce_sum(tf.stack(scores, axis=-1), axis=-1)
                    # # print('scores_combined:', np.shape(scores_combined))  # （batch, sequence, sequence）
                    #
                    # # Apply Softmax to scores
                    # attention_weights = tf.nn.softmax(scores_combined, axis=-1)
                    # # print('scores_softmax:', np.shape(attention_weights))  # (?, ?, ?) （batch, sequence, sequence）

                    # '''
                    # 这一行使用 softmax 函数对分数进行归一化，得到注意力权重。
                    # axis=-1 表示在最后一个轴上进行 softmax 操作，即在每个头的和注意力得分的维度上进行 softmax。
                    # attention_weights的shape为【batch, sequence_length, sequence_length】
                    # '''
                    # # print('z_values:', np.shape(z_values))  # (2,)
                    # z_values_combined = tf.concat(z_values, axis=-1)
                    # # print('z_values_combined:', np.shape(z_values_combined))  # (batch, sequence, num_heads * head_dim)

                    # Linear transformation with a learnable weight matrix W
                    # W0 = tf.get_variable('W0_attention', shape=[num_heads * head_dim, head_dim],
                    #                     initializer=tf.contrib.layers.xavier_initializer())
                    #
                    # z_transformed = tf.tensordot(z_values_combined, W0, axes=[-1, 0])
                    # # print('z_transformed:', np.shape(z_transformed))  # (batch, sequence, head_dim)
                    # Linear projection to get the final output

                    # # print('output:', np.shape(output))
                    # '''
                    # 这一行进行最终的线性投影，将加权和的结果映射到最终的输出。tf.layers.dense(z_transformed, head_dim)
                    # 相当于对加权和进行线性变换。
                    # output 的形状是 [batch_size, sequence_length, head_dim]。
                    # '''
                return attention_weights, z_value





            attention_outputs = []  # 存储每个时刻的注意力机制输出
            sub_inputs_new = []  # 存储每个时刻的输入的子数据
            attention_weights = []  # 存储每个时刻的的注意力权重矩阵
            # 将一行数据中的每个时刻的数据都进行一遍

            # 使用 .get_shape() 获取张量的形状
            shape_ = X_attention.get_shape().as_list()
            # 提取出第二个维度的大小，即 21
            num_batch = shape_[0]
            num_time = shape_[1]
            num_features = 10  # 每个序列的参数的个数
            def process_one_time_step(i):
            # for i in range(num_time):
                X_ONE_TIME = X_attention[:, i, :]
                # 把X_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
                # 定义每个时间步长的特征数

                # 以下是tensorflow1.x版本下的代码
                sub_inputsRag_all = []
                # 遍历每个样本
                for j in range(num_batch):
                    # 创建一个零张量
                    sub_inputsRag_j = tf.zeros([0, num_features], dtype=tf.float32)
                    # 记录当前位置
                    current_pos = 0
                    # 遍历每个特征的长度
                    for k, step_size in enumerate(step_sizes):
                        # 截取当前时间步长的特征
                        feature_slice = X_ONE_TIME[j, current_pos: current_pos + step_size]
                        # # print('feature_slice:',np.shape(feature_slice))
                        # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
                        if feature_slice.shape[0] < num_features:
                            pad_size = num_features - feature_slice.shape[0]
                            feature_slice = tf.pad(feature_slice, paddings=[[0, pad_size]])
                        # 在垂直方向堆叠
                        # # print('tf.expand_dims(feature_slice, axis=0):',np.shape(tf.expand_dims(feature_slice, axis=0)))
                        sub_inputsRag_j = tf.concat([sub_inputsRag_j, tf.expand_dims(feature_slice, axis=0)], axis=0)
                        # 更新当前位置
                        current_pos += step_size
                    sub_inputsRag_all.append(sub_inputsRag_j)

                # 在垂直方向堆叠，形成 RaggedTensor
                sub_inputsRag = tf.stack(sub_inputsRag_all, axis=0)

                print('sub_inputsRag:',np.shape(sub_inputsRag))
                sub_attention_weights, sub_output_ = self_attention_gpt(sub_inputsRag, num_heads=1, head_dim=head_dim_lstm, name_="self_attention_")
                # print('sub_attention_weights:',np.shape(sub_attention_weights))  # (?, ?, ?) （batch, sequence, sequence）
                # 每次循环得到的sub_attention_weights的shape为【batch, sequence_length, sequence_length】
                # 每次循环得到的sub_output的shape为【batch_size, sequence_length, 64】
                # 转化为二维
                # sub_output = tf.layers.dense(sub_output_, head_dim_lstm, activation=None)
                # [batch_size, sequence_length, head_dim_lstm]

                # LayerNormalization


                # normalized_output = tf.contrib.layers.layer_norm(sub_output_, begin_norm_axis=1) # 先归一化 begin_norm_axis=1对不同样本的同一对象进行归一化。
                # normalized_output_2d = tf.reshape(normalized_output, [nbatch, -1])  # 再展平
                # print('normalized_output:',np.shape(normalized_output))  # (10, 10, 64)
                # print('normalized_output_2d:',np.shape(normalized_output_2d))  # (10, 640)
                # sub_output = fc(normalized_output_2d, 'fc_att', nh=64, init_scale=np.sqrt(0), act=tf.nn.tanh)
                # print('sub_output:', np.shape(sub_output))  # (10, 64)
                # [batch_size, sequence_length, 256]
                # batch_size = tf.shape(sub_output)[0]
                # sequence_length = tf.shape(sub_output)[1]
                # head_dim = tf.shape(sub_output)[2]
                # reshaped_sub_output = tf.reshape(sub_output, [batch_size, -1])

                return sub_output_, sub_inputsRag, sub_attention_weights

                # attention_outputs.append(reshaped_sub_output)
                # sub_inputs_new.append(sub_inputsRag)
                # attention_weights.append(sub_attention_weights)

            # 使用 map_fn 并行计算
            attention_outputs, sub_inputs_new, attention_weights = tf.map_fn(process_one_time_step, tf.range(num_time),
                                                                             dtype=(tf.float32, tf.float32, tf.float32))
            # 把attention_outputs中的数据根据第二维给堆叠起来，形成shape为【batch_size, 21, sequence_length * head_dim】
            # print('attention_outputs:',np.shape(attention_outputs))  # (21,) (21, 10, 64)  # evaluate (21, 1, 64)
            # print('sub_inputs_new:', np.shape(sub_inputs_new))  # (21,) (21, 10, 10, 10)  # evaluate (21, 1, 10, 10)
            # print('attention_weights:', np.shape(attention_weights))  # (21,) (21, 10, 10, 10)  # evaluate (21, 1, 10, 10)
            stacked_attention_outputs = tf.reshape(attention_outputs, [21, nbatch, -1])  # (21,10,100)
            print('stacked_attention_outputs：', np.shape(stacked_attention_outputs))
            stacked_attention_outputs = tf.transpose(stacked_attention_outputs, perm=[1, 0, 2])  # (10,21,100)
            stacked_sub_inputs_new = tf.reshape(sub_inputs_new, [21, nbatch, -1])   # (21,10,100)
            stacked_sub_inputs_new = tf.transpose(stacked_sub_inputs_new, perm=[1, 0, 2])  # (10,21,100)
            print('注意力机制的输出x：', np.shape(stacked_attention_outputs), 'inputs_new:', np.shape(stacked_sub_inputs_new))
            concatenated_inputs = tf.concat([stacked_attention_outputs, stacked_sub_inputs_new], axis=-1)
            print('concatenated_inputs:',np.shape(concatenated_inputs))  # (nbatch, 21, 200)
            # stacked_attention_outputs = LayerNormalization(epsilon=1e-6)(x + inputs_new)
            # normalized_output = tf.contrib.layers.layer_norm(sub_output_, begin_norm_axis=1)  # 先归一化 begin_norm_axis=1对不同样本的同一对象进行归一化。
            normalized_stacked_attention_outputs = tf.contrib.layers.layer_norm(concatenated_inputs, begin_norm_axis=2) # 先归一化 begin_norm_axis=2对不同样本的不同时间分别进行归一化。

            # epsilon = 1e-6
            # layer_normalized_output_with_epsilon = normalized_stacked_attention_outputs + epsilon
            # stacked_attention_outputs = tf.stack(attention_outputs, axis=1)
            print('normalized_stacked_attention_outputs:', np.shape(normalized_stacked_attention_outputs))  # (nbatch, 21, 200)

            # self-attention接前馈层
            normalized_stacked_attention_outputs_ = tf.reshape(normalized_stacked_attention_outputs, [nbatch, -1])
            h1 = fc(normalized_stacked_attention_outputs_, 'fc1_att', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2_att', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h3 = fc(h2, 'fc3_att', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4_att', nh=256, init_scale=np.sqrt(0),
                    act=tf.nn.tanh)  # 创建4个全连接层 h1, h2, h3, h4，用于构建策略网络的前向传播。这些层接受观察状态 X 作为输入，并生成潜在策略。
            pi = fc(h4, 'pi', nact, act=lambda x: x)

            # self-attention接LSTM
            # X_LSTM = tf.reshape(normalized_stacked_attention_outputs, ob_shape_LSTM)  # 一个agent的包含历史时刻的观察值
            # # X_LSTM = stacked_attention_outputs
            # # 把stacked_attention_outputs输入lstm中，对时间序列进行分析，stacked_attention_outputs的shape为【batch，21，sequence_length * head_dim】
            # # 定义 LSTM 层
            # # inputs = tf.unstack(X_LSTM, axis=1)
            # lstm_layers = []
            # num_lstm_layers = 4  # 4层LSTM
            # lstm_units = 128  # 每一层128个LSTM单元
            # # 定义 LSTM 单元
            #
            # def lstm_cell():
            #     return tf.contrib.rnn.BasicLSTMCell(lstm_units)
            #
            # stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            #     [lstm_cell() for _ in range(num_lstm_layers)])
            # outputs, _ = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=X_LSTM, dtype=tf.float32)
            #
            # # 将 LSTM 输出 reshape 为二维张量
            # # print('outputs:',np.shape(outputs))  # (batch, 21, 128)
            # lstm_flat = tf.reshape(outputs[:, -1, :], [nbatch, -1])
            #
            # # 输出层，将输出限制在[-1, 1]范围内
            # pi = tf.tanh(tf.layers.dense(lstm_flat, ac_space.shape[0], activation=None))


        with tf.variable_scope('policy_{}'.format(name), reuse=reuse): # 再次进入相同的 TensorFlow 变量作用域，用于定义策略网络的标准差。标准差用于构建高斯策略的分布，决定了策略输出的随机性。
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0)) # 这行代码创建了表示高斯策略标准差的 TensorFlow 变量 logstd。shape=[nact] 指定了标准差的形状，这里的 nact 是行动数量，因此对于每个行动都有一个标准差。dtype=tf.float32 指定了变量的数据类型为浮点数。initializer=tf.constant_initializer(0.0) 表示将标准差初始化为0。
            logstd = tf.expand_dims(logstd, 0) # 这行代码通过 tf.expand_dims 在 logstd 上增加了一个维度，将其形状从 [nact] 变为 [1, nact]。这是为了与后面的操作兼容，因为策略输出需要一个标准差。
            std = tf.exp(logstd) # 这行代码通过取 logstd 的指数化来计算标准差 std，将标准差的值从对数空间转换为线性空间
            std = tf.tile(std, [nbatch, 1]) # 这行代码使用 tf.tile 复制标准差 std，使其具有与批次大小 nbatch 相匹配的形状。这是因为在高斯策略中，每个样本都有一个独立的标准差，而 nbatch 表示批次中的样本数量。

        # 这部分代码的作用是创建值函数网络，该网络用于估计状态的值或期望回报，以帮助智能体在强化学习任务中进行决策。值函数网络的输入通常包括观察状态和行动，用于更精确地估计状态的价值。
        # 与策略网络不同，值函数网络的输出是一个值，而不是行动的概率分布。这两个网络在强化学习中起着不同的作用，策略网络用于决定智能体的行动，而值函数网络用于评估状态的价值。
        with tf.variable_scope('value_{}'.format(name), reuse=reuse): # 定义 TensorFlow 变量作用域，用于创建值函数网络（Value Function Network），用于估计状态的价值或预测期望回报。
            if len(ob_spaces) > 1: # 如果agent的个数大于1
                Y = tf.concat([X_v_LSTM_att, A_v], axis=1) # 如果有多个智能体，这行代码将观察状态 X_v 和行动 A_v 沿着 axis=1 的轴连接起来，形成一个扩展的输入 Y。这是因为值函数通常需要考虑智能体的状态和行动来估计状态值。[10,1273 + 132],axis=1 表示在第1维度（即列）上进行连接，
            else:
                Y = X_v_LSTM_att

            h11 = fc(Y, 'fc11', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h12 = fc(h11, 'fc12', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h13 = fc(h12, 'fc13', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h14 = fc(h13, 'fc14', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh) # 创建4个全连接层 h11, h12, h13, h14，用于构建值函数网络的前向传播。这些层接受观察状态和行动作为输入，并生成值函数的估计。
            vf = fc(h14, 'v', 1, act=lambda x: x)

        # 这部分代码定义了策略网络和值函数网络的输出以及与这些网络相关的一些方法和属性。
        v0 = vf[:, 0] # 从值函数网络的输出 vf 中提取第一列，即值函数的估计值。这表示我们只关心单一数值的状态值，而不考虑多个状态值。
        # a0 = pi
        a0 = pi + tf.random.normal(tf.shape(pi), 0.0, 0.3) *std # tf.random.normal 这一行代码定义了采样的行动 a0。它基于策略网络的输出 pi 生成，同时添加了一个高斯噪声以增加策略的探索性。tf.random.normal(tf.shape(pi), 0.0, 0.5) 创建了一个与策略 pi 相同形状的高斯噪声，然后与 pi 相加，从而得到采样的行动。
        attention_weights_output = attention_weights
        # if a0[0] > 1:
        #     a0[0] = 1
        # elif a0[0] < -1:
        #     a0[0] = -1

        self.initial_state = []  # not stateful 这里初始化了 self.initial_state 属性为空列表，表示这个策略类不涉及状态信息的传递，所以这个属性保持为空。
        # ob_lstm[k], ob[k], obs, a_v
        def step(ob_attention, ob, obs, a_v, is_training, *_args, **_kwargs): # 定义了一个名为 step 的方法，用于执行策略并返回智能体的行动和值。方法接受一些参数，包括当前agent的观察状态 ob、所有智能体的观察状态 obs，以及其他所有智能体的行动 a_v。
            if is_training == True:
                obs_flat = obs.reshape(10, -1)
            else:
                obs_flat = obs.reshape(1, -1)
            if a_v is not None:
                # print('nbatch:',nbatch,'policy里的输入的ob_attention为:',np.shape(ob_attention))  #  (10, 21, 46)
                # print('nbatch:', nbatch, 'policy里的输入的ob为:', np.shape(ob))  # (10, 46)
                # print('nbatch:', nbatch, 'policy里的输入的obs为:', np.shape(obs), np.shape(obs_flat))  # (10, 21*8, 46)  (10,21*8*46)
                # print('nbatch:', nbatch, 'policy里的输入的a_v为:', np.shape(a_v))  # (10, 14)

                # 如果 a_v 不为 None，则通过 TensorFlow 会话 sess 运行策略网络，计算行动 a 和值函数估计 v，并将当前观察状态 ob、所有智能体的观察状态 obs 和所有智能体的行动 a_v 传递给 TensorFlow 图中的相应占位符。
                a, v, att_weights_output = sess.run([a0, v0, attention_weights_output], {X_attention: ob_attention, X_v_LSTM_att: obs_flat, A_v: a_v})
                # # print('policy里的动作和值函数为:',a, v )
            else:
                # 如果 a_v 为 None，则仅计算行动 a 和值函数估计 v，并将当前观察状态 ob 和所有智能体的观察状态 obs 传递给 TensorFlow 图中的相应占位符。
                a, v, att_weights_output = sess.run([a0, v0, attention_weights_output], {X_attention: ob_attention, X_v_LSTM_att: obs_flat})

            # a = masked(a, ob)     # 最后，通过 masked 函数对行动 a 进行处理，该函数可能用于对行动进行进一步的处理。方法返回行动 a、值函数估计 v 以及一个空列表，作为“虚拟状态”。

            return a, v, [], att_weights_output  # dummy state 虚拟状态

        def value(ob, a_v, *_args, **_kwargs): # 定义了一个名为 value 的方法，用于估算状态值。方法接受当前观察状态 ob 和所有智能体的行动 a_v。
            ob_flat = ob.reshape(10, -1)
            if a_v is not None:
                return sess.run(v0, {X_v_LSTM_att: ob_flat, A_v: a_v})
            else:
                return sess.run(v0, {X_v_LSTM_att: ob_flat})

        # 接下来的代码段为对象的属性赋值，将策略网络和值函数网络的输出以及定义的方法与对象关联起来，以便后续可以在训练中使用这些属性和方法。
        #
        # self.X = X、self.X_v = X_v、self.A_v = A_v：输入占位符，分别表示观察状态、所有智能体的观察状态和所有智能体的行动。
        # self.pi = pi：策略网络的输出，表示智能体的行动。
        # self.a0 = a0：采样的行动，是基于策略输出并添加高斯噪声得到的行动。
        # self.vf = vf：值函数估计，表示状态值。
        # self.step（执行策略的方法）、self.value（估算状态值的方法）。
        # 这些属性和方法将在训练和执行智能体策略时使用。
        self.X_attention = X_attention
        # self.X_LSTM = X_LSTM
        self.X_v_LSTM_att = X_v_LSTM_att
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.a0 = a0
        self.vf = vf
        # self.std = std
        # self.logstd = logstd
        self.step = step
        self.value = value
        self.attention_weights_output = attention_weights_output
        #self.step_log_prob =
        # self.mean_std = tf.concat([pi, std], axis=1)
        # 这部分代码的作用是定义了与策略网络和值函数网络相关的属性和方法，并将它们与对象关联起来，以便在训练中使用这些属性和方法来执行策略、估算状态值和进行价值评估。
        # 这些方法允许智能体与环境进行交互，并学习如何在给定观察状态下选择行动，以最大化累积奖励。同时，添加高斯噪声有助于增加策略的探索性，以更好地探索环境。

class MultiCategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbins = 11
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact * nbins, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        pi = tf.reshape(pi, [nbatch, nact, nbins])
        a0 = sample(pi, axis=2)
        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            # output continuous actions within [-1, 1]
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            a = transform(a)
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        def transform(a):
            # transform from [0, 9] to [-0.8, 0.8]
            a = np.array(a, dtype=np.float32)
            a = (a - (nbins - 1) / 2) / (nbins - 1) * 2.0
            return a

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value




'''
class CategoricalPolicy(object)
这段代码定义了一个类 CategoricalPolicy，该类代表一个离散动作空间的策略网络，用于在强化学习中训练代理（agent）来执行动作。这个策略网络采用了深度神经网络的结构，用于输出动作的概率分布，并且还有一个值函数网络用于估计状态值。

让我逐步解释这个类的主要功能和成员：

__init__ 方法：这是类的构造函数，用于初始化策略网络和值函数网络。它接受以下参数：

sess：TensorFlow 会话对象。
ob_space：观察空间（observation space）的描述，用于指定输入状态的形状和范围。
ac_space：动作空间（action space）的描述，用于指定输出动作的形状和范围。
ob_spaces 和 ac_spaces：观察空间和动作空间的列表，通常用于多智能体环境。
nenv：环境数量，用于确定批次大小。
nsteps：每个批次中的时间步数。
nstack：观察堆叠的数量。
reuse：一个布尔值，表示是否应该重用 TensorFlow 变量。
name：模型的名称。
在构造函数内部，首先计算了一些参数，如批次大小 nbatch、观察空间的形状 ob_shape、所有观察空间的形状 all_ob_shape 以及动作空间的数量 nact。

接下来，定义了三个 TensorFlow 占位符（tf.compat.v1.placeholder）：

actions：用于接收动作数据的占位符。
X：用于接收观察数据的占位符。
X_v 和 A_v：用于接收所有观察数据和动作数据的占位符。
使用 TensorFlow 变量作用域（tf.variable_scope）定义了策略网络和值函数网络的结构。策略网络输出动作的概率分布，而值函数网络估计状态值。这些网络的具体结构包括多个全连接层（fc 函数），并且使用 ReLU 激活函数。

通过策略网络的输出 pi 和占位符 actions，计算了负的稀疏交叉熵损失（self.log_prob）。

定义了一些辅助函数，包括 step_log_prob、step 和 value。这些函数用于在给定观察数据和动作数据时，计算策略的对数概率、执行动作和估计状态值。

最后，设置了一些成员变量，包括输入占位符 X、X_v、A_v，策略网络输出 pi，值函数输出 vf，以及用于执行动作和估计状态值的函数。

总的来说，这个 CategoricalPolicy 类定义了一个用于强化学习的策略网络和值函数网络，用于离散动作空间的问题。这个类的实例可以在强化学习算法中用于训练代理模型，使其学会在给定观察数据下执行合适的离散动作。

'''