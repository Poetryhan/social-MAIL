import tensorflow as tf
import numpy as np
import joblib
from rl.acktr.utils import Scheduler, find_trainable_variables
from rl.acktr.utils import fc, mse
from rl.acktr import kfac
from irl.mack.tf_util import relu_layer, linear, tanh_layer
import math

disc_types = ['decentralized', 'centralized', 'single', 'decentralized-all']
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class Discriminator(object):
    def __init__(self, sess, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, scope="discriminator", kfac_clip=0.001, max_grad_norm=0.5,
                 l2_loss_ratio=0.01):
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        self.l2_loss_ratio = l2_loss_ratio
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.scope = scope
        self.index = index
        self.sess = sess
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        print('判别器的ob_space：', ob_space, np.shape(ob_space))  # Box(46,) (46,)
        print('判别器的ac_space：', ac_space, np.shape(ac_space))  # Box(2,) (2,)
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        print('判别器的selfob_shape：', self.ob_shape, np.shape(self.ob_shape))  # self.ob_shape = 46
        print('判别器的selfall_ob_shape：', self.all_ob_shape, np.shape(self.all_ob_shape))  # self.all_ob_shape = 368 = 8 * 46
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        print('nact:',nact)  # nact = 2
        self.ac_shape = nact * nstack
        print('判别器的selfac_shape：', self.all_ob_shape, np.shape(self.all_ob_shape))
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size
        self.type_ = tf.placeholder(tf.string, shape=[])
        self.batch_num = tf.placeholder(tf.int32, shape=[])
        print('之后判别器的selfall_ob_shape：', self.all_ob_shape, np.shape(self.all_ob_shape))  # self.all_ob_shape = 368
        print('之后判别器的selfall_ac_shape：', self.all_ac_shape, np.shape(self.all_ac_shape))  # self.all_ac_shape = 16
        self.canshu_social_allbatch = tf.placeholder(tf.float32, (None, 4))  # 存放所有batch的四个和利己利他奖励有关的参数
        if disc_type == 'decentralized':
            self.obs = tf.placeholder(tf.float32, (None, 21 * self.ob_shape))
            self.nobs = tf.placeholder(tf.float32, (None, 21 * self.ob_shape))
            self.act = tf.placeholder(tf.float32, (None, self.ac_shape))
            self.labels = tf.placeholder(tf.float32, (None, 1))
            self.lprobs = tf.placeholder(tf.float32, (None, 1))
        elif disc_type == 'decentralized-all':
            self.obs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
            self.nobs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
            self.act = tf.placeholder(tf.float32, (None, self.all_ac_shape))
            self.labels = tf.placeholder(tf.float32, (None, 1))
            self.lprobs = tf.placeholder(tf.float32, (None, 1))
        else:
            assert False

        # self.obs = tf.reshape(self.obs, [-1, 21 * 46])
        # self.nobs = tf.reshape(self.nobs, [-1, 21 * 46])

        self.lr_rate = tf.placeholder(tf.float32, ())

        with tf.variable_scope(self.scope):
            rew_input = self.obs
            # rew_input_flat = tf.reshape(rew_input, [-1, 21 * 57])
            # rew_input_fuyuan = tf.reshape(rew_input, [tf.shape(rew_input)[0], 21, 57])
            canshu_social = self.canshu_social_allbatch
            print('rew_input:', np.shape(rew_input))\
            # print('rew_input_flat:',np.shape(rew_input_flat),
            #       'rew_input_fuyuan:',np.shape(rew_input_fuyuan),'canshu_social:',np.shape(canshu_social)) # (?, 21, 46)
            # rew_input: (?, 1176) rew_input_flat: (?, 1176) rew_input_fuyuan: (?, 21, 56) canshu_social: (?, 4)
            if not self.state_only:
                rew_input = tf.concat([self.obs, self.act], axis=1)

            with tf.variable_scope('reward'):
                # 先计算社交倾向φ
                # 定义第一个隐藏层，增加神经元数量
                # 定义输入层
                input_layer = rew_input

                # 定义第一个隐藏层，使用 Leaky ReLU 作为激活函数
                hidden_layer1 = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.leaky_relu)

                # 定义第二个隐藏层，使用 ELU 作为激活函数
                hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=100, activation=tf.nn.elu)

                # 定义第三个隐藏层，使用 SELU 作为激活函数
                hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=100, activation=tf.nn.selu)

                # 定义输出层，使用 tanh 作为激活函数
                social_pre = tf.layers.dense(inputs=hidden_layer3, units=1, activation=tf.nn.tanh)

                # social_pre = fc(rew_input, 'social_pre', nh=1, init_scale=np.sqrt(0), act=tf.nn.tanh)
                social_pre = tf.multiply(social_pre, tf.constant(np.pi / 2, dtype=tf.float32))
                # social_pre = tf.atan(self.relu_net(rew_input, dout=1))  # 确保在-pai/2~pai/2之间 (batch,1)
                # 提取 social_pre 中的值
                social_pre_values = tf.squeeze(social_pre, axis=1)  # 移除 social_pre 的单维度，形状变为 (batch_size,)

                # 以下这种做法是 对于没有交互对象的情况 人为的定义社交倾向为全利己
                # 判断第四列是否为 0
                # column4_is_zero = tf.equal(canshu_social[:, -1], 0)
                #
                # # 根据条件判断处理方式
                # def condition(column4_is_zero, social_pre_values, canshu_social):
                #     # 计算 sin 和 cos
                #     sin_a = tf.sin(social_pre_values)
                #     cos_a = tf.cos(social_pre_values)
                #
                #     # 计算 canshu_social 的前三列的均值
                #     canshu_social_avg = tf.reduce_mean(canshu_social[:, :3], axis=1)
                #
                #     # 如果第四列不为 0
                #     def true_fn():
                #         # 计算 canshu_social 的前三列的均值乘以 sin(a)
                #         canshu_social_avg_sin = canshu_social_avg * sin_a
                #
                #         # 计算 canshu_social 的最后一列乘以 cos(a)
                #         canshu_social_cos = canshu_social[:, -1] * cos_a
                #
                #         return canshu_social_avg_sin, canshu_social_cos
                #
                #     # 如果第四列为 0
                #     def false_fn():
                #         # 将 sin(a) 设为 1，cos(a) 设为 0
                #         sin_a = tf.ones_like(social_pre_values)
                #         cos_a = tf.zeros_like(social_pre_values)
                #
                #         # 计算 canshu_social 的前三列的均值乘以 sin(a)
                #         canshu_social_avg_sin = canshu_social_avg * sin_a
                #
                #         # 计算 canshu_social 的最后一列乘以 cos(a)
                #         canshu_social_cos = canshu_social[:, -1] * cos_a
                #
                #         return canshu_social_avg_sin, canshu_social_cos
                #
                #     # 根据条件选择处理方式
                #     canshu_social_avg_sin, canshu_social_cos = tf.cond(column4_is_zero, true_fn, false_fn)
                #
                #     return canshu_social_avg_sin, canshu_social_cos
                #
                # # 根据条件选择处理方式并计算新的值
                # new_values_sin, new_values_cos = condition(column4_is_zero, social_pre_values, canshu_social)
                #
                # # 合并两部分结果
                # new_values = new_values_sin + new_values_cos

                # 这种做法是，对任何情况都不人为的定义，全部由网络自己学习，如果无交互对象，那么就期待倾向更利己
                # 计算 cos（利己）和 sin（利他）
                cos_a = tf.cos(social_pre_values)
                sin_a = tf.sin(social_pre_values)

                # 计算 canshu_social 的前三列的均值
                canshu_social_avg = tf.reduce_mean(canshu_social[:, :3], axis=1)

                # 计算 canshu_social 的前三列的均值乘以 cos(a) 利己
                canshu_social_avg_sin = canshu_social_avg * cos_a

                # 计算 canshu_social 的最后一列乘以 sin(a) 利他
                canshu_social_cos = canshu_social[:, -1] * sin_a

                # 合并两部分结果
                new_values = canshu_social_avg_sin + canshu_social_cos
                self.reward = tf.expand_dims(new_values, axis=1)
                # 将 self.reward 中的每个值都乘以相同的权重
                # weight = 10
                # self.reward = self.reward * weight
                print('判别器内部的输出格式：', 'self.reward:', np.shape(self.reward), type(self.reward))

                # 把社交倾向转化为（batch，1）的shape
                # 将列表转换为张量

                self.social_pre = social_pre
                # print('判别器内部的输出格式：', 'self.social_pre:', np.shape(self.social_pre), type(self.social_pre))

                # self.reward = self.tanh_net(rew_input, dout=1)

            with tf.variable_scope('vfn'):
                self.value_fn_n = self.relu_net(self.nobs, dout=1)
                # self.value_fn_n = self.tanh_net(self.nobs, dout=1)
            with tf.variable_scope('vfn', reuse=True):
                self.value_fn = self.relu_net(self.obs, dout=1)
                # self.value_fn = self.tanh_net(self.obs, dout=1)

            log_q_tau = self.lprobs
            log_p_tau = self.reward + self.gamma * self.value_fn_n - self.value_fn
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)  # 进行对数求和指数运算。
            self.discrim_output = tf.exp(log_p_tau - log_pq)

        self.total_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
        self.var_list = self.get_trainable_variables()
        params = find_trainable_variables(self.scope)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
        self.total_loss += self.l2_loss

        grads = tf.gradients(self.total_loss, params)
        # fisher_loss = -self.total_loss
        # self.d_optim = tf.train.AdamOptimizer(self.lr_rate, beta1=0.5, beta2=0.9).minimize(self.total_loss, var_list=self.var_list)
        with tf.variable_scope(self.scope + '/d_optim'):
            # d_optim = kfac.KfacOptimizer(
            #     learning_rate=self.lr_rate, clip_kl=kfac_clip,
            #     momentum=0.9, kfac_update=1, epsilon=0.01,
            #     stats_decay=0.99, async=0, cold_iter=10,
            #     max_grad_norm=max_grad_norm)
            # update_stats_op = d_optim.compute_and_apply_stats(fisher_loss, var_list=params)
            # train_op, q_runner = d_optim.apply_gradients(list(zip(grads, params)))
            # self.q_runner = q_runner
            d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_rate*0.001)  # 降低学习率
            train_op = d_optim.apply_gradients(list(zip(grads, params)))
            # train_op, qr = d_optim.apply_gradients(list(zip(grads, params)))
        self.d_optim = train_op
        self.saver = tf.train.Saver(self.get_variables())

        self.params_flat = self.get_trainable_variables()

    def relu_net(self, x, layers=2, dout=1, hidden_size=128):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=hidden_size, name='l%d' % i)
        out = linear(out, dout=dout, name='lfinal')
        return out

    def tanh_net(self, x, layers=2, dout=1, hidden_size=128):
        out = x
        for i in range(layers):
            out = tanh_layer(out, dout=hidden_size, name='l%d' % i)
        out = linear(out, dout=dout, name='lfinal')
        return out

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        # print('这里报错了？')
        # print('tf.GraphKeys.TRAINABLE_VARIABLES:',tf.GraphKeys.TRAINABLE_VARIABLES,'self.scope:',self.scope)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    # 这段代码定义了判别器（discriminator）类的 get_reward 方法，该方法用于计算判别器对给定的轨迹数据的奖励信号
    # 输入的是一个agent的观察值,动作,新的观察值,存储路径
    def get_reward(self, obs, acs, obs_next, path_probs, canshu_social_allbatch, discrim_score=False):
        # self.canshu_social_allbatch = canshu_social_allbatch
        # 改这里！！！并且要看一下判别器是如何更新的，奖励是越大越好还是越小越好。self.reward
        if len(obs.shape) == 1: # 检查 obs 是否是一维数组（向量）的形式。如果是，将其扩展为二维数组，以便在后续操作中能够正确处理多个观察数据。
            obs = np.expand_dims(obs, 0) # 如果 obs 是一维数组，使用 np.expand_dims 将其扩展为二维数组，其中第一个维度的大小为1。这是为了确保 obs 可以正确传递给 TensorFlow 计算图。
        if len(acs.shape) == 1: # 类似地，检查 acs 是否是一维数组的形式，如果是，将其扩展为二维数组，以便正确处理多个动作数据。
            acs = np.expand_dims(acs, 0)  # 如果 acs 是一维数组，使用 np.expand_dims 将其扩展为二维数组，其中第一个维度的大小为1。这是为了确保 acs 可以正确传递给 TensorFlow 计算图。
        if discrim_score: # 检查 discrim_score 参数是否为 True。如果为 True，表示需要计算判别器的分数（score），否则需要计算奖励。
            feed_dict = {self.obs: obs,
                         self.act: acs,
                         self.nobs: obs_next,
                         self.lprobs: path_probs}
            scores = self.sess.run(self.discrim_output, feed_dict)
            score = np.log(scores + 1e-20) - np.log(1 - scores + 1e-20)
        else:
            # print('喂进判别器getreward的canshu_social_allbatch：', np.shape(canshu_social_allbatch),type(canshu_social_allbatch))  # (10, 4)
            # if np.shape(obs)[0] == 21:
            #     obs = np.reshape(obs, [-1, 21 * 57])
            #     obs_next = np.reshape(obs_next, [-1, 21 * 57])
            # else:
            # obs = obs
            # obs_next = obs_next
            obs = np.reshape(obs, [-1, 21 * 57])
            obs_next = np.reshape(obs_next, [-1, 21 * 57])
            # print('喂进判别器getreward的obs：', np.shape(obs), type(obs))  # (10, 4)

            feed_dict = {self.obs: obs,
                         self.act: acs,
                         self.canshu_social_allbatch: canshu_social_allbatch}
            score, pre = self.sess.run([self.reward, self.social_pre], feed_dict)
            # print('判别器内部输出的：','score:',np.shape(score),type(score),'pre:',np.shape(pre),type(pre))
        return score, pre

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs, canshu_social_allbatch_array):
        labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
        g_obs_reshape = np.reshape(g_obs, [-1, 21*57])
        g_nobs_reshape = np.reshape(g_nobs, [-1, 21*57])
        e_obs_reshape = np.reshape(e_obs, [-1, 21*57])
        e_nobs_reshape = np.reshape(e_nobs, [-1, 21*57])
        # print('g_obs_reshapes:', np.shape(g_obs_reshape))  # (50, 966)
        # print('e_obs_reshape:', np.shape(e_obs_reshape))  # (50, 966)
        # print('g_nobs_reshape:', np.shape(g_nobs_reshape))  # (50, 966)
        # print('e_nobs_reshape:', np.shape(e_nobs_reshape))  # (50, 966)
        # print('canshu_social_allbatch_array:', np.shape(canshu_social_allbatch_array))  # (batch*2, 4)
        feed_dict = {self.obs: np.concatenate([g_obs_reshape, e_obs_reshape], axis=0),
                     self.act: np.concatenate([g_acs, e_acs], axis=0),
                     self.nobs: np.concatenate([g_nobs_reshape, e_nobs_reshape], axis=0),
                     self.lprobs: np.concatenate([g_probs, e_probs], axis=0),
                     self.labels: labels,
                     self.lr_rate: self.lr.value(),
                     self.canshu_social_allbatch: canshu_social_allbatch_array}
        # print('self.obs:',np.shape(np.concatenate([g_obs, e_obs], axis=0)))  # (100, 21, 46)
        # print('self.obs_reshape:', np.shape(np.concatenate([g_obs_reshape, e_obs_reshape], axis=0)))
        # print('self.act:', np.shape(np.concatenate([g_acs, e_acs], axis=0)))  # (100, 2)
        # print('self.nobs:', np.shape(np.concatenate([g_nobs, e_nobs], axis=0)))  # (100, 21, 46)
        # print('self.nobs_reshape:', np.shape(np.concatenate([g_nobs_reshape, e_nobs_reshape], axis=0)))  #
        # print('self.lprobs:', np.shape(np.concatenate([g_probs, e_probs], axis=0)))  # (100, 1)
        # print('self.labels:', np.shape(labels))  # (100, 1)
        print('self.lr_rate:', self.lr.value())  # 0.09999090909090909
        # print('feed_dict:',feed_dict)
        loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
        print('total_loss:', loss)
        return loss

    def restore(self, path):
        print('restoring from:' + path)
        self.saver.restore(self.sess, path)

    def save(self, save_path):
        ps = self.sess.run(self.params_flat)
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params_flat, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)
