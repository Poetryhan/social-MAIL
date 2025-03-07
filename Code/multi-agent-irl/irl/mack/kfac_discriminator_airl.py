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
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size
        self.type_ = tf.placeholder(tf.string, shape=[])
        self.batch_num = tf.placeholder(tf.int32, shape=[])
        self.canshu_social_allbatch = tf.placeholder(tf.float32, (None, 4))
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

        self.lr_rate = tf.placeholder(tf.float32, ())

        with tf.variable_scope(self.scope):
            rew_input = self.obs
            canshu_social = self.canshu_social_allbatch
            if not self.state_only:
                rew_input = tf.concat([self.obs, self.act], axis=1)

            with tf.variable_scope('reward'):
                input_layer = rew_input
                hidden_layer1 = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.leaky_relu)
                hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=100, activation=tf.nn.elu)
                hidden_layer3 = tf.layers.dense(inputs=hidden_layer2, units=100, activation=tf.nn.selu)
                social_pre = tf.layers.dense(inputs=hidden_layer3, units=1, activation=tf.nn.tanh)
                social_pre = tf.multiply(social_pre, tf.constant(np.pi / 2, dtype=tf.float32))
                social_pre_values = tf.squeeze(social_pre, axis=1)
                cos_a = tf.cos(social_pre_values)
                sin_a = tf.sin(social_pre_values)
                canshu_social_avg = tf.reduce_mean(canshu_social[:, :3], axis=1)
                canshu_social_avg_sin = canshu_social_avg * cos_a
                canshu_social_cos = canshu_social[:, -1] * sin_a

                new_values = canshu_social_avg_sin + canshu_social_cos
                self.reward = tf.expand_dims(new_values, axis=1)
                self.social_pre = social_pre

            with tf.variable_scope('vfn'):
                self.value_fn_n = self.relu_net(self.nobs, dout=1)
            with tf.variable_scope('vfn', reuse=True):
                self.value_fn = self.relu_net(self.obs, dout=1)

            log_q_tau = self.lprobs
            log_p_tau = self.reward + self.gamma * self.value_fn_n - self.value_fn
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)


        self.total_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
        self.var_list = self.get_trainable_variables()
        params = find_trainable_variables(self.scope)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
        self.total_loss += self.l2_loss

        grads = tf.gradients(self.total_loss, params)
        with tf.variable_scope(self.scope + '/d_optim'):
            d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_rate*0.001)
            train_op, qr = d_optim.apply_gradients(list(zip(grads, params)))
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
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs, obs_next, path_probs, canshu_social_allbatch, discrim_score=False):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        if discrim_score:
            feed_dict = {self.obs: obs,
                         self.act: acs,
                         self.nobs: obs_next,
                         self.lprobs: path_probs}
            scores = self.sess.run(self.discrim_output, feed_dict)
            score = np.log(scores + 1e-20) - np.log(1 - scores + 1e-20)
        else:
            obs = np.reshape(obs, [-1, 21 * 57])
            obs_next = np.reshape(obs_next, [-1, 21 * 57])
            feed_dict = {self.obs: obs,
                         self.act: acs,
                         self.canshu_social_allbatch: canshu_social_allbatch}
            score, pre = self.sess.run([self.reward, self.social_pre], feed_dict)
        return score, pre

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs, canshu_social_allbatch_array):
        labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
        g_obs_reshape = np.reshape(g_obs, [-1, 21*57])
        g_nobs_reshape = np.reshape(g_nobs, [-1, 21*57])
        e_obs_reshape = np.reshape(e_obs, [-1, 21*57])
        e_nobs_reshape = np.reshape(e_nobs, [-1, 21*57])
        feed_dict = {self.obs: np.concatenate([g_obs_reshape, e_obs_reshape], axis=0),
                     self.act: np.concatenate([g_acs, e_acs], axis=0),
                     self.nobs: np.concatenate([g_nobs_reshape, e_nobs_reshape], axis=0),
                     self.lprobs: np.concatenate([g_probs, e_probs], axis=0),
                     self.labels: labels,
                     self.lr_rate: self.lr.value(),
                     self.canshu_social_allbatch: canshu_social_allbatch_array}
        loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
        return loss

    def restore(self, path):
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
