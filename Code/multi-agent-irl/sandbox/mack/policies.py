import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
import rl.common.tf_util as U
from rl.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div #, masked
# from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope

def masked(ac, X):
    for i in range(len(ac)):
        if X[i][0] == 0 or (abs(X[i][4]) <0.01 and abs(X[i][5]) < 0.01) :
            ac[i] = np.zeros(2)
    return ac

class CategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.n
        actions = tf.placeholder(tf.int32, (nbatch))
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
            else:
                a, v = sess.run([pi, v0], {X: ob, X_v: obs})
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
                 nenv, nsteps, nstack, reuse=False, name='model'):

        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0] # 2
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h3 = fc(h2, 'fc3', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            pi = fc(h4, 'pi', nact, act=lambda x: x)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v

            h11 = fc(Y, 'fc11', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h12 = fc(h11, 'fc12', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h13 = fc(h12, 'fc13', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h14 = fc(h13, 'fc14', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            vf = fc(h14, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi + tf.random.normal(tf.shape(pi), 0.0, 0.6) *std

        self.initial_state = []

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

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

class MASKATTGaussianPolicy(object):

    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        step_sizes = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        sequence_length_lstm = len(step_sizes)
        head_dim_lstm = 3

        X = tf.placeholder(tf.float32, ob_shape)
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)

        ob_shape_attention = (nbatch, 21, ob_space.shape[0] * nstack)
        X_attention = tf.placeholder(tf.float32, ob_shape_attention)
        ob_shape_LSTM = (nbatch, 21, 100 * nstack)

        all_ob_shape_LSTM_att = (nbatch, 21*8*ob_space.shape[0] * nstack)
        X_v_LSTM_att = tf.placeholder(tf.float32, all_ob_shape_LSTM_att)

        mask_onetime_shape = (21, nbatch, 10, 10)
        Mask_onetime_all = tf.placeholder(tf.bool, mask_onetime_shape)

        mask_alltime_shape = (nbatch, 21, 21)
        Mask_alltime = tf.placeholder(tf.bool, mask_alltime_shape)


        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            # define self_attention
            def self_attention_gpt(inputs, mask_tensor, num_heads=1, head_dim=3, name_="self_attention"):
                # Linearly project the inputs into queries, keys, and values
                with tf.variable_scope(name_, reuse=tf.AUTO_REUSE):
                    num_units = 128
                    queries = tf.layers.dense(inputs=inputs, units=num_units, name='fc_q')
                    keys = tf.layers.dense(inputs=inputs, units=num_units, name='fc_k')
                    values = tf.layers.dense(inputs=inputs, units=num_units, name='fc_v')
                    padding_val = -2 ** 32
                    scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(8, dtype=tf.float32))  # (batch,sequence,sequence)
                    scores_mask = tf.where(mask_tensor, scores, tf.ones_like(scores) * padding_val) / tf.sqrt(
                        tf.cast(num_units, tf.float32))
                    # Apply Softmax to scores
                    attention_weights = tf.nn.softmax(scores_mask, axis=-1)  # （batch, sequence, sequence）
                    z_value = tf.matmul(attention_weights, values)  # (batch, sequence, num_units)

                    inputs_add = tf.layers.dense(inputs, num_units, activation=None)

                    add_z = tf.add(inputs_add, z_value)  # (batch,seq,num_units)
                    normalized_add_z = tf.contrib.layers.layer_norm(add_z, begin_norm_axis=2)
                    hidden1 = tf.layers.dense(inputs=normalized_add_z, units=num_units,activation=tf.nn.relu, name='att_fc1')
                    hidden2 = tf.layers.dense(inputs=hidden1, units=num_units, activation=tf.nn.relu, name='att_fc2')
                    hidden3 = tf.layers.dense(inputs=hidden2, units=num_units, activation=tf.nn.relu, name='att_fc2')
                    feed_forward_output = tf.layers.dense(inputs=hidden3, units=normalized_add_z.get_shape().as_list()[-1],activation=None, name='att_fc2')
                    feed_forward_output_add = tf.add(normalized_add_z, feed_forward_output)
                    normalized_feed_forward_output_add = tf.contrib.layers.layer_norm(feed_forward_output_add,
                                                                                      begin_norm_axis=2)
                return attention_weights, normalized_feed_forward_output_add


            shape_ = X_attention.get_shape().as_list()
            num_batch = shape_[0]
            num_time = shape_[1]
            num_features = 10
            def process_one_time_step(i):
                X_ONE_TIME = X_attention[:, i, :]
                sub_inputsRag_all = []
                for j in range(num_batch):
                    sub_inputsRag_j = tf.zeros([0, num_features], dtype=tf.float32)
                    current_pos = 0
                    for k, step_size in enumerate(step_sizes):
                        feature_slice = X_ONE_TIME[j, current_pos: current_pos + step_size]
                        if feature_slice.shape[0] < num_features:
                            pad_size = num_features - feature_slice.shape[0]
                            feature_slice = tf.pad(feature_slice, paddings=[[0, pad_size]])
                        sub_inputsRag_j = tf.concat([sub_inputsRag_j, tf.expand_dims(feature_slice, axis=0)], axis=0)
                        current_pos += step_size
                    sub_inputsRag_all.append(sub_inputsRag_j)

                sub_inputsRag = tf.stack(sub_inputsRag_all, axis=0)
                Mask_onetime = Mask_onetime_all[i]
                sub_attention_weights, sub_output_ = self_attention_gpt(sub_inputsRag, mask_tensor=Mask_onetime,
                                                                        num_heads=1, head_dim=head_dim_lstm,
                                                                        name_="self_attention_")
                return sub_output_, sub_inputsRag, sub_attention_weights

            attention_outputs, sub_inputs_new, attention_weights = tf.map_fn(process_one_time_step,
                                                                             tf.range(num_time),
                                                                             dtype=(tf.float32, tf.float32, tf.float32))

            attention_outputs_reshape = tf.transpose(attention_outputs,perm=[1, 0, 2, 3])
            attention_outputs_reshaped = tf.reshape(attention_outputs_reshape,[tf.shape(attention_outputs_reshape)[0], 21, -1])
            sub_attention_weights_time, sub_output_time = self_attention_gpt(attention_outputs_reshaped,
                                                                             mask_tensor=Mask_alltime, num_heads=1,
                                                                             head_dim=head_dim_lstm,
                                                                             name_="self_attention_time")

            sub_output_time_reshaped = tf.reshape(sub_output_time,[tf.shape(attention_outputs_reshape)[0],-1])
            model1_h1 = fc(sub_output_time_reshaped, 'model1_fc1_att_time', nh=128, init_scale=np.sqrt(0),act=tf.nn.sigmoid)
            model1_h2 = fc(model1_h1, 'model1_fc2_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)
            model1_h3 = fc(model1_h2, 'model1_fc3_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)
            model1_h4 = fc(model1_h3, 'model1_fc4_att_time', nh=128, init_scale=np.sqrt(0), act=tf.nn.tanh)
            model1_h5 = fc(model1_h4, 'model1_fc5_att_time', nh=128, init_scale=np.sqrt(0),act=tf.nn.tanh)
            pi = fc(model1_h5, 'pi', nact, act=lambda x: x)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v_LSTM_att, A_v], axis=1)
            else:
                Y = X_v_LSTM_att
            h11 = fc(Y, 'fc11', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h12 = fc(h11, 'fc12', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h13 = fc(h12, 'fc13', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            h14 = fc(h13, 'fc14', nh=256, init_scale=np.sqrt(0), act=tf.nn.tanh)
            vf = fc(h14, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi + tf.random_normal(tf.shape(pi), 0.0, 0.6) *std
        attention_weights_output_spatial = attention_weights
        attention_weights_output_temporal = sub_attention_weights_time

        self.initial_state = []

        def step(ob_attention, ob, obs, a_v, is_training, mask_atime, mask_times, *_args, **_kwargs):
            if is_training == True:
                obs_flat = obs.reshape(ob_attention.shape[0], -1)
            else:
                obs_flat = obs.reshape(1, -1)
            if a_v is not None:
                # print('nbatch:',nbatch,'policy里的输入的ob_attention为:',np.shape(ob_attention))
                # print('nbatch:', nbatch, 'policy里的输入的ob为:', np.shape(ob))
                # print('nbatch:', nbatch, 'policy里的输入的obs为:', np.shape(obs), np.shape(obs_flat))
                # print('nbatch:', nbatch, 'policy里的输入的a_v为:', np.shape(a_v))
                # print('nbatch:', nbatch, 'policy里的输入的mask_atime为:', np.shape(mask_atime))
                # print('nbatch:', nbatch, 'policy里的输入的mask_times为:', np.shape(mask_times))
                a, v, att_weights_output_spatial, att_weights_output_temporal = sess.run([a0, v0, attention_weights_output_spatial, attention_weights_output_temporal], {X_attention: ob_attention, X_v_LSTM_att: obs_flat,
                                                                                         A_v: a_v, Mask_onetime_all: mask_atime, Mask_alltime: mask_times})
            else:
                a, v, att_weights_output_spatial, att_weights_output_temporal = sess.run([a0, v0, attention_weights_output_spatial, attention_weights_output_temporal], {X_attention: ob_attention, X_v_LSTM_att: obs_flat})

            return a, v, [], att_weights_output_spatial, att_weights_output_temporal

        def value(ob, a_v, *_args, **_kwargs):
            ob_flat = ob.reshape(10, -1)
            if a_v is not None:
                return sess.run(v0, {X_v_LSTM_att: ob_flat, A_v: a_v})
            else:
                return sess.run(v0, {X_v_LSTM_att: ob_flat})

        self.X_attention = X_attention
        self.X_v_LSTM_att = X_v_LSTM_att
        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.a0 = a0
        self.vf = vf
        self.step = step
        self.value = value
        self.attention_weights_output_spatial = attention_weights_output_spatial
        self.attention_weights_output_temporal = attention_weights_output_temporal
        self.Mask_onetime_all = Mask_onetime_all
        self.Mask_alltime = Mask_alltime

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