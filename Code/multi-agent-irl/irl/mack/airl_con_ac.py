import os.path as osp
import random
import time

import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from rl.acktr.utils import Scheduler, find_trainable_variables, discount_with_dones
from rl.acktr.utils import cat_entropy, mse, onehot, multionehot

from rl import logger
from rl.acktr import kfac
from rl.common import set_global_seeds, explained_variance
from irl.mack.kfac_discriminator_airl import Discriminator
# from irl.mack.kfac_discriminator_wgan import Discriminator
from irl.dataset import Dset
import math
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


n_ac = 2

class Model():
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = False
        self.sess = sess = tf.Session(config=config)
        nbatch = nenvs * nsteps
        self.num_agents = num_agents = len(ob_space)
        if identical is None:
            identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]

        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents

        A, ADV, R, PG_LR = [], [], [], []
        for k in range(num_agents):
            if identical[k]:
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
            else:
                A.append(tf.placeholder(tf.float32, [nbatch * scale[k], n_ac]))
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, []))

        pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = []
        self.model2 = train_model = []
        self.pg_fisher = pg_fisher_loss = []
        self.logits = logits = []
        sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = []
        self.lld = lld = []
        self.log_pac = []

        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1])
            else:
                step_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                         nenvs, 1, nstack, reuse=False, name='%d' % k))
                train_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                          nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))

            logpac = (tf.reduce_mean(mse(train_model[k].a0, A[k]),1))
            self.log_pac.append(-logpac)

            lld.append(tf.reduce_mean(logpac))
            logits.append(train_model[k].a0)

            pg_loss.append(tf.reduce_mean(ADV[k] * logpac))
            entropy.append(tf.constant([ 0.01]))
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k]
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k])))
            train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])

            pg_fisher_loss.append(-tf.reduce_mean(logpac))
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf)))
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2)))
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k])

        self.policy_params = []
        self.value_params = []

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))

        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]

        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ]

        self.optim = optim = []
        self.clones = clones = []
        update_stats_op = []
        train_op, clone_op, q_runner = [], [], []

        for k in range(num_agents):
            if identical[k]:
                optim.append(optim[-1])
                train_op.append(train_op[-1])
                q_runner.append(q_runner[-1])
                clones.append(clones[-1])
                clone_op.append(clone_op[-1])
            else:
                with tf.variable_scope('optim_%d' % k):
                    optim.append(tf.train.AdamOptimizer(learning_rate=PG_LR[k]*0.001, Async=0))
                    train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                    train_op.append(train_op_)
                    q_runner.append(q_runner_)

                with tf.variable_scope('clone_%d' % k):
                    clones.append(tf.train.AdamOptimizer(learning_rate=PG_LR[k]*0.01, Async=0))
                    clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    clone_op.append(clone_op_)

        update_stats_op = tf.group(*update_stats_op)
        train_ops = train_op
        clone_ops = clone_op
        train_op = tf.group(*train_op)
        clone_op = tf.group(*clone_op)

        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = [rewards[k] - values[k] for k in range(num_agents)]
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1)
            td_map = {}
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = []
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([actions[i]
                                                   for i in range(num_agents) if i != k], axis=1))

                    action_v = np.concatenate(action_v, axis=0)
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})

                X_attention_train = np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0)  # (batch,21,46)
                batch_train = X_attention_train.shape[0]
                time_train = X_attention_train.shape[1]
                train_trj_go_step = np.empty((batch_train, 1))
                num_features = 10
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all_train = []
                mask_times_train = np.ones([batch_train, 21, 21], dtype=bool)
                for time_i in range(time_train):
                    X_attention_ONE_TIME_train = X_attention_train[:, time_i, :]
                    sub_inputsRag_all_np_train = []
                    for j in range(batch_train):
                        sub_inputsRag_j_np_train = np.zeros([0, num_features], dtype=np.float32)
                        current_pos_np_train = 0
                        for k_, step_size in enumerate(step_sizes_np):
                            feature_slice = X_attention_ONE_TIME_train[j,
                                            current_pos_np_train: current_pos_np_train + step_size]
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            sub_inputsRag_j_np_train = np.concatenate(
                                [sub_inputsRag_j_np_train, np.expand_dims(feature_slice, axis=0)], axis=0)
                            current_pos_np_train += step_size
                        sub_inputsRag_all_np_train.append(sub_inputsRag_j_np_train)

                    sub_inputsRag_np_train = np.stack(sub_inputsRag_all_np_train, axis=0)
                    mask_atime_train = np.ones([batch_train, 10, 10], dtype=bool)
                    for j_mask in range(batch_train):
                        for i_mask in range(10):
                            if sub_inputsRag_np_train[j_mask, i_mask, 0] == 0:
                                mask_atime_train[j_mask, i_mask, :] = False
                                mask_atime_train[j_mask, :, i_mask] = False
                    mask_atime_all_train.append(mask_atime_train)
                mask_atime_all_new_train = np.stack(mask_atime_all_train, axis=0)

                for i_batch in range(batch_train):
                    if X_attention_train[i_batch][20][0] == 0:
                        mask_times_train[i_batch, :, :] = False
                    else:
                        for time_i_batch in range(time_train):
                            if X_attention_train[i_batch][time_i_batch][0] != 0:
                                mask_times_train[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break


                new_map.update({
                    train_model[k].X_attention: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].X_v_LSTM_att: np.concatenate([ob.copy().reshape(len(ob),-1) for j in range(k, pointer[k])], axis=0),
                    train_model[k].Mask_onetime_all: mask_atime_all_new_train,
                    train_model[k].Mask_alltime: mask_times_train,
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    ADV[k]: np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0),
                    R[k]: np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })

                sess.run(train_ops[k], feed_dict=new_map)
                td_map.update(new_map)

                if states[k] != []:
                    td_map[train_model[k].S] = states
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy = sess.run(
                [pg_loss, vf_loss, entropy],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            td_map = {}
            cur_lr = self.clone_lr.value()
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(clone_ops[k], feed_dict=new_map)
                td_map.update(new_map)
            lld_loss = sess.run([lld], td_map)
            return lld_loss

        def get_log_action_prob(obs, actions):
            action_prob = []
            for k in range(num_agents):
                if identical[k]:
                    continue

                is_training = True
                X_attention = np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0)
                batch_get = X_attention.shape[0]
                time_get = X_attention.shape[1]
                get_trj_go_step = np.empty((batch_get, 1))
                num_features = 10
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all_get = []
                mask_times_get = np.ones([batch_get, 21, 21],
                                         dtype=bool)
                for time_i in range(time_get):
                    X_attention_ONE_TIME = X_attention[:, time_i, :]
                    sub_inputsRag_all_np_get = []
                    for j in range(batch_get):
                        sub_inputsRag_j_np_get = np.zeros([0, num_features], dtype=np.float32)
                        current_pos_np_get = 0
                        for k_, step_size in enumerate(step_sizes_np):
                            feature_slice = X_attention_ONE_TIME[j, current_pos_np_get: current_pos_np_get + step_size]
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            sub_inputsRag_j_np_get = np.concatenate(
                                [sub_inputsRag_j_np_get, np.expand_dims(feature_slice, axis=0)], axis=0)
                            current_pos_np_get += step_size
                        sub_inputsRag_all_np_get.append(sub_inputsRag_j_np_get)

                    sub_inputsRag_np_get = np.stack(sub_inputsRag_all_np_get, axis=0)
                    mask_atime_get = np.ones([batch_get, 10, 10], dtype=bool)
                    for j_mask in range(batch_get):
                        for i_mask in range(10):
                            if sub_inputsRag_np_get[j_mask, i_mask, 0] == 0:
                                mask_atime_get[j_mask, i_mask, :] = False
                                mask_atime_get[j_mask, :, i_mask] = False
                    mask_atime_all_get.append(mask_atime_get)
                mask_atime_all_new_get = np.stack(mask_atime_all_get, axis=0)

                for i_batch in range(batch_get):
                    if X_attention[i_batch][20][0] == 0:
                        mask_times_get[i_batch, :, :] = False
                    else:
                        for time_i_batch in range(time_get):
                            if X_attention[i_batch][time_i_batch][0] != 0:
                                mask_times_get[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break

                new_map = {
                    train_model[k].X_attention: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].Mask_onetime_all: mask_atime_all_new_get,
                    train_model[k].Mask_alltime: mask_times_get,
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)
                }
                log_pac = sess.run(self.log_pac[k], feed_dict=new_map)
                if scale[k] == 1:
                    action_prob.append(log_pac)
                else:
                    log_pac = np.split(log_pac, scale[k], axis=0)
                    action_prob += log_pac
            return action_prob

        self.get_log_action_prob = get_log_action_prob

        def get_log_action_prob_step(obs, actions):
            action_prob = []
            for k in range(num_agents):
                action_prob.append(step_model[k].step_log_prob(obs[k], actions[k]))
            return action_prob

        self.get_log_action_prob_step = get_log_action_prob_step

        def save(save_path):
            ps = sess.run(params_flat)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params_flat, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob_lstm, ob, av, *_args, **_kwargs):
            a, v, s, att_weights_spatial, att_weights_temporal = [], [], [], [], []
            obs = np.concatenate(ob, axis=1)
            obs_lstm = np.concatenate(ob_lstm, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([av[i]
                                      for i in range(num_agents) if i != k], axis=1)
                is_training = False
                k_ob_lstm = ob_lstm[k]
                num_batch = k_ob_lstm.shape[0]
                num_time = k_ob_lstm.shape[1]
                num_features = 10
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all = []
                mask_times = np.ones([num_batch, 21, 21], dtype=bool)
                for time_i in range(num_time):
                    k_ob_lstm_ONE_TIME = k_ob_lstm[:, time_i, :]
                    sub_inputsRag_all_np = []
                    for j in range(num_batch):
                        sub_inputsRag_j_np = np.zeros([0, num_features], dtype=np.float32)
                        current_pos_np = 0
                        for k_, step_size in enumerate(step_sizes_np):
                            feature_slice = k_ob_lstm_ONE_TIME[j, current_pos_np: current_pos_np + step_size]
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            sub_inputsRag_j_np = np.concatenate(
                                [sub_inputsRag_j_np, np.expand_dims(feature_slice, axis=0)], axis=0)
                            current_pos_np += step_size
                        sub_inputsRag_all_np.append(sub_inputsRag_j_np)

                    sub_inputsRag_np = np.stack(sub_inputsRag_all_np, axis=0)
                    mask_atime = np.ones([num_batch, 10, 10], dtype=bool)
                    for j_mask in range(num_batch):
                        for i_mask in range(10):
                            if sub_inputsRag_np[j_mask, i_mask, 0] == 0:
                                mask_atime[j_mask, i_mask, :] = False
                                mask_atime[j_mask, :, i_mask] = False

                    mask_atime_all.append(mask_atime)
                mask_atime_all_new = np.stack(mask_atime_all, axis=0)

                for i_batch in range(num_batch):
                    if k_ob_lstm[i_batch][20][0] == 0:
                        mask_times[i_batch, :, :] = False
                    else:
                        for time_i_batch in range(num_time):
                            if k_ob_lstm[i_batch][time_i_batch][0] != 0:
                                mask_times[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break

                a_, v_, s_, att_weights_spatial_, att_weights_temporal_ = step_model[k].step(ob_lstm[k], ob[k], obs_lstm, a_v, is_training,
                                                                                    mask_atime_all_new, mask_times) # 输入的是K agent的观察值, 所有agent的观察值,以及其他所有agent的动作
                a.append(a_)
                v.append(v_)
                s.append(s_)
                att_weights_spatial.append(att_weights_spatial_)
                att_weights_temporal.append(att_weights_temporal_)
            return a, v, s, att_weights_spatial, att_weights_temporal

        def attention_step(sv, obs_lstm_sv_t, state, obs_lstm, a_v, is_training, mask_atime_all_new, mask_times):
            a_, v_, s_, att_weights_spatial_, att_weights_temporal_ = step_model[sv].step(obs_lstm_sv_t, state, obs_lstm, a_v, is_training, mask_atime_all_new, mask_times)  # 输入的是K agent的观察值, 所有agent的观察值,以及其他所有agent的动作
            return a_, v_, s_, att_weights_spatial_, att_weights_temporal_
        self.step = step
        self.attention_step = attention_step

        def value(obs, av):
            v = []
            ob = np.concatenate(obs, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([av[i]
                                      for i in range(num_agents) if i != k], axis=1)
                v_ = step_model[k].value(ob, a_v)
                v.append(v_)
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type, nobs_flag=False):
        self.env = env

        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.nobs_flag = nobs_flag
        self.num_agents = len(env.observation_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.batch_ob_shape_lstm = [
            (nenv * nsteps, 21, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.batch_ac_shape = [
            (nenv * nsteps, nstack * env.action_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.obs_lstm = [
            np.zeros((nenv, 21, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((nenv, n_ac )) for _ in range(self.num_agents)]

        obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm = env.reset()

        self.update_obs(obs)
        self.update_obs_lstm(obs_lstm)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [np.array([False for _ in range(self.nenv)]) for k in range(self.num_agents)]

        self.ini_steps = ini_step_n
        self.ini_obs = ini_obs
        self.ini_obs_lstm = ini_obs_lstm
        self.diedai_cishu = 0
        self.reset_infos = reset_infos
        self.N = [np.array([0 for _ in range(nenv)])]
        self.env_GO_STEP = [np.array([0]) for k in range(10)]
        self.trj_GO_STEP = [np.array([0 for _ in range(self.num_agents)]) for k in
                            range(self.nenv)]

        self.update = 0

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs

    def update_obs_lstm(self, obs_lstm):
        # TODO: Potentially useful for stacking.
        self.obs_lstm = obs_lstm


    def run(self, update):
        mb_obs_lstm = [[] for _ in range(self.num_agents)]
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_obs_next_lstm = [[] for _ in range(self.num_agents)]
        mb_obs_next = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_report_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states


        for n in range(self.nsteps):

            ini_update_inf = [np.array([0 for _ in range(self.num_agents)]) for k in range(self.nenv)]
            for k in range(self.num_agents):
                for i in range(self.nenv):
                    if self.reset_infos[0][i][0] == True:
                        self.N[0][i] = n + (update - 1) * self.nsteps

                    ini_step_k = self.ini_steps[k][i][0]
                    if self.ini_obs[k][i][0] == 0:
                        self.obs[k][i] = np.zeros(57)
                        self.actions[k][i] = np.zeros(2)
                        self.obs_lstm[k][i][:] = 0
                        ini_update_inf[i][k] = False
                    else:
                        if n + (update - 1) * self.nsteps - self.N[0][i] < ini_step_k:
                            self.obs[k][i] = np.zeros(57)
                            self.actions[k][i] = np.zeros(2)
                            self.obs_lstm[k][i][:] = 0
                            ini_update_inf[i][k] = False
                        elif n + (update - 1) * self.nsteps - self.N[0][i] == ini_step_k:

                            ini_update_inf[i][k] = True
                            self.obs[k][i] = self.ini_obs[k][i]
                            self.actions[k][i] = np.zeros(2)
                            self.obs_lstm[k][i] = self.ini_obs_lstm[k][i]
                        else:
                            ini_update_inf[i][k] = False


            ini_obs_old_list = []

            for i in range(self.nenv):
                ini_obs_old_list.append(
                    [np.concatenate((self.actions[k_][i], np.array([self.env_GO_STEP[i][0]]),
                                     np.array([self.trj_GO_STEP[i][k_]]),
                                     np.array([self.ini_steps[k_][i][0]]),
                                     self.ini_obs[k_][i], np.array([ini_update_inf[i][k_]]))) for k_ in range(self.num_agents)])

            obs_lstm_nowstep, obs_nowstep = self.env.ini_obs_update(ini_obs_old_list)

            for i in range(self.nenv):
                for k in range(self.num_agents):
                    if ini_update_inf[i][k] == True:
                        for j in range(len(self.obs_lstm)):
                            self.obs_lstm[j][i] = obs_lstm_nowstep[j][i]
                            self.obs[j][i] = obs_nowstep[j][i]
                        self.actions[k][i] = np.zeros(2)

            actions, values, states, atten_weights_spatial, atten_weights_temporal = self.model.step(self.obs_lstm, self.obs, self.actions)

            self.actions = actions

            for k in range(self.num_agents):
                mb_obs_lstm[k].append(np.copy(self.obs_lstm[k]))
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])

            actions_list = []

            for i in range(self.nenv):
                env_go_step = n + (update - 1) * self.nsteps - self.N[0][i]
                self.env_GO_STEP[i][0] = env_go_step
                for k in range(self.num_agents):
                    trj_go_step = n + (update - 1) * self.nsteps - self.N[0][i] - self.ini_steps[k][i][0]
                    if trj_go_step > 0:
                        self.trj_GO_STEP[i][k] = trj_go_step
                    if trj_go_step <= 0:
                        self.trj_GO_STEP[i][k] = trj_go_step

                actions_list.append([np.concatenate((self.actions[k][i], np.array([env_go_step]),
                                                     np.array([self.trj_GO_STEP[i][k]]),
                                                     np.array([self.ini_steps[k][i][0]]),
                                                     self.ini_obs[k][i])) for k in
                                     range(self.num_agents)])

            obs_lstm, obs, true_rewards, dones, _, ini_steps_all, ini_obs, reset_infos, ini_obs_lstm, actions_new = self.env.step(actions_list)

            self.actions = actions_new

            self.reset_infos = reset_infos
            self.dones = dones
            re_obs = self.obs
            re_obs_lstm = self.obs_lstm
            re_actions = self.actions
            self.update_obs(obs)
            self.update_obs_lstm(obs_lstm)

            for i in range(self.nenv):
                for k in range(self.num_agents):

                    if reset_infos[0][i][0] == True:
                        self.ini_obs[k][i] = ini_obs[k][i]
                        self.ini_obs_lstm[k][i] = ini_obs_lstm[k][i]
                        obs[k][i] = obs[k][i] * 0.0
                        obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                        self.actions[k][i] = self.actions[k][i] * 0.0
                    else:
                        if dones[k][i] == True:
                            obs[k][i] = obs[k][i] * 0.0
                            obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                            self.obs[k][i] = obs[k][i] * 0.0
                            self.obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                            self.actions[k][i] = self.actions[k][i] * 0.0
                            # true_rewards[k][i] = true_rewards[k][i] * 0

            for k in range(self.num_agents):
                mb_actions[k].append(self.actions[k])

            self.ini_steps = ini_steps_all

            for k in range(self.num_agents):
                mb_obs_next[k].append(np.copy(obs[k]))
                mb_obs_next_lstm[k].append(np.copy(obs_lstm[k]))
            re_obs_next_lstm = obs_lstm
            re_obs_next = obs
            re_path_prob = np.zeros(self.num_agents)


            # get reward from discriminator
            def calculate_rectangle_vertices(center_x, center_y, length, width, angle):
                angle_rad = np.radians(angle)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)

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

                def orientation(p, q, r):
                    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                    if val == 0:
                        return 0
                    return 1 if val > 0 else 2

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

                    if o1 != o2 and o3 != o4:
                        return True

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
                       Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                       Jiaohu_direction):

                dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
                if dis_between_agent_jiaohu <= 15:
                    agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                    neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

                    veh_length = 5
                    veh_width = 2

                    a_agent = math.tan(np.radians(Agent_angle_last))
                    a_neig = math.tan(np.radians(Jiaohu_angle_last))

                    b_agent = (Agent_y) - a_agent * (Agent_x)
                    b_neig = (Jiaohu_y) - a_neig * (Jiaohu_x)

                    vertices_agent_now = calculate_rectangle_vertices(Agent_x, Agent_y, veh_length, veh_width,
                                                                      Agent_angle_last)
                    vertices_neig_now = calculate_rectangle_vertices(Jiaohu_x, Jiaohu_y, veh_length, veh_width,
                                                                     Jiaohu_angle_last)

                    intersect_now = check_intersection(vertices_agent_now, vertices_neig_now)
                    if intersect_now == True:
                        GT_value = 0
                    else:
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
                        elif Agent_angle_last == 270:
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
                                elif Agent_angle_last == 270:
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
                                elif Jiaohu_angle_last == 270:
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

                            GT_value = None
                else:
                    GT_value = None
                return GT_value

            if self.disc_type == 'decentralized':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    if k <= 2:
                        direction_agent = 'left'
                    else:
                        direction_agent = 'straight'
                    batch_num = np.shape(re_obs_lstm[k])[0]
                    rew_input_fuyuan = re_obs_lstm[k]
                    rew_social_allbatch = []
                    for i_batch in range(batch_num):
                        if rew_input_fuyuan[i_batch][20][0] != 0:
                            use_GT = []
                            pianyi_distance = rew_input_fuyuan[i_batch][20][-2]
                            agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                            agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                            agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                            agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                            agent_angle_last = rew_input_fuyuan[i_batch][20][6] * 191 - 1

                            agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                            for agent_k_ in range(self.num_agents):
                                if agent_k_ != k:
                                    if agent_k_ <= 2:
                                        direction_jiaohu = 'left'
                                    else:
                                        direction_jiaohu = 'straight'

                                    rew_input_fuyuan_agent_k_ = re_obs_lstm[agent_k_]

                                    if rew_input_fuyuan_agent_k_[i_batch][20][0] != 0:
                                        jiaohu_agent_x = rew_input_fuyuan_agent_k_[i_batch][20][0] * 38 - 4
                                        jiaohu_agent_y = rew_input_fuyuan_agent_k_[i_batch][20][1] * 23 + 14
                                        jiaohu_agent_vx = rew_input_fuyuan_agent_k_[i_batch][20][2] * 21 - 14
                                        jiaohu_agent_vy = rew_input_fuyuan_agent_k_[i_batch][20][3] * 12 - 2
                                        jiaohu_agent_angle_last = rew_input_fuyuan_agent_k_[i_batch][20][6] * 191 - 1

                                        jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                agent_angle_last, direction_agent,
                                                                                jiaohu_agent_x, jiaohu_agent_y,
                                                                                jiaohu_agent_vx, jiaohu_agent_vy,
                                                                                jiaohu_agent_angle_last,
                                                                                direction_jiaohu)
                                        use_GT.append(jiaohu_agent_GT_value)
                                    else:
                                        jiaohu_agent_x = -4
                                        jiaohu_agent_y = 14
                                        jiaohu_agentk_vx = -14
                                        jiaohu_agent_vy = -2
                                        jiaohu_agent_angle_last = -1
                                        jiaohu_agent_GT_value = None
                                        use_GT.append(jiaohu_agent_GT_value)

                            # landmark in left
                            if rew_input_fuyuan[i_batch][20][38] != 0:
                                delta_left_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][38] * 29 - 14
                                delta_left_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][39] * 30 - 15
                                delta_left_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][40] * 35 - 21
                                delta_left_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][41] * 16 - 5
                                left_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][53] * 360 - 90
                                left_jiaohu_landmark_x = agent_x - delta_left_jiaohu_landmark_x
                                left_jiaohu_landmark_y = agent_y - delta_left_jiaohu_landmark_y
                                left_jiaohu_landmark_vx = agent_vx - delta_left_jiaohu_landmark_vx
                                left_jiaohu_landmark_vy = agent_vy - delta_left_jiaohu_landmark_vy
                                left_jiaohu_landmark_angle_last = left_jiaohu_landmark_angle_last
                            else:
                                left_jiaohu_landmark_x = -5
                                left_jiaohu_landmark_y = -3
                                left_jiaohu_landmark_vx = -16
                                left_jiaohu_landmark_vy = -10
                                left_jiaohu_landmark_angle_last = -90

                            # landmark in right
                            if rew_input_fuyuan[i_batch][20][42] != 0:
                                delta_right_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][42] * 35 - 15
                                delta_right_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][43] * 29 - 15
                                delta_right_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][44] * 25 - 14
                                delta_right_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][45] * 17 - 7
                                right_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][54] * 360 - 90
                                right_jiaohu_landmark_x = agent_x - delta_right_jiaohu_landmark_x
                                right_jiaohu_landmark_y = agent_y - delta_right_jiaohu_landmark_y
                                right_jiaohu_landmark_vx = agent_vx - delta_right_jiaohu_landmark_vx
                                right_jiaohu_landmark_vy = agent_vy - delta_right_jiaohu_landmark_vy
                                right_jiaohu_landmark_angle_last = right_jiaohu_landmark_angle_last
                            else:
                                right_jiaohu_landmark_x = -5
                                right_jiaohu_landmark_y = -3
                                right_jiaohu_landmark_vx = -16
                                right_jiaohu_landmark_vy = -10
                                right_jiaohu_landmark_angle_last = -90

                            if left_jiaohu_landmark_x != -5:
                                direction_landmark = 'landmark'
                                left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                agent_angle_last, direction_agent,
                                                                                left_jiaohu_landmark_x,
                                                                                left_jiaohu_landmark_y,
                                                                                left_jiaohu_landmark_vx,
                                                                                left_jiaohu_landmark_vy,
                                                                                left_jiaohu_landmark_angle_last,
                                                                                direction_landmark)

                                use_GT.append(left_jiaohu_landmark_GT_value)
                            else:
                                left_jiaohu_landmark_GT_value = None
                                use_GT.append(left_jiaohu_landmark_GT_value)

                            if right_jiaohu_landmark_x != -5:
                                direction_landmark = 'landmark'
                                right_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                 agent_angle_last, direction_agent,
                                                                                 right_jiaohu_landmark_x,
                                                                                 right_jiaohu_landmark_y,
                                                                                 right_jiaohu_landmark_vx,
                                                                                 right_jiaohu_landmark_vy,
                                                                                 right_jiaohu_landmark_angle_last,
                                                                                 direction_landmark)

                                use_GT.append(right_jiaohu_landmark_GT_value)

                            else:
                                right_jiaohu_landmark_GT_value = None
                                use_GT.append(right_jiaohu_landmark_GT_value)

                            penalty = 1
                            delta_angle_last1 = rew_input_fuyuan[i_batch][20][56]
                            # comfort
                            comfort_adj = 0
                            if direction_agent == 'left':
                                left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                                left_delta_angle_last1_realmean = 1.085
                                left_delta_angle_last1_realstd = 0.702
                                if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                                    comfort_adj = 0

                                else:
                                    dis_left_delta_angle_last1 = abs(
                                        left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                                    if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(
                                                dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty

                            else:
                                right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                                right_delta_angle_last1_realmean = 0.001
                                right_delta_angle_last1_realstd = 0.076
                                if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                                    comfort_adj = 0

                                else:
                                    dis_right_delta_angle_last1 = abs(right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                                    if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty

                            # efficiency
                            rew_avespeed = agent_v / 6.8  # 85th expert speed
                            # lane shift
                            rew_lane_pianyi = pianyi_distance

                            # GT(PET)
                            use_GT_list_0 = [x for x in use_GT if x is not None]
                            use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                            rew_minGT_mapped = 0
                            if len(use_GT_list) != 0:
                                rew_minGT = sum(use_GT_list) / len(use_GT_list)
                                if rew_minGT <= 1.5:
                                    normalized_data = (rew_minGT - 0) / (1.5 - 0)
                                    rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                                elif 1.5 < rew_minGT < 3:
                                    normalized_data = (rew_minGT - 1.5) / (3 - 1.5)
                                    rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                                elif 3 <= rew_minGT <= 4:
                                    normalized_data = (rew_minGT - 3) / (4 - 3)
                                    rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                                elif rew_minGT > 4:
                                    normalized_data = np.exp(-(1 / (rew_minGT - 4)))
                                    rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                            else:
                                rew_minGT_mapped = 0

                            rew_social_allbatch.append([10*rew_avespeed, -10*rew_lane_pianyi, 5*comfort_adj, 10*rew_minGT_mapped])
                        else:
                            rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])

                    canshu_social_allbatch_array = np.array(rew_social_allbatch)

                    score, pre = self.discriminator[k].get_reward(re_obs_lstm[k],  # re_obs[k]
                                                     re_actions[k],
                                                     re_obs_next_lstm[k],
                                                     re_path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)

                    rewards.append(np.squeeze(score)) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                    score_report, pre_report = self.discriminator[k].get_reward(re_obs_lstm[k],
                                                     re_actions[k],
                                                     re_obs_next_lstm[k],
                                                     re_path_prob[k], canshu_social_allbatch_array,
                                                     discrim_score=False)
                    report_rewards.append(np.squeeze(score_report))

            elif self.disc_type == 'decentralized-all':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs_lstm, axis=1),np.concatenate(re_actions_onehot, axis=1),np.concatenate(re_obs_next_lstm, axis=1),re_path_prob[k],discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs_lstm, axis=1),
                                                                               np.concatenate(re_actions_onehot, axis=1),
                                                                               np.concatenate(re_obs_next_lstm, axis=1),
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))
            else:
                assert False

            for k in range(self.num_agents):
                mb_report_rewards[k].append(report_rewards[k])

            self.states = states


            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])
                mb_rewards[k].append(rewards[k] + true_rewards[k])  #  + true_rewards[k]

        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])
        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs_lstm[k] = np.asarray(mb_obs_lstm[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape_lstm[k])
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_obs_next_lstm[k] = np.asarray(mb_obs_next_lstm[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape_lstm[k])
            mb_obs_next[k] = np.asarray(mb_obs_next[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_report_rewards[k] = np.asarray(mb_report_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ac_shape[k])
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_report_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs_lstm, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, report_rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_report_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                report_rewards = report_rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    report_rewards = discount_with_dones(report_rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    report_rewards = discount_with_dones(report_rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_report_returns[k][n] = report_rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_report_returns[k] = mb_report_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()

        mh_actions = [mb_actions[k] for k in range(self.num_agents)]
        mb_all_obs_lstm = np.concatenate(mb_obs_lstm, axis=1)
        mb_all_nobs_lstm = np.concatenate(mb_obs_next_lstm, axis=1)
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mb_all_nobs = np.concatenate(mb_obs_next, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)

        if self.nobs_flag:
            return mb_obs_lstm, mb_obs_next_lstm, mb_obs, mb_obs_next, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions, \
                   mb_values, mb_all_obs, mb_all_nobs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns
        else:
            return mb_obs_lstm, mb_obs_next_lstm, mb_obs, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions,\
                   mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns


def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=10, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, l2=0.1, d_iters=1, rew_scale=0.1):
    tf.reset_default_graph()
    set_global_seeds(seed)
    buffer = None
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))

    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    if disc_type == 'decentralized' or disc_type == 'decentralized-all':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space,
                          state_only=True, discount=gamma, nstack=nstack, index=k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr, l2_loss_ratio=l2) for k in range(num_agents)
        ]
    else:
        assert False

    # add reward regularization
    if env_id == 'simple_tag':
        reward_reg_loss = tf.reduce_mean(
            tf.square(discriminator[0].reward + discriminator[3].reward) +
            tf.square(discriminator[1].reward + discriminator[3].reward) +
            tf.square(discriminator[2].reward + discriminator[3].reward)
        ) + rew_scale * tf.reduce_mean(
            tf.maximum(0.0, 1 - discriminator[0].reward) +
            tf.maximum(0.0, 1 - discriminator[1].reward) +
            tf.maximum(0.0, 1 - discriminator[2].reward) +
            tf.maximum(0.0, discriminator[3].reward + 1)
        )

        reward_reg_lr = tf.placeholder(tf.float32, ())
        reward_reg_optim = tf.train.AdamOptimizer(learning_rate=reward_reg_lr)
        reward_reg_train_op = reward_reg_optim.minimize(reward_reg_loss)

    tf.global_variables_initializer().run(session=model.sess)
    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type,
                    nobs_flag=True)
    nbatch = nenvs * nsteps
    
    tstart = time.time()
    logger.record_tabular("time", tstart)

    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    for _ in range(bc_iters):
        e_obs, e_actions, e_nobs, _, _ = expert.get_next_batch(nenvs * nsteps)
        e_a = e_actions

    update_policy_until = 0  # 10

    for update in range(1, 2000 + 1):
        obs_lstm, obs_next_lstm, obs, obs_next, states, rewards, report_rewards, masks, actions, values, all_obs, all_nobs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns = runner.run(update)
        
        trun = time.time()
        logger.record_tabular("time_1", trun-tstart)

        total_loss = np.zeros((num_agents, d_iters))

        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs_lstm = [obs_lstm[k][idxs] for k in range(num_agents)]
        mh_obs_next_lstm = [obs_next_lstm[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_obs_next = [obs_next[k][idxs] for k in range(num_agents)]
        mh_values = [values[k][idxs] for k in range(num_agents)]
        if buffer:
            buffer.update(mh_obs_lstm, mh_actions, mh_obs_next_lstm, all_obs, mh_values)
        else:
            buffer = Dset(mh_obs_lstm, mh_actions, mh_obs_next_lstm, all_obs, mh_values, randomize=True, num_agents=num_agents,
                          nobs_flag=True)

        d_minibatch = nenvs * nsteps

        d_iters_new = 1
        for d_iter in range(d_iters_new):
            e_obs, e_actions, e_nobs, e_all_obs, _ = expert.get_next_batch(d_minibatch)
            g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch)

            e_a = e_actions
            g_a = g_actions

            g_log_prob = model.get_log_action_prob(g_obs, g_a)
            e_log_prob = model.get_log_action_prob(e_obs, e_a)

            def calculate_rectangle_vertices(center_x, center_y, length, width, angle):
                angle_rad = np.radians(angle)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
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
                def orientation(p, q, r):
                    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                    if val == 0:
                        return 0
                    return 1 if val > 0 else 2

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

                    if o1 != o2 and o3 != o4:
                        return True

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
                          Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                          Jiaohu_direction):

                dis_between_agent_jiaohu = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
                if dis_between_agent_jiaohu <= 15:
                    agent_v = np.sqrt((Agent_vx) ** 2 + (Agent_vy) ** 2)
                    neig_v = np.sqrt((Jiaohu_vx) ** 2 + (Jiaohu_vy) ** 2)

                    veh_length = 5
                    veh_width = 2

                    a_agent = math.tan(np.radians(Agent_angle_last))
                    a_neig = math.tan(np.radians(Jiaohu_angle_last))

                    b_agent = (Agent_y) - a_agent * (Agent_x)
                    b_neig = (Jiaohu_y) - a_neig * (Jiaohu_x)

                    vertices_agent_now = calculate_rectangle_vertices(Agent_x, Agent_y, veh_length, veh_width,
                                                                      Agent_angle_last)
                    vertices_neig_now = calculate_rectangle_vertices(Jiaohu_x, Jiaohu_y, veh_length, veh_width,
                                                                     Jiaohu_angle_last)
                    intersect_now = check_intersection(vertices_agent_now, vertices_neig_now)
                    if intersect_now == True:
                        GT_value = 0
                    else:
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
                        elif Agent_angle_last == 270:
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
                            if a_neig == a_agent:
                                GT_value = None
                            else:
                                jiaodianx = (b_agent - b_neig) / (a_neig - a_agent)
                                jiaodiany = a_neig * jiaodianx + b_neig

                                agent_b = np.zeros(2)
                                if 0 <= Agent_angle_last < 90:  # tan>0
                                    agent_b = np.array([1, math.tan(math.radians(Agent_angle_last))])
                                elif Agent_angle_last == 90:
                                    agent_b = np.array([0, 2])
                                elif 90 < Agent_angle_last <= 180:  # tan<0
                                    agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                                elif 180 < Agent_angle_last < 270:  # tan>0
                                    agent_b = np.array([-1, -1 * math.tan(math.radians(Agent_angle_last))])
                                elif Agent_angle_last == 270:
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
                                elif Jiaohu_angle_last == 270:
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

                                    if (agent_first_dis / agent_v) < (neig_last_dis / neig_v):  # agent first
                                        GT_value = abs(neig_last_dis / neig_v - agent_first_dis / agent_v)
                                    elif (neig_first_dis / neig_v) < (agent_last_dis / agent_v):  # neig first
                                        GT_value = abs(agent_last_dis / agent_v - neig_first_dis / neig_v)
                                    else:
                                        GT_value = min(abs(agent_last_dis / agent_v - neig_first_dis / neig_v),
                                                       abs(neig_last_dis / neig_v - agent_first_dis / agent_v))

                                elif dot_product_agent >= 0 and dot_product_neig < 0:
                                    GT_value = None  # safe

                                else:
                                    GT_value = None  # safe
                        else:
                            GT_value = None
                else:
                    GT_value = None
                return GT_value

            if disc_type == 'decentralized':
                for k in range(num_agents):
                    if k <= 2:
                        direction_agent = 'left'
                    else:
                        direction_agent = 'straight'

                    obs_k = np.concatenate([g_obs[k], e_obs[k]],axis=0)
                    batch_num_ = np.shape(obs_k)[0]

                    rew_input_fuyuan = obs_k
                    rew_social_allbatch = []

                    for i_batch in range(batch_num_):
                        if rew_input_fuyuan[i_batch][20][0] != 0:
                            use_GT = []
                            pianyi_distance = rew_input_fuyuan[i_batch][20][-2]
                            agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                            agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                            agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                            agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                            agent_angle_last = rew_input_fuyuan[i_batch][20][6] * 191 - 1

                            agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                            for agent_k_ in range(num_agents):
                                if agent_k_ != k:
                                    if agent_k_ <= 2:
                                        direction_jiaohu = 'left'
                                    else:
                                        direction_jiaohu = 'straight'

                                    obs_agent_k_ = np.concatenate([g_obs[agent_k_], e_obs[agent_k_]], axis=0)
                                    batch_num_agent_k_ = np.shape(obs_agent_k_)[0]
                                    rew_input_fuyuan_agent_k_ = obs_agent_k_

                                    if rew_input_fuyuan_agent_k_[i_batch][20][0] != 0:
                                        jiaohu_agent_x = rew_input_fuyuan_agent_k_[i_batch][20][0] * 38 - 4
                                        jiaohu_agent_y = rew_input_fuyuan_agent_k_[i_batch][20][1] * 23 + 14
                                        jiaohu_agent_vx = rew_input_fuyuan_agent_k_[i_batch][20][2] * 21 - 14
                                        jiaohu_agent_vy = rew_input_fuyuan_agent_k_[i_batch][20][3] * 12 - 2
                                        jiaohu_agent_angle_last = rew_input_fuyuan_agent_k_[i_batch][20][6] * 191 - 1

                                        jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                agent_angle_last, direction_agent,
                                                                                jiaohu_agent_x, jiaohu_agent_y,
                                                                                jiaohu_agent_vx, jiaohu_agent_vy,
                                                                                jiaohu_agent_angle_last,
                                                                                direction_jiaohu)
                                        use_GT.append(jiaohu_agent_GT_value)
                                    else:
                                        jiaohu_agent_x = -4
                                        jiaohu_agent_y = 14
                                        jiaohu_agentk_vx = -14
                                        jiaohu_agent_vy = -2
                                        jiaohu_agent_angle_last = -1
                                        jiaohu_agent_GT_value = None
                                        # dis_min = 100000
                                        use_GT.append(jiaohu_agent_GT_value)

                            # landmark in left
                            if rew_input_fuyuan[i_batch][20][38] != 0:
                                delta_left_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][38] * 29 - 14
                                delta_left_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][39] * 30 - 15
                                delta_left_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][40] * 35 - 21
                                delta_left_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][41] * 16 - 5
                                left_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][53] * 360 - 90
                                left_jiaohu_landmark_x = agent_x - delta_left_jiaohu_landmark_x
                                left_jiaohu_landmark_y = agent_y - delta_left_jiaohu_landmark_y
                                left_jiaohu_landmark_vx = agent_vx - delta_left_jiaohu_landmark_vx
                                left_jiaohu_landmark_vy = agent_vy - delta_left_jiaohu_landmark_vy
                                left_jiaohu_landmark_angle_last = left_jiaohu_landmark_angle_last
                            else:
                                left_jiaohu_landmark_x = -5
                                left_jiaohu_landmark_y = -3
                                left_jiaohu_landmark_vx = -16
                                left_jiaohu_landmark_vy = -10
                                left_jiaohu_landmark_angle_last = -90

                            # # landmark in right
                            if rew_input_fuyuan[i_batch][20][42] != 0:
                                delta_right_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][42] * 35 - 15
                                delta_right_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][43] * 29 - 15
                                delta_right_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][44] * 25 - 14
                                delta_right_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][45] * 17 - 7
                                right_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][54] * 360 - 90
                                right_jiaohu_landmark_x = agent_x - delta_right_jiaohu_landmark_x
                                right_jiaohu_landmark_y = agent_y - delta_right_jiaohu_landmark_y
                                right_jiaohu_landmark_vx = agent_vx - delta_right_jiaohu_landmark_vx
                                right_jiaohu_landmark_vy = agent_vy - delta_right_jiaohu_landmark_vy
                                right_jiaohu_landmark_angle_last = right_jiaohu_landmark_angle_last
                            else:
                                right_jiaohu_landmark_x = -5
                                right_jiaohu_landmark_y = -3
                                right_jiaohu_landmark_vx = -16
                                right_jiaohu_landmark_vy = -10
                                right_jiaohu_landmark_angle_last = -90

                            if left_jiaohu_landmark_x != -5:
                                direction_landmark = 'landmark'
                                left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                agent_angle_last, direction_agent,
                                                                                left_jiaohu_landmark_x,
                                                                                left_jiaohu_landmark_y,
                                                                                left_jiaohu_landmark_vx,
                                                                                left_jiaohu_landmark_vy,
                                                                                left_jiaohu_landmark_angle_last,
                                                                                direction_landmark)

                                use_GT.append(left_jiaohu_landmark_GT_value)
                            else:
                                left_jiaohu_landmark_GT_value = None
                                use_GT.append(left_jiaohu_landmark_GT_value)

                            if right_jiaohu_landmark_x != -5:
                                direction_landmark = 'landmark'
                                right_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                 agent_angle_last, direction_agent,
                                                                                 right_jiaohu_landmark_x,
                                                                                 right_jiaohu_landmark_y,
                                                                                 right_jiaohu_landmark_vx,
                                                                                 right_jiaohu_landmark_vy,
                                                                                 right_jiaohu_landmark_angle_last,
                                                                                 direction_landmark)

                                use_GT.append(right_jiaohu_landmark_GT_value)

                            else:
                                right_jiaohu_landmark_GT_value = None
                                use_GT.append(right_jiaohu_landmark_GT_value)

                            penalty = 1
                            delta_angle_last1 = rew_input_fuyuan[i_batch][20][56]
                            # comfort
                            comfort_adj = 0
                            if direction_agent == 'left':
                                left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                                left_delta_angle_last1_realmean = 1.085
                                left_delta_angle_last1_realstd = 0.702
                                if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                                    comfort_adj = 0

                                else:
                                    dis_left_delta_angle_last1 = abs(
                                        left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                                    if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(
                                                dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty

                            else:
                                right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                                right_delta_angle_last1_realmean = 0.001
                                right_delta_angle_last1_realstd = 0.076
                                if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                                    comfort_adj = 0

                                else:
                                    dis_right_delta_angle_last1 = abs(right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                                    if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty

                            # efficiency
                            rew_avespeed = agent_v / 6.8  # 85th expert speed
                            # lane shift
                            rew_lane_pianyi = pianyi_distance
                            # GT(PET)
                            use_GT_list_0 = [x for x in use_GT if x is not None]
                            use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                            rew_minGT_mapped = 0
                            if len(use_GT_list) != 0:
                                # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                                rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                                if rew_minGT <= 1.5:
                                    normalized_data = (rew_minGT - 0) / (1.5 - 0)
                                    rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                                elif 1.5 < rew_minGT < 3:
                                    normalized_data = (rew_minGT - 1.5) / (3 - 1.5)
                                    rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                                elif 3 <= rew_minGT <= 4:
                                    normalized_data = (rew_minGT - 3) / (4 - 3)
                                    rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                                elif rew_minGT > 4:
                                    normalized_data = np.exp(-(1 / (rew_minGT - 4)))
                                    rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75

                            else:
                                rew_minGT_mapped = 0

                            rew_social_allbatch.append([10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])

                        else:
                            rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])


                    canshu_social_allbatch_array = np.array(rew_social_allbatch)

                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs[k],
                        g_actions[k],
                        g_nobs[k],
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs[k],
                        e_actions[k],
                        e_nobs[k],
                        e_log_prob[k].reshape([-1, 1]), canshu_social_allbatch_array)
            elif disc_type == 'decentralized-all':
                g_obs_all = np.concatenate(g_obs, axis=1)
                g_actions_all = np.concatenate(g_actions, axis=1)
                g_nobs_all = np.concatenate(g_nobs, axis=1)
                e_obs_all = np.concatenate(e_obs, axis=1)
                e_actions_all = np.concatenate(e_actions, axis=1)
                e_nobs_all = np.concatenate(e_nobs, axis=1)
                for k in range(num_agents):
                    batch_num_ = np.shape(g_obs_all[k])[0]
                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs_all,
                        g_actions_all,
                        g_nobs_all,
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs_all,
                        e_actions_all,
                        e_nobs_all,
                        e_log_prob[k].reshape([-1, 1]), batch_num_)
            else:
                assert False

            if env_id == 'simple_tag':
                if disc_type == 'decentralized':
                    feed_dict = {discriminator[k].obs: np.concatenate([g_obs[k], e_obs[k]], axis=0)
                                 for k in range(num_agents)}
                elif disc_type == 'decentralized-all':
                    feed_dict = {discriminator[k].obs: np.concatenate([g_obs_all, e_obs_all], axis=0)
                                 for k in range(num_agents)}
                else:
                    assert False
                feed_dict[reward_reg_lr] = discriminator[0].lr.value()
                model.sess.run(reward_reg_train_op, feed_dict=feed_dict)

            idx += 1

        tdistr = time.time()
        logger.record_tabular("time_2", tdistr-trun)



        if update > update_policy_until:
            policy_loss, value_loss, policy_entropy = model.train(obs_lstm, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        
        tpoltr = time.time()
        logger.record_tabular("time_3", tpoltr-tdistr)
        
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                if update > update_policy_until:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
                        
                    except:
                        pass

            total_loss_m = np.mean(total_loss, axis=1)
            total_reward = (np.mean(rewards, axis = 1))
            for k in range(num_agents):
                logger.record_tabular("total_loss %d" % k, total_loss_m[k])
            logger.record_tabular("total_loss" , np.sum(total_loss_m))
            logger.record_tabular('rewards ' , np.sum(total_reward))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            # print('Saving to', savepath)
            model.save(savepath)
            if disc_type == 'decentralized' or disc_type == 'decentralized-all':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            else:
                assert False
    coord.request_stop()
    # coord.join(enqueue_threads)
    env.close()
