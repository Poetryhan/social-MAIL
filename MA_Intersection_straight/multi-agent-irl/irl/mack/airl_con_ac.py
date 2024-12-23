import os.path as osp
import random
import time

import joblib
import numpy as np
import pandas as pd
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
import os
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
                                inter_op_parallelism_threads=nprocs) # 创建 TensorFlow 配置对象，用于配置 TensorFlow 会话的行为。
        config.gpu_options.allow_growth = False    # 配置 GPU 内存管理，允许动态分配 GPU 内存。
        self.sess = sess = tf.Session(config=config)
        # self.sess = sess = tf.Session(config=config) # 创建 TensorFlow 会话（Session），该会话将用于运行模型和进行训练。
        nbatch = nenvs * nsteps # 计算每个训练批次中的样本数量，通常为环境数量乘以每个环境的步数。 nsteps 是 5 nbatch 是 50
        self.num_agents = num_agents = len(ob_space)
        # self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        if identical is None:
            identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]

        # 这段代码的主要作用是根据智能体的相似性情况，将它们分为不同的组，并记录每个组的起始智能体的指针（pointer）。
        # 相似的智能体共享相同的组，而不同组的智能体具有不同的组指针。这对于在多智能体强化学习中管理和处理不同组智能体的数据和信息非常有用。
        scale = [1 for _ in range(num_agents)] # scale 用于存储每个智能体的比例因子，后续会根据相似性情况来调整这些因子。
        pointer = [i for i in range(num_agents)] # pointer 用于记录每个智能体的指针。
        h = 0 # 初始化一个变量 h 为0，该变量用于跟踪智能体的相似性分组。
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k # 将 pointer[h] 的值设置为 k，表示当前智能体的指针。这是为了记录不同组的起始智能体。
                h = k # 更新 h 的值为 k，以便标记下一个相似组的起始智能体。
        pointer[h] = num_agents

        # 这段代码用于创建用于存储智能体相关信息的 TensorFlow 占位符（placeholders）。这些占位符将用于在训练过程中接收输入数据和相关参数。

        # !!! 以下的代码要再好好看一看 还有policy的nbatch这些参数

        A, ADV, R, PG_LR = [], [], [], [] # 创建了四个空列表 A、ADV、R 和 PG_LR，分别用于存储智能体的动作（Action）、优势值（Advantage）、奖励（Reward）和策略梯度学习率（Policy Gradient Learning Rate）
        for k in range(num_agents):
            if identical[k]:  # 这里检查当前智能体是否被认为是相同的，即与前一个智能体共享信息。如果是相同的，那么就将当前智能体的信息列表（A、ADV、R、PG_LR）与前一个智能体的信息列表相同，这样它们共享相同的信息。这是因为在某些情况下，多个智能体可能共享相同的策略或网络参数。
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
            else:
                # # 为当前智能体创建一个 TensorFlow 占位符，用于存储动作信息（Action）。占位符的形状是 [nbatch * scale[k], n_ac]，其中 nbatch 是批量大小，n_ac 是动作空间的维度。这个占位符将在训练过程中接收动作数据。
                A.append(tf.placeholder(tf.float32, [nbatch * scale[k], n_ac])) # 对于第一个agent来说,[50*36,2],对于第36个agent来说,[50*31,2]
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]])) #
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, [])) # 这个占位符的形状是 []，表示它是一个标量，将在训练中接收策略梯度学习率的数值。

        # 这段代码主要用于定义和初始化与每个智能体相关的神经网络模型、损失函数以及其他与训练相关的变量。下面是每一行代码的详细解释：
        pg_loss, entropy, vf_loss, train_loss = [], [], [], [] # 创建四个空列表，分别用于存储策略损失 (pg_loss)、熵 (entropy)、值函数损失 (vf_loss) 和总体训练损失 (train_loss)。
        self.model = step_model = []
        self.model2 = train_model = [] # 创建两个空列表 step_model 和 train_model，并将它们同时赋值给类的属性 self.model 和 self.model2。这些列表将用于存储智能体的两个神经网络模型，分别用于采样动作和训练。
        self.pg_fisher = pg_fisher_loss = []
        self.logits = logits = []
        sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = [] # 创建多个空列表，用于存储策略梯度损失 (pg_fisher_loss)、策略的逻辑回归输出 (logits)、样本网络 (sample_net)、值函数梯度损失 (vf_fisher_loss)、联合梯度损失 (joint_fisher_loss) 和对数似然损失 (lld)。
        self.lld = lld = []
        self.log_pac = []

        # 这段代码的主要作用是初始化多智能体对抗生成逆强化学习模型的各个组件，并为每个智能体初始化相应的神经网络模型和损失函数，以便在后续的训练中使用。不同的智能体可以共享模型参数或拥有独立的模型参数，这取决于 identical[k] 的值。
        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1]) # 将前一个智能体的 step_model 和 train_model 添加到当前智能体的对应列表中。这表示当前智能体与前一个智能体共享相同的神经网络模型。
            else: # 如果 identical[k] 为假，表示当前智能体有自己的神经网络模型，因此使用 policy 函数创建一个新的模型，并将其添加到模型列表中。
                # # print('ob_space:',ob_space)
                step_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                         nenvs, 1, nstack, reuse=False, name='%d' % k))
                # step_model 是用于进行行为策略采样的模型。这个模型用于在环境中执行动作，以便智能体与环境进行交互。
                # 具体而言，step_model 用于根据当前观察（状态）生成动作，并且通常是在每个时间步骤中使用的模型。它是一个轻量级模型，通常只包含生成动作所需的部分，因此相对较快。
                # step_model 用于在环境中执行动作，因此其 nbatch 反映了并行执行的环境数量和每个环境中的时间步数。在这里，您提到有10个并行的CPU线程，每个线程执行5个时间步，所以 nbatch 为10 * 5 = 50。这是因为在一个时间步中，每个线程都会执行一个动作，因此总的批处理大小是每个线程的数量乘以每个线程的时间步数。
                train_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                          nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))
                # train_model 是用于训练智能体的模型。这个模型用于计算损失函数、计算策略梯度以及更新模型的权重。
                # 在强化学习中，train_model 通常需要更多的计算资源，因为它要执行训练的反向传播和优化步骤。train_model 是在训练时使用的模型，用于优化智能体的策略以便改进性能。
                # train_model 用于训练智能体的策略，因此其 nbatch 反映了用于训练的样本数量。在这里，您提到有36个共享相同策略的智能体，并且每个智能体执行5个时间步，所以 nbatch 为10（并行线程数）乘以36（共享策略的智能体数量）乘以5（每个智能体的时间步数），即10 * 36 * 5 = 1800。这是因为在训练过程中，每个智能体的轨迹都会被收集并用于更新策略，因此 nbatch 表示了训练时用于计算梯度的样本数量。

            logpac = (tf.reduce_mean(mse(train_model[k].a0, A[k]),1)) # A[k] 在这段代码中表示第 k 个智能体的行动。它是一个占位符（placeholder），在训练时将被真实的行动数据填充。在强化学习中，行动通常由智能体根据当前观察状态选择，然后由环境返回新的观察状态和奖励信号。
            # 计算策略损失 logpac：对于每个智能体，计算策略损失 logpac。具体计算方式可能依赖于模型的类型和任务的不同。这里使用了 均方误差 mse 函数来计算策略损失。损失值被添加到 log_pac 列表中。

            # logpac = tf.reduce_mean(mse(train_model[k].a0, A[k]))
			#tf.nn.sparse_softmax_cross_entropy_with_logits(
                #logits=train_model[k].pi, labels=A[k])
            self.log_pac.append(-logpac)   # 将策略损失添加到 self.log_pac 列表中，负号表示后续会进行梯度上升优化。 log_pac越大越好,是负的,越接近于0越好

            lld.append(tf.reduce_mean(logpac)) # 计算对数似然损失 lld，并将其添加到 lld 列表中。
            logits.append(train_model[k].a0) # 将训练模型的策略输出 a0 添加到 logits 列表中。

            pg_loss.append(tf.reduce_mean(ADV[k] * logpac)) # 计算策略损失 pg_loss，这是策略梯度损失。越小越好,越接近0越好
            entropy.append(tf.constant([0.01])) # 将一个常数0.0添加到 entropy 列表中。通常情况下，这个常数是用于控制策略的熵正则化项，但在这里被设置为0.0。
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k] # 将策略梯度损失 pg_loss 减去策略熵正则化项，以兼顾探索和利用的平衡。
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k]))) # 计算值函数损失 vf_loss，这是值函数的均方误差损失。 越小越好,越接近0越好
            # train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])  # 将策略损失和值函数损失组合成总体的训练损失 train_loss，其中 vf_coef 是一个权重因子，用于调整两者之间的权重。越接近0越好,越小越好

            # 1. 定义 L2 正则化系数
            l2_lambda = 0.01  # L2 正则化系数，通常是一个小的值

            # 2. 计算 L2 正则化项
            l2_regularizer = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])  # 计算所有可训练变量的 L2 正则化项
            # 总损失包括 L2 正则化项
            total_loss = pg_loss[k] + vf_coef * vf_loss[k] + l2_lambda * l2_regularizer
            # 将总损失添加到 train_loss 列表中
            train_loss.append(total_loss)

            pg_fisher_loss.append(-tf.reduce_mean(logpac)) # 计算策略梯度损失  负值
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf))) # 这行代码的目的是为每个智能体的值函数输出添加随机噪声，以增加策略的多样性和探索性，有助于更好地学习策略。
            # train_model[k].vf 表示智能体 k 训练模型的值函数输出，它代表了每个状态的估计值。这是一个具有特定形状的张量。
            # tf.random.normal(tf.shape(train_model[k].vf)) 是 TensorFlow 中生成一个与 train_model[k].vf 张量相同形状的随机正态分布噪声的操作。它的作用是添加一些随机性，以增加探索性。
            # sample_net.append(...) 将生成的随机噪声添加到 sample_net 列表中，用于后续的计算.
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2))) #  计算了用于策略梯度更新的 Fisher Loss（pg_fisher_loss）和值函数 Fisher Loss（vf_fisher_loss）。这些损失通常用于计算策略梯度的二阶导数信息，用于更稳定地更新策略。
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k]) # 将策略损失和值函数损失组合成联合损失 joint_fisher_loss，用于后续的优化。
            # 这段代码的主要作用是初始化多智能体对抗生成逆强化学习模型的各个组件，包括神经网络模型、损失函数以及与训练相关的变量和损失。
            # 这些组件将在后续的训练中使用，用于优化策略和值函数以实现逆强化学习任务。不同的智能体可以共享模型参数，也可以有各自独立的模型参数，这取决于 identical[k] 的值。


        # 这部分代码的主要作用是初始化和管理多智能体对抗生成逆强化学习模型的参数、梯度。
        self.policy_params = [] # 是用来存储多智能体的策略网络的参数的列表。
        self.value_params = []  # 是用来存储多智能体的值函数网络的参数的列表。

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))

                # 上述代码中的一部分使用了这个 find_trainable_variables(key) 函数，具体来说，它用于在每个智能体的策略网络和值函数网络中查找对应作用域下的可训练变量。
                # 这是因为在多智能体的场景中，每个智能体都有自己的策略和值函数，它们可能具有不同的参数，因此需要通过指定作用域 key 来获取与每个智能体相关的参数列表。
                # 通过这种方式，可以将不同智能体的参数进行独立管理，以便在训练过程中进行更新和优化。

        # print('self.policy_params:',np.shape(self.policy_params),'self.value_params:',np.shape(self.value_params))
        # (8, 11) self.value_params: (8, 10)
        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        # self.params是一个包含所有智能体的策略和值函数参数列表的总参数列表，通过将 self.policy_params 和 self.value_params 中对应智能体的参数逐一相加得到。
        # print('self.params:', np.shape(self.params), self.params, 'train_loss:', train_loss)
        params_flat = [] # params_flat 是将 self.params 中的参数扁平化为一个一维列表。
        for k in range(num_agents):
            params_flat.extend(params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]   # self.grads_check 存储了用于训练的损失函数 train_loss 对于每个智能体的参数 params 的梯度，通过 tf.gradients 计算。

        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ] # clone_grads 存储了对数似然损失 lld 对于每个智能体的参数 params 的梯度
        # print('grads:', np.shape(grads), grads)
        # 这段代码的主要作用是配置和初始化用于多智能体对抗生成逆强化学习模型训练的优化器和操作。
        self.optim = optim = [] # self.optim 存储了每个智能体的优化器（在后续代码中进行了初始化）。
        self.clones = clones = [] # self.clones 存储了对于每个智能体的克隆操作，用于后续的策略梯度的计算。
        update_stats_op = [] # update_stats_op 是用于更新统计信息的操作。
        train_op, clone_op, q_runner = [], [], [] # train_op 存储了每个智能体的训练操作。clone_op 存储了每个智能体的克隆操作。q_runner 是一个用于多线程操作的队列操作。

        for k in range(num_agents):
            if identical[k]:
                optim.append(optim[-1])
                train_op.append(train_op[-1])
                # q_runner.append(q_runner[-1])
                clones.append(clones[-1])
                clone_op.append(clone_op[-1])
            else:
                with tf.variable_scope('optim_%d' % k): # 创建一个新的优化器（kfac.KfacOptimizer）：为每个智能体创建一个新的 K-FAC 优化器，用于优化策略网络和值函数网络的参数。这里包括了一系列超参数，如学习率 (learning_rate)、梯度裁剪 (clip_kl)、动量 (momentum) 等。

                    print('PG_LR[k]*0.1:', PG_LR[k]*0.1)
                    optim.append(tf.train.AdamOptimizer(learning_rate=PG_LR[k]))
                    # 剪枝
                    # 获取梯度和参数
                    # print('paramsk:',np.shape(params[k]), params[k])  # (21,)
                    # print('joint_fisher_loss:',np.shape(joint_fisher_loss), joint_fisher_loss)  # (8,)
                    # # 做法一
                    # grads_and_vars_train = optim[k].compute_gradients(joint_fisher_loss, var_list=params[k])  # 替换 'loss' 为你的损失函数
                    #
                    # # capped_grads_and_vars_train_ = []
                    # # 可选：对梯度进行裁剪，以防止梯度爆炸
                    # # for grad, var in grads_and_vars_train:
                    # #     if grad == None:
                    # #         print('var:',var)
                    # #     capped_grads_and_vars_train_.append((tf.clip_by_value(grad, -1.0, 1.0), var))
                    #
                    # capped_grads_and_vars_train = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars_train]
                    #
                    # # 应用梯度更新
                    # # print('capped_grads_and_vars_train:',np.shape(capped_grads_and_vars_train),capped_grads_and_vars_train)
                    # train_op_, q_runner_ = optim[k].apply_gradients(capped_grads_and_vars_train)

                    # 做法二
                    # 定义梯度裁剪的阈值
                    clip_threshold = 1.0  # 可以根据需要调整阈值大小
                    # 对梯度进行裁剪
                    print('gradsk:', np.shape(grads[k]),grads[k])

                    # # 对第 k 个 agent 的梯度进行裁剪
                    # clipped_grads_k = [tf.clip_by_norm(grad, clip_threshold) if grad is not None else grad for grad in grads[k]]

                    # print('clipped_grads_k:', np.shape(clipped_grads_k), clipped_grads_k)
                    # train_op_, q_runner_ = optim[k].apply_gradients(zip(grads[k], params[k]))
                    # 对梯度进行按全局范数剪枝
                    clipped_grads = [
                        tf.clip_by_global_norm(agent_grads, clip_norm=3.0)[0]
                        for agent_grads in grads
                    ]
                    train_op_ = optim[k].apply_gradients(list(zip(clipped_grads[k], params[k])))
                    # train_op_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                    train_op.append(train_op_) # 将训练操作和队列运行器添加到相应的列表中：将训练操作 train_op_ 添加到 train_op 列表，将队列运行器 q_runner_ 添加到 q_runner 列表。
                    # q_runner.append(q_runner_)

                    # optim.append(kfac.KfacOptimizer(learning_rate=PG_LR[k], clip_kl=kfac_clip,momentum=0.9, kfac_update=1, epsilon=0.01, stats_decay=0.99, Async=0, cold_iter=10,max_grad_norm=max_grad_norm))
                    # # print('joint_fisher_loss:',joint_fisher_loss,params[k],k)
                    # # print('grads:', np.shape(grads[k]), grads[k])
                    # # print('params:', np.shape(params[k]), params[k])
                    # update_stats_op.append(optim[k].compute_and_apply_stats(joint_fisher_loss, var_list=params[k])) # 使用优化器计算和应用统计信息：调用 compute_and_apply_stats 方法计算并应用统计信息，这些信息用于 K-FAC 优化器的训练更新。
                    # train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k]))) # 创建训练操作和队列运行器：通过优化器的 apply_gradients 方法创建训练操作 train_op_，并创建与之相关的队列运行器 q_runner_。
                    # train_op.append(train_op_) # 将训练操作和队列运行器添加到相应的列表中：将训练操作 train_op_ 添加到 train_op 列表，将队列运行器 q_runner_ 添加到 q_runner 列表。
                    # q_runner.append(q_runner_)

                with tf.variable_scope('clone_%d' % k): # with tf.compat.v1.variable_scope('clone_%d' % k):：创建一个名为 'clone_k' 的 TensorFlow 变量作用域，其中 k 是智能体的索引，用于区分不同智能体的克隆优化器。
                    clones.append(tf.train.AdamOptimizer(learning_rate=PG_LR[k]))

                    # 获取梯度和参数
                    # print('self.policy_paramsk:', np.shape(self.policy_params[k]), self.policy_params[k]) # (11,)
                    # print('pg_fisher_lossk:', np.shape(pg_fisher_loss[k]), pg_fisher_loss[k]) # () Tensor("Neg_1:0", shape=(), dtype=float32)
                    # # 做法一
                    # grads_and_vars_clone = clones[k].compute_gradients([pg_fisher_loss[k]], var_list=self.policy_params[k])  # 替换 'loss' 为你的损失函数
                    #
                    # # 可选：对梯度进行裁剪，以防止梯度爆炸
                    # capped_grads_and_vars_clone = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars_clone]
                    #
                    # # 应用梯度更新
                    # clone_op_, q_runner_ = clones[k].apply_gradients(capped_grads_and_vars_clone)

                    # 做法二
                    # 对梯度进行按全局范数剪枝
                    clipped_clone_grads = [
                        tf.clip_by_global_norm(agent_clone_grads, clip_norm=3.0)[0]
                        for agent_clone_grads in clone_grads
                    ]
                    clone_op_ = clones[k].apply_gradients(list(zip(clipped_clone_grads[k], self.policy_params[k])))

                    # clone_op_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    # clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    clone_op.append(clone_op_)


                    # clones.append(kfac.KfacOptimizer(learning_rate=PG_LR[k], clip_kl=kfac_clip,momentum=0.9, kfac_update=1, epsilon=0.01,stats_decay=0.99, Async=0, cold_iter=10,max_grad_norm=max_grad_norm))
                    # # 创建一个 K-FAC 优化器对象，该优化器用于对策略网络参数进行优化。这里设置了一系列超参数，包括学习率
                    # # print('clones:',np.shape(clones),'pg_fisher_loss:',np.shape(pg_fisher_loss),'policy_params:',np.shape(self.policy_params),'policy_params:',self.policy_params)
                    # # clones: (1,) pg_fisher_loss: (8,) policy_params: (8, 1)
                    # update_stats_op.append(clones[k].compute_and_apply_stats(
                    #     pg_fisher_loss[k], var_list=self.policy_params[k])) # 使用克隆优化器计算并应用统计信息。compute_and_apply_stats 方法将根据传入的损失函数 pg_fisher_loss[k] 和参数列表 var_list=self.policy_params[k] 计算并应用 K-FAC 优化所需的统计信息。这些统计信息用于优化策略网络的参数。
                    # clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    # # 使用克隆优化器应用梯度更新。apply_gradients 方法接受一个梯度-参数对的列表（由 list(zip(clone_grads[k], self.policy_params[k])) 创建），并将这些梯度应用到策略网络参数上，以实现参数的更新。
                    # # 同时，它返回 clone_op_ 用于执行这个操作，以及 q_runner_ 用于管理队列运行。这些操作和队列运行将用于在训练过程中执行参数更新。
                    # clone_op.append(clone_op_) # 将克隆操作 clone_op_ 添加到 clone_op 列表中，以便后续的训练过程中一次性执行克隆操作。

        update_stats_op = tf.group(*update_stats_op) # update_stats_op = tf.group(*update_stats_op)：将统计信息更新操作整合为一个组操作，用于一次性运行。
        train_ops = train_op # 将训练操作整合为一个操作，用于一次性运行。
        clone_ops = clone_op # 将克隆操作整合为一个操作，用于一次性运行。
        train_op = tf.group(*train_op)  # 将训练操作整合为一个组操作，用于一次性运行。
        clone_op = tf.group(*clone_op)  # 将克隆操作整合为一个组操作，用于一次性运行。

        # 设置学习率调度器：创建了学习率调度器 self.lr 和 self.clone_lr，用于在训练过程中动态调整学习率。这些学习率调度器可能根据 lrschedule 和 total_timesteps 的设置，以一定规则更新学习率。
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # 克隆是智能体或代理的不同副本，每个克隆具有自己的策略网络，用于学习不同的策略。克隆用于多智能体环境，以便不同智能体学习不同的行为。
        # 优化器是一个用于参数优化的算法，它负责在每个智能体或代理的策略网络内部调整权重和偏差，以最小化损失函数。优化器是用于神经网络训练的通用工具，不仅适用于多智能体情况，也适用于单一智能体的任务。
        # 在这段代码中，每个克隆都有自己的优化器，用于独立地优化其策略网络参数。优化器负责更新克隆的策略网络，以便在对抗生成逆强化学习中进行训练更新。
        # 总之，克隆是智能体的独立副本，而优化器是用于参数优化的算法。在多智能体对抗生成逆强化学习中，每个克隆都有自己的优化器，用于学习其策略网络的参数。这样可以使不同智能体学习不同的策略，以增强对抗性。

        # 这段代码是用于执行训练的函数，其主要作用是根据输入的观察数据（obs）、状态（states）、奖励（rewards）、掩码（masks）、行动（actions）和值函数估计（values）来进行模型的训练更新，其中包括策略网络和值函数网络的更新。
        # obs_lstm, states, rewards, masks, actions, values
        # mb_obs_lstm,  mb_states,  mb_returns, mb_masks, mb_actions, mb_values
        # mb_obs_lstm(8, 50, 21, 46), mb_states空, mb_returns(8, 50), mb_masks(8, 50), mb_actions(8, 50, 2), mb_values(8, 50)
        def train(obs, states, rewards, masks, actions, values):
            advs = [rewards[k] - values[k] for k in range(num_agents)] # ：首先，对每个智能体（代理）计算优势函数（advs），这是奖励（rewards）减去值函数估计（values）的结果，用于衡量每个行动的优劣。
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1) # 观察数据处理：将所有智能体的观察数据（obs）进行拼接，以便输入到模型中。
            # print('train_model中的ob:',np.shape(ob))  # (50, 8*21, 46)
            td_map = {} # 构建一个字典 td_map，用于将数据传递给 TensorFlow 图中的占位符。这个字典将在后续的训练中使用，包括策略网络和值函数网络的更新。
            for k in range(num_agents):
                # 如果 identical[k] 为真，表示该智能体与前一个智能体共享策略网络参数，因此跳过后续操作。
                # 否则，构建一个新的子字典 new_map，用于将数据传递给该智能体的策略网络和值函数网络。这包括观察数据、行动、优势函数、奖励和学习率等信息
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = [] # 创建一个空列表 action_v，用于存储其他智能体的行动信息。
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([actions[i]
                                                   for i in range(num_agents) if i != k], axis=1)) # 在每次循环中，将其他智能体的行动信息（actions[i]）进行拼接，并添加到 action_v 中。

                    action_v = np.concatenate(action_v, axis=0) # 最后，将所有其他智能体的行动信息拼接成一个数组 action_v。
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})
                # print('action_v:', np.shape(action_v))  # (150or250, 14)
                # print('X_attention:', np.shape(np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0)))  # (150or250,21,46)
                # print('X_v:', np.shape(np.concatenate([ob.copy() for j in range(k, pointer[k])], axis=0)))  # (150or250,21*8,46)
                # print('Ak:', np.shape(np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)))  # (150or250,2)
                # print('Ak:', np.shape(np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)))  # (150or250,2)
                # print('ADVk:', np.shape(np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0)))  # (150or250,1)
                # print('Rk:', np.shape(np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0)))  # (150or250,1)
                # print('PG_LRk:', cur_lr / float(scale[k]))  # ()

                # 这里处理ob_lstm（8, batch, 21, 46）得到mask_atime（21，batch,10,10）, mask_times（batch,21,21）
                X_attention_train = np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0)  # (batch,21,46)
                batch_train = X_attention_train.shape[0]
                time_train = X_attention_train.shape[1]
                # print('训练模型得到的batch_train：', batch_train, time_train)  # (8, 10, 10, 2)
                # 主要是trj_go_step,shape应该为（batch，1）
                train_trj_go_step = np.empty((batch_train, 1))
                num_features = 10  # 每个序列的参数的个数
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all_train = []  # 存放对于第k个agent来说，每一个时刻的mask矩阵，shape为（21，batch，10,10）
                mask_times_train = np.ones([batch_train, 21, 21],
                                         dtype=bool)  # 存放对于第k个agent来说，在时刻维度的mask矩阵，shape为（batch，21,21）
                for time_i in range(time_train):
                    X_attention_ONE_TIME_train = X_attention_train[:, time_i, :]  # (batch, 46)
                    # print('X_attention_ONE_TIME_train:', np.shape(X_attention_ONE_TIME_train))
                    # 把X_attention_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
                    # 定义每个时间步长的特征数
                    # 以下是numpy版本下的代码
                    sub_inputsRag_all_np_train = []  # 存放每一个时刻的拆分之后的数据
                    # 遍历每个样本
                    for j in range(batch_train):
                        # 创建一个零张量
                        sub_inputsRag_j_np_train = np.zeros([0, num_features], dtype=np.float32)
                        # 记录当前位置
                        current_pos_np_train = 0
                        # 遍历每个特征的长度
                        for k_, step_size in enumerate(step_sizes_np):
                            # 截取当前时间步长的特征
                            feature_slice = X_attention_ONE_TIME_train[j,
                                            current_pos_np_train: current_pos_np_train + step_size]
                            # print('feature_slice:',np.shape(feature_slice))
                            # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            # 在垂直方向堆叠
                            # print('feature_slice:',np.shape(feature_slice))
                            sub_inputsRag_j_np_train = np.concatenate(
                                [sub_inputsRag_j_np_train, np.expand_dims(feature_slice, axis=0)], axis=0)
                            # 最后一步会得到(10,10)
                            # 更新当前位置
                            current_pos_np_train += step_size
                        sub_inputsRag_all_np_train.append(sub_inputsRag_j_np_train)
                        # 最后会得到（batch，10,10）

                    # 在垂直方向堆叠，形成 RaggedTensor
                    sub_inputsRag_np_train = np.stack(sub_inputsRag_all_np_train, axis=0)  # （nbatch, 10, 10)
                    # print('sub_inputsRag_np_train:', np.shape(sub_inputsRag_np_train))
                    # 形成这个时刻的mask （batch，10，10）
                    mask_atime_train = np.ones([batch_train, 10, 10], dtype=bool)
                    for j_mask in range(batch_train):
                        for i_mask in range(10):
                            if sub_inputsRag_np_train[j_mask, i_mask, 0] == 0:
                                # 说明这个交互对象没有/agent没有
                                mask_atime_train[j_mask, i_mask, :] = False
                                mask_atime_train[j_mask, :, i_mask] = False
                    mask_atime_all_train.append(mask_atime_train)
                mask_atime_all_new_train = np.stack(mask_atime_all_train, axis=0)  # （21, nbatch, 10, 10)

                # 另一种做法
                # # 时间维度上的mask （batch,21,21）
                # # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
                # for i_batch in range(batch_get):
                #     for i_time in range(time_get):
                #         if X_attention[i_batch, i_time, 0] == 0:
                #             # 说明这个agent还没往前走，那么所有的都是要掩码的
                #             mask_times_get[i_batch, i_time, :] = False
                #             mask_times_get[i_batch, :, i_time] = False

                # 时间维度上的mask （batch,21,21）
                # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
                for i_batch in range(batch_train):
                    if X_attention_train[i_batch][20][0] == 0:  # k_ob_lstm (batch,21,46)
                        # 说明这个agent还没往前走或者是无效的，那么所有的都是要掩码的
                        mask_times_train[i_batch, :, :] = False
                    else:
                        # 说明这个agent往前走了，只需要找到在哪个时刻往前走的就可以了
                        for time_i_batch in range(time_train):
                            if X_attention_train[i_batch][time_i_batch][0] != 0:
                                mask_times_train[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break  # 退出循环，已经找到第一个有效的时刻


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
                print('cur_lr / float(scale[k]):',cur_lr / float(scale[k]))
                # new_map.update({...})：将以下信息添加到 new_map 中，这些信息将传递给当前智能体的策略网络和值函数网络进行训练更新：
                #
                # train_model[k].X 和 train_model[k].X_v：观察数据的输入占位符。
                # A[k]：当前智能体的行动数据。
                # ADV[k]：当前智能体的优势函数。
                # R[k]：当前智能体的奖励。
                # PG_LR[k]：当前智能体的学习率，通过除以 scale[k] 进行缩放。

                sess.run(train_ops[k], feed_dict=new_map) # 调用 sess.run 来执行训练操作 train_ops[k]，并将数据传递给 TensorFlow 图中的占位符。这一步实际上是进行策略网络的训练更新。
                td_map.update(new_map)  # 将 new_map 中的数据更新到主字典 td_map 中，以便传递给下一个智能体的训练操作。

                if states[k] != []: # if states[k] != []:：检查当前智能体是否具有状态信息。
                    td_map[train_model[k].S] = states # td_map[train_model[k].S] 和 td_map[train_model[k].M]：如果智能体具有状态信息，则将状态信息和掩码信息添加到主字典 td_map 中，以便传递给智能体的策略网络和值函数网络。
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy, train_loss_out = sess.run(
                [pg_loss, vf_loss, entropy, train_loss],
                td_map
            ) # 通过 sess.run 计算策略损失 pg_loss、值函数损失 vf_loss 和策略熵 entropy，并将这些值返回作为训练的结果。
            print('policy_loss:',policy_loss,'value_loss:',value_loss,'policy_entropy:',policy_entropy)
            return policy_loss, value_loss, policy_entropy, train_loss_out

        # 这段代码的主要作用是进行克隆（Clone）操作，它用于多智能体对抗生成逆强化学习中的克隆训练。
        def clone(obs, actions):
            td_map = {} # 创建一个空的 TensorFlow 字典 td_map，用于传递数据给 TensorFlow 图中的占位符。
            cur_lr = self.clone_lr.value() #  获取当前的克隆学习率，这个学习率用于克隆训练。
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {} # 创建一个新的子字典 new_map，用于将数据传递给该智能体的策略网络。这包括观察数据、行动和学习率等信息。
                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                }) # 更新 new_map，将智能体 k 的观察数据、行动以及学习率等信息添加到字典中。这些数据将在克隆训练中用于更新智能体的策略。
                sess.run(clone_ops[k], feed_dict=new_map) # 使用 TensorFlow 的 sess.run 方法来执行克隆操作 clone_ops[k]，并将数据传递给 TensorFlow 图中的占位符。这一步实际上是进行策略网络的克隆训练更新。
                td_map.update(new_map) # 将 new_map 中的数据更新到 td_map 中，以便后续计算。
            lld_loss = sess.run([lld], td_map) # 使用 TensorFlow 的 sess.run 方法计算策略损失 lld，并将 td_map 中的数据传递给 TensorFlow 图中的占位符。这一步计算克隆训练中的策略损失。
            return lld_loss

        # 其作用是获取多智能体的对数动作概率（log action probability）
        def get_log_action_prob(obs, actions): #  定义一个函数 get_log_action_prob，该函数接受两个参数 obs（观察数据）和 actions（行动数据）。
            # obs: (8, 50, 21, 46) actions: (8, 50, 2)
            action_prob = []  # 创建一个空的列表 action_prob，用于存储每个智能体的对数动作概率。
            for k in range(num_agents):
                if identical[k]:
                    continue

                is_training = True

                # 这里处理ob_lstm（8, batch, 21, 46）得到mask_atime（21，batch,10,10）, mask_times（batch,21,21）
                X_attention = np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0)  # (batch,21,46)
                batch_get = X_attention.shape[0]
                time_get = X_attention.shape[1]
                # print('训练模型得到的batch_get：', batch_get, time_get)  # (8, 10, 10, 2)
                # 主要是trj_go_step,shape应该为（batch，1）
                get_trj_go_step = np.empty((batch_get, 1))
                num_features = 10  # 每个序列的参数的个数
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all_get = []  # 存放对于第k个agent来说，每一个时刻的mask矩阵，shape为（21，batch，10,10）
                mask_times_get = np.ones([batch_get, 21, 21],
                                         dtype=bool)  # 存放对于第k个agent来说，在时刻维度的mask矩阵，shape为（batch，21,21）
                for time_i in range(time_get):
                    X_attention_ONE_TIME = X_attention[:, time_i, :]  # (batch, 46)
                    # print('X_attention_ONE_TIME:', np.shape(X_attention_ONE_TIME))
                    # 把X_attention_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
                    # 定义每个时间步长的特征数
                    # 以下是numpy版本下的代码
                    sub_inputsRag_all_np_get = []  # 存放每一个时刻的拆分之后的数据
                    # 遍历每个样本
                    for j in range(batch_get):
                        # 创建一个零张量
                        sub_inputsRag_j_np_get = np.zeros([0, num_features], dtype=np.float32)
                        # 记录当前位置
                        current_pos_np_get = 0
                        # 遍历每个特征的长度
                        for k_, step_size in enumerate(step_sizes_np):
                            # 截取当前时间步长的特征
                            feature_slice = X_attention_ONE_TIME[j, current_pos_np_get: current_pos_np_get + step_size]
                            # print('feature_slice:',np.shape(feature_slice))
                            # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            # 在垂直方向堆叠
                            # print('feature_slice:',np.shape(feature_slice))
                            sub_inputsRag_j_np_get = np.concatenate(
                                [sub_inputsRag_j_np_get, np.expand_dims(feature_slice, axis=0)], axis=0)
                            # 最后一步会得到(10,10)
                            # 更新当前位置
                            current_pos_np_get += step_size
                        sub_inputsRag_all_np_get.append(sub_inputsRag_j_np_get)
                        # 最后会得到（batch，10,10）

                    # 在垂直方向堆叠，形成 RaggedTensor
                    sub_inputsRag_np_get = np.stack(sub_inputsRag_all_np_get, axis=0)  # （nbatch, 10, 10)
                    # print('sub_inputsRag_np_get:', np.shape(sub_inputsRag_np_get))
                    # 形成这个时刻的mask （batch，10，10）
                    mask_atime_get = np.ones([batch_get, 10, 10], dtype=bool)
                    for j_mask in range(batch_get):
                        for i_mask in range(10):
                            if sub_inputsRag_np_get[j_mask, i_mask, 0] == 0:
                                # 说明这个交互对象没有/agent没有
                                mask_atime_get[j_mask, i_mask, :] = False
                                mask_atime_get[j_mask, :, i_mask] = False
                    mask_atime_all_get.append(mask_atime_get)
                mask_atime_all_new_get = np.stack(mask_atime_all_get, axis=0)  # （21, nbatch, 10, 10)

                # 另一种做法
                # # 时间维度上的mask （batch,21,21）
                # # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
                # for i_batch in range(batch_get):
                #     for i_time in range(time_get):
                #         if X_attention[i_batch, i_time, 0] == 0:
                #             # 说明这个agent还没往前走，那么所有的都是要掩码的
                #             mask_times_get[i_batch, i_time, :] = False
                #             mask_times_get[i_batch, :, i_time] = False


                # 时间维度上的mask （batch,21,21）
                # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
                for i_batch in range(batch_get):
                    if X_attention[i_batch][20][0] == 0:  # k_ob_lstm (batch,21,46)
                        # 说明这个agent还没往前走或者是无效的，那么所有的都是要掩码的
                        mask_times_get[i_batch, :, :] = False
                    else:
                        # 说明这个agent往前走了，只需要找到在哪个时刻往前走的就可以了
                        for time_i_batch in range(time_get):
                            if X_attention[i_batch][time_i_batch][0] != 0:
                                mask_times_get[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break  # 退出循环，已经找到第一个有效的时刻
                # print('测试下时间维度上的mask对不对,batch 0：',k_ob_lstm[0],mask_times[0])
                # print('测试下一个时间上的mask对不对,batch 0,最后一个时刻：', mask_atime_all_new[20][0])

                new_map = {
                    train_model[k].X_attention: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].Mask_onetime_all: mask_atime_all_new_get,
                    train_model[k].Mask_alltime: mask_times_get,
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)
                } # 创建一个新的字典 new_map，用于将数据传递给智能体 k 的策略网络，包括观察数据和行动数据。这些数据是从 obs 和 actions 中选择性地提取的，以适应当前智能体的范围。
                log_pac = sess.run(self.log_pac[k], feed_dict=new_map) # 使用 TensorFlow 的 sess.run 方法计算智能体 k 的对数动作概率 log_pac，并将 new_map 中的数据传递给 TensorFlow 图中的占位符。这一步计算每个智能体的对数动作概率。
                if scale[k] == 1:
                    action_prob.append(log_pac) # 和前一个agent的policy network是同一个,将计算得到的对数动作概率 log_pac 添加到 action_prob 列表中。
                else:
                    log_pac = np.split(log_pac, scale[k], axis=0) # 如果 scale[k] 不等于1，表示当前智能体有多个子智能体，每个子智能体都有自己独立的策略网络。在这种情况下，需要将计算得到的对数动作概率按子智能体进行拆分，以便后续的处理可以针对每个子智能体独立进行。
                    action_prob += log_pac # action_prob += log_pac: 将拆分后的子概率添加到 action_prob 列表中
            return action_prob # 返回包含每个智能体对数动作概率的列表 action_prob。

        self.get_log_action_prob = get_log_action_prob # 将一个名为 get_log_action_prob 的函数赋值给了类的属性 self.get_log_action_prob。具体来说，它将 get_log_action_prob 函数变成了该类的一个可调用的方法。

        # 它的作用是计算智能体在给定观察值（obs）和行动（actions）的情况下，根据策略网络中的步骤模型（step_model）计算出的对数行动概率。接着，将这个函数赋值给了类的属性 self.get_log_action_prob_step，使其成为类的一个可调用方法。
        def get_log_action_prob_step(obs, actions):
            action_prob = [] # 创建一个空列表 action_prob，用于存储每个智能体的对数行动概率。
            for k in range(num_agents):
                action_prob.append(step_model[k].step_log_prob(obs[k], actions[k])) # 对于每个智能体，调用其对应的步骤模型 step_model 中的 step_log_prob 方法，传递该智能体的观察值 obs[k] 和行动 actions[k] 作为参数，并将计算得到的对数行动概率添加到 action_prob 列表中。
            return action_prob # 返回包含每个智能体对应的对数行动概率的列表

        self.get_log_action_prob_step = get_log_action_prob_step

        # 虽然get_log_action_prob,get_log_action_prob_step这两个函数的输入参数都是所有智能体的观察和动作数据，但它们在计算方式上有所不同，一个是集体计算，一个是个体计算。

        def save(save_path): # 函数的作用是将神经网络模型的参数保存到磁盘上的文件中 # save_path 是要保存参数的文件路径。
            ps = sess.run(params_flat) # sess.run(params_flat) 通过 TensorFlow 会话执行 params_flat，这个操作会获取当前神经网络模型的所有参数，并将它们的值转换为 NumPy 数组。
            joblib.dump(ps, save_path) # joblib.dump(ps, save_path) 使用 joblib 库将参数数组 ps 保存到指定的文件路径 save_path 中。

        def load(load_path): # load(load_path) 函数的作用是从磁盘上的文件中加载神经网络模型的参数 # load_path 是要加载参数的文件路径。
            loaded_params = joblib.load(load_path) # loaded_params = joblib.load(load_path) 使用 joblib 库从文件路径 load_path 中加载参数数组，并将其存储在 loaded_params 变量中。
            restores = [] # 这个列表的作用是用来存储 TensorFlow 的操作，这些操作将被用于将已加载的参数值赋值给神经网络模型的对应参数。
            for p, loaded_p in zip(params_flat, loaded_params): # 接下来，通过循环遍历 params_flat 中的参数和 loaded_params 中的已加载参数，将已加载的参数值赋值给神经网络模型的对应参数。这是通过 p.assign(loaded_p) 操作完成的。
                restores.append(p.assign(loaded_p))
            sess.run(restores) # 通过 sess.run(restores) 执行参数赋值操作，将已加载的参数值应用到当前的神经网络模型中，以便后续的使用。 这种方式可以方便地批量处理参数的赋值操作。

        # save load这两个函数用于实现神经网络模型参数的保存和加载，可以在训练期间保存模型的参数，以便以后恢复模型状态或进行迁移学习。

        # 下面这段代码主要是为了构建一个多智能体的训练和推断接口，方便在训练和测试过程中与多智能体模型进行交互。
        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        # self.train, self.clone, self.save, self.load, self.train_model, self.step_model: 这些属性分别是之前定义的 train, clone, save, load, train_model, 和 step_model 函数或模型的引用，它们将在外部使用，方便与智能体模型进行交互。

        # 这个函数是一个接口，用于在给定所有智能体的观察 ob 和动作 av 的情况下，执行一个步骤并返回智能体的动作、值函数估计和状态。
        # 它首先将所有智能体的观察 ob 连接起来以形成一个全局观察 obs。
        # 然后，它循环遍历每个智能体（通过 num_agents 控制），将其他智能体的动作 av 连接起来以形成一个全局动作 a_v。
        # 对于每个智能体，它调用 step_model[k] 中的 step 方法，传递智能体自己的观察 ob[k]、全局观察 obs 和全局动作 a_v，并收集智能体的动作、值函数估计和状态，最后返回这些信息。
        # self.obs_lstm, self.obs, self.actions
        def step(ob_lstm, ob, av, *_args, **_kwargs): # 输入的是一个场景内一个时刻的所有agent的观察值(19个值)和动作(2个值)
            a, v, s, att_weights_spatial, att_weights_temporal = [], [], [], [], []
            # print('airl_ob_lstm的shape：', np.shape(ob_lstm))  # (8, 10, 21, 46)  # evaluate (8, 1, 21, 46)
            # print('airl_ob的shape：',np.shape(ob))  # (8, 10, 46)  # evaluate (8, 1, 46)
            # print('airl_av的shape：', np.shape(av))  # (8, 10, 2)  # evaluate (8, 1, 2)
            obs = np.concatenate(ob, axis=1)
            obs_lstm = np.concatenate(ob_lstm, axis=1)
            # print('airl_obs的shape：', np.shape(obs))  # (10, 8*46)  # evaluate (1, 368)
            # print('airl_obs_lstm的shape：', np.shape(obs_lstm))  # (10, 8*21, 46)  # evaluate  (1, 168, 46)
            for k in range(num_agents):
                a_v = np.concatenate([av[i]
                                      for i in range(num_agents) if i != k], axis=1)
                # print('airl_ob_lstmk:', np.shape(ob_lstm[k]))  # (10, 21, 46)  # evaluate (1, 21, 46)
                # print('airl_obk:',np.shape(ob[k]))  # (10, 46)  # evaluate (1, 46)
                # print('airl_obs_lstm:', np.shape(obs_lstm))  # (10, 8*21, 46)  # evaluate (1, 168, 46)
                # print('airl_a_v:', np.shape(a_v))  # (10, 14)  # evaluate (1, 14)
                is_training = True

                # 这里处理ob_lstm（8, batch, 21, 46）得到mask_atime（21，batch,10,10）, mask_times（batch,21,21）
                k_ob_lstm = ob_lstm[k]  # (batch,21,46)
                # print('k_ob_lstm', np.shape(k_ob_lstm))
                num_batch = k_ob_lstm.shape[0]
                # 提取出第二个维度的大小，即 21
                num_time = k_ob_lstm.shape[1]
                # print('num_batch:', num_batch, 'num_time:', num_time)
                num_features = 10  # 每个序列的参数的个数
                step_sizes_np = [10, 4, 4, 4, 4, 4, 4, 4, 4, 4]
                mask_atime_all = []  # 存放对于第k个agent来说，每一个时刻的mask矩阵，shape为（21，batch，10,10）
                mask_times = np.ones([num_batch, 21, 21], dtype=bool)  # 存放对于第k个agent来说，在时刻维度的mask矩阵，shape为（batch，21,21）
                for time_i in range(num_time):
                    k_ob_lstm_ONE_TIME = k_ob_lstm[:, time_i, :]  # (batch,46)
                    # print('k_ob_lstm_ONE_TIME:', np.shape(k_ob_lstm_ONE_TIME))
                    # 把k_ob_lstm_ONE_TIME数据处理为shape为【batch，10，10】，第一个10代表序列数，第二个10代表每个序列的参数的个数
                    # 定义每个时间步长的特征数
                    # 以下是numpy版本下的代码
                    sub_inputsRag_all_np = []  # 存放每一个时刻的拆分之后的数据
                    # 遍历每个样本
                    for j in range(num_batch):
                        # 创建一个零张量
                        sub_inputsRag_j_np = np.zeros([0, num_features], dtype=np.float32)
                        # 记录当前位置
                        current_pos_np = 0
                        # 遍历每个特征的长度
                        for k_, step_size in enumerate(step_sizes_np):
                            # 截取当前时间步长的特征
                            feature_slice = k_ob_lstm_ONE_TIME[j, current_pos_np: current_pos_np + step_size]
                            # print('feature_slice:',np.shape(feature_slice))
                            # 如果特征长度小于 10，使用 tf.pad 在末尾填充 0
                            if feature_slice.shape[0] < num_features:
                                pad_size = num_features - feature_slice.shape[0]
                                feature_slice = np.append(feature_slice, np.zeros(pad_size))
                            # 在垂直方向堆叠
                            # print('feature_slice:',np.shape(feature_slice))
                            sub_inputsRag_j_np = np.concatenate(
                                [sub_inputsRag_j_np, np.expand_dims(feature_slice, axis=0)], axis=0)
                            # 最后一步会得到(10,10)
                            # 更新当前位置
                            current_pos_np += step_size
                        sub_inputsRag_all_np.append(sub_inputsRag_j_np)
                        # 最后会得到（batch，10,10）

                    # 在垂直方向堆叠，形成 RaggedTensor
                    sub_inputsRag_np = np.stack(sub_inputsRag_all_np, axis=0)  # （nbatch, 10, 10)
                    # print('sub_inputsRag_np:', np.shape(sub_inputsRag_np))
                    # 形成这个时刻的mask （batch，10，10）
                    mask_atime = np.ones([num_batch, 10, 10], dtype=bool)
                    for j_mask in range(num_batch):
                        for i_mask in range(10):
                            if sub_inputsRag_np[j_mask, i_mask, 0] == 0:
                                # 说明这个交互对象没有/agent没有
                                mask_atime[j_mask, i_mask, :] = False
                                mask_atime[j_mask, :, i_mask] = False

                    mask_atime_all.append(mask_atime)
                mask_atime_all_new = np.stack(mask_atime_all, axis=0)  # （21, nbatch, 10, 10)

                # 时间维度上的mask （batch,21,21）
                # trj_GO_STEP (batch,8) # 每个batch的每个agent的前进的步数
                for i_batch in range(num_batch):
                    if k_ob_lstm[i_batch][20][0] == 0:  # k_ob_lstm (batch,21,46)
                        # 说明这个agent还没往前走或者是无效的，那么所有的都是要掩码的
                        mask_times[i_batch, :, :] = False
                    else:
                        # 说明这个agent往前走了，只需要找到在哪个时刻往前走的就可以了
                        for time_i_batch in range(num_time):
                            if k_ob_lstm[i_batch][time_i_batch][0] != 0:
                                mask_times[i_batch, 0:time_i_batch, 0:time_i_batch] = False
                                break  # 退出循环，已经找到第一个有效的时刻
                # print('测试下时间维度上的mask对不对,batch 0：',k_ob_lstm[0],mask_times[0])
                # print('测试下一个时间上的mask对不对,batch 0,最后一个时刻：', mask_atime_all_new[20][0])

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
        self.step = step # 将之前定义的 step 函数赋值给了对象的属性，以便在外部使用。
        self.attention_step = attention_step
        # value(obs, av): 这个函数与 step 函数类似，不同之处在于它只计算值函数估计而不执行动作。
        # 它也循环遍历每个智能体，将其他智能体的动作 av 连接起来，
        # 然后调用 step_model[k] 中的 value 方法，传递智能体自己的观察 ob[k]、全局观察 obs 和全局动作 a_v，并收集智能体的值函数估计，最后返回这些值函数估计。
        def value(obs, av):
            # print('value中的obs：',np.shape(obs), 'av:',np.shape(av)) # value中的obs： (8, 10, 21, 46) av: (8, 10, 2)
            v = []
            ob = np.concatenate(obs, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([av[i]
                                      for i in range(num_agents) if i != k], axis=1)
                # print('每个agentvalue中的obs：', np.shape(obs), 'a_v:', np.shape(a_v))
                v_ = step_model[k].value(ob, a_v)
                v.append(v_)
            return v

        self.value = value # 将之前定义的 value 函数赋值给了对象的属性，以便在外部使用。
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)] # self.initial_state: 这是一个包含多个智能体初始状态的列表，每个智能体的初始状态由 step_model[k].initial_state 提供。这些初始状态在训练和测试时可能用于重置智能体的状态。

# 这段代码定义了一个名为 Runner 的类，用于在环境中与模型进行交互并生成训练数据
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
        ] # self.batch_ob_shape 和 self.batch_ac_shape：分别存储了观察和动作的批处理形状信息。这些信息在构建训练批次时会用到。
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ] # 存储一个包含多个智能体观察的列表。每个元素是一个二维数组，用于存储观察数据。
        self.obs_lstm = [
            np.zeros((nenv, 21, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]  # 存储一个包含多个智能体观察的列表。每个元素是一个三维数组，用于存储有时续的观察数据。
        self.actions = [np.zeros((nenv, n_ac )) for _ in range(self.num_agents)]   ###############
        # obs = env.reset() # 某一个时刻的观察值,19个值 # , reset_infos
        scenario_test_name = 0  # 训练的时候不会用到
        traning_label = True # 训练的标签
        reset_inf = []
        for i in range(self.nenv):
            # 10个线程上的重置信息
            reset_inf.append([scenario_test_name,traning_label])
        print('reset_inf:', reset_inf)
        obs_lstm, obs, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, rew_n_social_generate, collide_situation = env.reset(reset_inf)  # 某一个时刻的观察值,22个值，和每个agent开始的step
        # obs_lstm, obs, ini_steps, ini_obs, reset_infos, ini_obs_lstm
        # 10个并行环境中的obs： (18, 10, 22) ini_steps: (18, 10, 1) ini_obs: (18, 10, 22)
        # print('第几次初始化场景参数呢？？？？？')
        # # print('run初始化中的obs:', np.shape(obs))
        self.update_obs(obs)
        self.update_obs_lstm(obs_lstm)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps # 存储每个环境中执行的时间步数。
        self.states = model.initial_state # 初始化模型的状态信息。【【】，【】，【】，【】，【】，【】，【】，【】】空
        #self.n_actions = [env.action_space[k].n for k in range(self.num_agents)]
        self.dones = [np.array([False for _ in range(self.nenv)]) for k in range(self.num_agents)]  # (8,10)

        # 记录每个agent开始的step
        self.ini_steps = ini_step_n  # 每个cpu线程的每个agent开始的步长
        self.ini_obs = ini_obs  # 记录这次循环的时候，初始化的场景的obs
        self.ini_obs_lstm = ini_obs_lstm  # 记录这次循环的时候，初始化的场景的ini_obs_lstm
        self.diedai_cishu = 0
        self.reset_infos = reset_infos
        self.N = [np.array([0 for _ in range(nenv)])]  # (1, 10) N是记录在cpui线程下这个agent开始之前已经迭代过的步数
        self.env_GO_STEP = [np.array([0]) for k in range(10)]  # (10, 1) 每个环境已经迭代的步数
        self.trj_GO_STEP = [np.array([0 for _ in range(self.num_agents)]) for k in
                            range(self.nenv)]  # (10, 8) 每个轨迹从ini_step开始已经走得步数

        # self.reset_infos = reset_infos
        self.update = 0 # 初始化的大迭代次数 update
        # print('run初始化中的参数:', np.shape(self.ini_steps), np.shape(self.ini_obs), np.shape(self.reset_infos),
        #       np.shape(self.N), np.shape(self.env_GO_STEP), np.shape(self.trj_GO_STEP), np.shape(self.ini_obs_lstm))
        #   (8, 10, 1) (8, 10, 46) (1, 10, 1) (1, 10) (10, 1) (10, 8) (8, 10, 21, 46)
        #   (8, 10, 1) (8, 10, 46) (1, 10, 1) (1, 10) (10, 1) (10, 8) (8, 10, 21, 46)
        # for agent_id in range(self.num_agents):
        #     if agent_id < 2:
        #         # 设置 agent.id < 36 的第二个动作为 0
        #         self.actions[agent_id][:, 1] = 0.0
        #     else:
        #         # 设置 agent.id >= 36 的第二个动作在 [164.09, 176.82] 之间
        #         self.actions[agent_id][:, 1] = np.random.uniform(164.09, 176.82, size=nenv)
        # 存储一个包含多个智能体的完成状态的列表。每个元素是一个布尔数组，用于表示每个环境是否完成。

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs


    def update_obs_lstm(self, obs_lstm):
        # TODO: Potentially useful for stacking.
        self.obs_lstm = obs_lstm


    # 用于在环境中运行多个时间步骤并收集数据以用于训练。以下是每行代码的详细解释：
    def run(self, update):
        # 初始化一系列空列表，这些列表将用于存储每个智能体的数据，包括观察、奖励、动作、值函数估计等。
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
        mb_states = self.states # mb_states = self.states：将当前的模型状态存储在 mb_states 中，以便后续使用。


        for n in range(self.nsteps): # 使用循环迭代执行 nsteps 次，每次执行一个时间步骤，收集数据。
            # print('runner中的第几次nsteps:', n + (update - 1) * self.nsteps, self.ini_steps[1][1][0], self.obs[1][1][0],self.reset_infos[0][1][0])
            for i in range(self.nenv):  # 这里的GO_STEP存储每个cpu中环境的每个agent走了多少步了
                if self.reset_infos[0][i][0] == True:  # 说明第i个线程的环境reset了
                    self.N[0][i] = n + (update - 1) * self.nsteps  # 需要重新判断的当前时刻   N是记录在这个agent开始之前已经迭代过的步数
                env_go_step = n + (update - 1) * self.nsteps - self.N[0][i]
                self.env_GO_STEP[i][0] = env_go_step  # self.env_GO_STEP (10,1) 环境累计前进了多少步
                for k in range(self.num_agents):
                    trj_go_step = n + (update - 1) * self.nsteps - self.N[0][i] - self.ini_steps[k][i][0]
                    if trj_go_step > 0:
                        self.trj_GO_STEP[i][k] = trj_go_step  # self.trj_GO_STEP (10,8) 环境中的轨迹前进了多少步
                    if trj_go_step <= 0:
                        # trj_go_step = 0
                        self.trj_GO_STEP[i][k] = trj_go_step
            ini_update_inf = [np.array([0 for _ in range(self.num_agents)]) for k in range(self.nenv)]  # (10, 8) True代表初始状态需要更新，False代表不需要
            for k in range(self.num_agents):
                for i in range(self.nenv):  # self.reset_infos (1, 10, 1)
                    if self.reset_infos[0][i][0] == True:  # 说明第i个线程的环境reset了
                        self.N[0][i] = n + (update - 1) * self.nsteps  # 需要重新判断的当前时刻   N是记录在这个agent开始之前已经迭代过的步数
                        # if i == 2 and k == 0:
                            # print('cpu 2 agent 0 重置', self.reset_infos[0][i][0], '重置时刻为', self.N[0][i], '轨迹是否done：',
                            #       self.dones[k][i])
                    ini_step_k = self.ini_steps[k][i][0]  # (8, 10, 1) cpui agentk的开始步长
                    if self.ini_obs[k][i][0] == 0:  # 说明这个cpui的agentk是空的，没有值 agent无效
                        self.obs[k][i] = np.zeros(57)
                        self.actions[k][i] = np.zeros(2)
                        self.obs_lstm[k][i][:] = 0  # 每个历史时刻和当前时刻都为0
                        ini_update_inf[i][k] = False
                    else:
                        if n + (update - 1) * self.nsteps - self.N[0][i] < ini_step_k:  # ini_steps的shape是(18, 10, 1) 还未到这个agent的开始时刻
                            self.obs[k][i] = np.zeros(57)
                            self.actions[k][i] = np.zeros(2)
                            self.obs_lstm[k][i][:] = 0  # 每个历史时刻和当前时刻都为0
                            ini_update_inf[i][k] = False
                        elif n + (update - 1) * self.nsteps - self.N[0][i] == ini_step_k:
                            # 记录下来每一个cpu环境初始化状态的agent的编号
                            ini_update_inf[i][k] = True
                            # 因为这个时候可能有的agent已经进入环境，所以iniobs不再是原来的了
                            self.obs[k][i] = self.ini_obs[k][i]  # 如果到了这个开始的时刻，那么就把初始观测值赋值给cpui 的 每一个agentk 的obs
                            self.actions[k][i] = np.zeros(2)
                            self.obs_lstm[k][i] = self.ini_obs_lstm[k][i]
                        else:
                            ini_update_inf[i][k] = False

                    # if self.dones[k][i] == True:
                    #     self.obs[k][i] = np.zeros(22)  # 如果cpu i agent k已经结束，那么就把0赋值给cpui 的 每一个agentk 的obs
                    # self.actions[k][i] = np.zeros(2)
                    # if i ==1 and k==1:
                    #     # print('开始时刻runner中的第几次nsteps:', n + (update - 1) * self.nsteps, self.ini_steps[1][1][0],
                    #       self.obs[1][1][0], self.reset_infos[0][1][0])

            # 对这一步，每一个环境下的每一个agent的状态都更新，到时候根据具体的情况判断是否更新每个cpu
            ini_obs_old_list = []
            # print('ini_update_inf:',np.shape(ini_update_inf))
            for i in range(self.nenv):
                ini_obs_old_list.append(
                    [np.concatenate((self.actions[k_][i], np.array([self.env_GO_STEP[i][0]]),
                                     np.array([self.trj_GO_STEP[i][k_]]),
                                     np.array([self.ini_steps[k_][i][0]]),
                                     self.ini_obs[k_][i], np.array([ini_update_inf[i][k_]]))) for k_ in range(self.num_agents)])
            # print('ini_obs_old_list:', np.shape(ini_obs_old_list))  # (10, 8, 63)
            obs_lstm_nowstep, obs_nowstep, collide_label_nowstep = self.env.ini_obs_update(ini_obs_old_list)  # 初始观测值输入  # , reset_info
            # print('obs_lstm_nowstep:', np.shape(obs_lstm_nowstep), type(obs_lstm_nowstep),
            #      'obs_nowstep:',np.shape(obs_nowstep),type(obs_nowstep))
            # obs_lstm_nowstep: (8, 10, 21, 57) <class 'list'> obs_nowstep: (8, 10, 57) <class 'list'>
            # print('self.obs_lstm：',np.shape(self.obs_lstm))  #  (8, 10, 21, 57)
            # print('self.obs：', np.shape(self.obs))   # (8, 10, 57)
            # obs_lstm_nowstep: (8, 10, 21, 57) obs_nowstep: (8, 10, 57)

            #print('self.obs_lstm:',type(self.obs_lstm),np.shape(self.obs_lstm))
            for i in range(self.nenv):
                for k in range(self.num_agents):
                    if ini_update_inf[i][k] == True:
                        for j in range(len(self.obs_lstm)):
                            self.obs_lstm[j][i] = obs_lstm_nowstep[j][i] # 对于这个cpu环境，每个agent的状态都得更新
                            self.obs[j][i] = obs_nowstep[j][i]  # 对于这个cpu环境，每个agent的状态都得更新
                        self.actions[k][i] = np.zeros(2)  # 对于这个cpu环境，只需要刚开始进入交叉口的agent的动作进行更新

            # # print('这里的obs：',np.shape(self.obs), self.obs) # (8, 10, 46)
            # # print('这里的obs_lstm：', np.shape(self.obs_lstm), self.obs_lstm)   # (8, 10, 21, 46)
            actions, values, states, atten_weights_spatial, atten_weights_temporal = self.model.step(self.obs_lstm, self.obs, self.actions) # 使用模型的 step 方法获取(所有)每一个智能体的动作、值函数估计和状态信息。且有线程作用在其中 输入的是当前的状态(8,10,18)和初始的动作(8,10,2)

            # model.step得到了当前一个场景这一步每一个agent的动作,值函数,还有一个空的state,其中因为在神经网络的最后一层使用了 tanh（双曲正切）激活函数，这会将输出缩放到 [-1, 1] 范围内，以确保行动的值在合理范围内。
            # 也就是说action中的加速度和角度变化都是[-1,1]之间,再更新状态时,需要对其进行转化
            # # print('run运行多线程中的self.obs:',np.shape(self.obs),self.obs[0]) # obs的shape是 (67, 10, 19)
            # # print('run运行多线程中的self.actions:', np.shape(self.actions), self.actions[0]) # # actions的shape是 (8, 10, 2)
            # print('run运行多线程中的atten_weights:',np.shape(atten_weights))  # (8,21,10,10,10)

            self.actions = actions
            # print('run过程中的多线程的self.actions :', np.shape(self.actions))  # (8, 10, 2)

            for k in range(self.num_agents):
                # 循环遍历每个智能体，将当前时间步骤的观察、动作、值函数估计、完成状态等数据添加到相应的数据列表中。
                mb_obs_lstm[k].append(np.copy(self.obs_lstm[k]))  # mb_obs_lstm: (8, 10, 21, 46)
                mb_obs[k].append(np.copy(self.obs[k]))  # obs: (8, 10, 46)
                # mb_actions[k].append(self.actions[k])  # actions: (8, 10, 2)
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])



            # # print('run过程中的mb_obs', np.shape(mb_obs), mb_obs)  # (8, 0)
            # # print('run过程中的obs', np.shape(self.obs), self.obs)  # (8, 10, 22)
            # # print('run过程中的actions', np.shape(actions), actions)  # (8, 10, 2)
            actions_list = [] # 其中包含每个环境中的所有智能体的动作。这是为了传递给环境的 step 方法。

            for i in range(self.nenv):  # 这里的GO_STEP存储每个cpu中环境的每个agent走了多少步了
                # env_go_step = n + (update - 1) * self.nsteps - self.N[0][i]
                # self.env_GO_STEP[i][0] = env_go_step  # self.env_GO_STEP (10,1) 环境累计前进了多少步
                for k in range(self.num_agents):
                    trj_go_step = n + (update - 1) * self.nsteps - self.N[0][i] - self.ini_steps[k][i][0]
                    if trj_go_step > 0:
                        self.trj_GO_STEP[i][k] = trj_go_step  # self.trj_GO_STEP (10,8) 环境中的轨迹前进了多少步
                    if trj_go_step <= 0:
                        # trj_go_step = 0
                        self.trj_GO_STEP[i][k] = trj_go_step

                # 10个线程上这个场景内所有agent的动作
                actions_list.append([np.concatenate((self.actions[k][i], np.array([self.env_GO_STEP[i][0]]),
                                                     np.array([self.trj_GO_STEP[i][k]]),
                                                     np.array([self.ini_steps[k][i][0]]),
                                                     self.ini_obs[k][i])) for k in
                                     range(self.num_agents)])
                # 这里的actions_list包括了动作（acc，yaw）,cpui的累计前进步数，
                # cpui中agentk的前进步数，cpui中agentk的ini_step，cpuiagentk的初始观测值

            # obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos
            # obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n
            # print('actions_list:',np.shape(actions_list))
            obs_lstm, obs, true_rewards, dones, _, ini_steps_all, ini_obs, reset_infos, ini_obs_lstm, \
            actions_new, rew_n_social_generate, collide_situation = self.env.step(actions_list)
            # 使用动作列表执行环境的 step 方法，得到下一步的新的观察、奖励和完成状态。这里的action_list是这个场景内所有agent的动作

            # print('cpu 0 agent 0 dones:',dones[0][0], obs[0][0][0]*38 - 4, obs[0][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[0][0],
            #       'cpu 0 agent 1 dones:',dones[1][0], obs[1][0][0]*38 - 4, obs[1][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[1][0],
            #       'cpu 0 agent 2 dones:',dones[2][0], obs[2][0][0]*38 - 4, obs[2][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[2][0],
            #       'cpu 0 agent 3 dones:',dones[3][0], obs[3][0][0]*38 - 4, obs[3][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[3][0],
            #       'cpu 0 agent 4 dones:',dones[4][0], obs[4][0][0]*38 - 4, obs[4][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[4][0],
            #       'cpu 0 agent 5 dones:',dones[5][0], obs[5][0][0]*38 - 4, obs[5][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[5][0],
            #       'cpu 0 agent 6 dones:',dones[6][0], obs[6][0][0]*38 - 4, obs[6][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[6][0],
            #       'cpu 0 agent 7 dones:',dones[7][0], obs[7][0][0]*38 - 4, obs[7][0][1] * 23 + 14, '环境步长：',self.env_GO_STEP[0][0],'初始步长：', ini_steps_all[7][0])
            # print('运行完了env.step，都是10个并行线程同时运行的',
            #       np.shape(obs),np.shape(true_rewards),np.shape(dones), np.shape(ini_steps_all), np.shape(ini_obs),
            #       np.shape(reset_infos),np.shape(obs_lstm),np.shape(ini_obs_lstm), np.shape(actions_new))
            # (8, 10, 57) (8, 10) (8, 10) (8, 10, 1) (8, 10, 57) (1, 10, 1) (8, 10, 21, 57) (8, 10, 21, 57) (8, 10, 2)
            #
            # print('run过程中的多线程的actions_list:',np.shape(actions_list))   #  (10, 8, 23)
            # # print('准备运行环境的step了,让我来看看运行了哪一个函数???????????????????????????')
            # # print('检查dones', dones[0][1], np.shape(dones),dones)  # Dones (6, 10)

            self.actions = actions_new

            self.reset_infos = reset_infos
            self.dones = dones
            re_obs = self.obs  # 旧的观察值
            re_obs_lstm = self.obs_lstm  # 旧的观察值
            re_actions = self.actions  # 将 self.actions 更新为新的动作，以备后续使用。
            self.update_obs(obs)  # 将新的观察值放进self.obs中
            self.update_obs_lstm(obs_lstm)  # 将新的lstm观察值放进self.obs_lstm中

            # print('cpu2 agent0：', self.reset_infos[0][2][0], '重置时刻为', self.N[0][2], '轨迹是否done：', self.dones[0][2],
            #       actions_list[2][0][2], actions_list[2][0][3], actions_list[2][0][4], actions_list[2][0][5],
            #       actions_list[2][0][6])

            for i in range(self.nenv):
                for k in range(self.num_agents):

                    if reset_infos[0][i][0] == True:  # 这个线程的环境重置了，重新reset了
                        self.ini_obs[k][i] = ini_obs[k][i]  # 重置了就需要更新这个cpu环境中场景的ini_obs
                        self.ini_obs_lstm[k][i] = ini_obs_lstm[k][i]  # 重置了就需要更新这个cpu环境中场景的ini_obs_lstm
                        # 这个cpu中重置之前的场景的下一个时刻所有agent的obs都得是0，因为重置了的obs是新的场景中的obs了
                        obs[k][i] = obs[k][i] * 0.0
                        obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                        self.actions[k][i] = self.actions[k][i] * 0.0
                    else:  # cpu i的场景没有重置，根据当前场景中的agent的done是否为True，决定是否要i将下一时刻的obs赋值为0，如果是True，就赋值为0
                        if dones[k][i] == True:  # 第ni线程的第k个agent的done为True，说明这个agent在执行完动作之后新的时刻会驶出交叉口大边界了，或者上一时刻的点和终点之间的距离已经小于0.01m了
                            obs[k][i] = obs[k][i] * 0.0  # 处理完成状态，如果一个智能体完成了，将其观察数据置为零。
                            obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                            self.obs[k][i] = obs[k][i] * 0.0
                            self.obs_lstm[k][i] = obs_lstm[k][i] * 0.0
                            self.actions[k][i] = self.actions[k][i] * 0.0
                            # true_rewards[k][i] = true_rewards[k][i] * 0

            for k in range(self.num_agents):
                # 循环遍历每个智能体，将当前时间步骤的观察、动作、值函数估计、完成状态等数据添加到相应的数据列表中。
                mb_actions[k].append(self.actions[k])  # actions: (8, 10, 2)

            self.ini_steps = ini_steps_all  # 每一步都会计算10个cpu当前场景每个agent的ini_steps，即使reset也会计算，所以可以直接替换掉

            # # print('run过程中的obs, true_rewards, dones, _:',np.shape(obs), np.shape(true_rewards), np.shape(dones), np.shape(_)) # run过程中的obs, true_rewards, dones, _: (67, 10, 19) (67, 10) (67, 10) (10,)
            # 在对抗逆强化学习（Adversarial Inverse Reinforcement Learning，AIRL）或其他逆强化学习方法中，执行环境的 step 方法时将奖励函数设置为0通常是为了在环境中执行智能体的行为，而不依赖于先前定义的奖励函数。
            # 逆强化学习的主要目标之一是从专家演示中推断出一个合适的奖励函数，这个奖励函数可以用于训练智能体。因此，初始阶段，我们并不知道真实的奖励函数是什么，所以将奖励函数设置为0只是为了在环境中执行行为以生成轨迹数据。
            # self.reset_infos = reset_infos

            for k in range(self.num_agents):
                mb_obs_next[k].append(np.copy(obs[k])) # 将新的观察数据（状态）复制并添加到经验数据列表 mb_obs_next[k] 中，记录每个智能体的下一个观察数据。
                mb_obs_next_lstm[k].append(np.copy(obs_lstm[k]))
            re_obs_next_lstm = obs_lstm  # 将 re_obs_next_lstm 更新为新的观察数据，以备后续使用。
            re_obs_next = obs # 将 re_obs_next 更新为新的观察数据，以备后续使用。
            re_path_prob = np.zeros(self.num_agents) # 创建一个长度为 num_agents 的零数组，用于存储路径概率信息。在当前代码中，这个数组被初始化为零，但后续可能会用于其他用途。 self.model.get_log_action_prob_step(re_obs, re_actions)  # [num_agent, nenv, 1]
            #re_actions_onehot = [multionehot(re_actions[k], self.n_actions[k]) for k in range(self.num_agents)]

            # get reward from discriminator

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

            def Cal_GT_crash(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                       Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                       Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

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
                        GT_value = -2  # 非常不安全
                        dis_min = 0
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
                                GT_value = 1  # 安全
                                dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
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
                                    # 2.1 AGENT 会把交互对象看做 有冲突的对象;交互对象也会把agent看做 有冲突的对象
                                    # 判断当agent到交点的时候，neig在哪，如果撞了，GT_value=-1,否则None
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录快车走到交点的路程中，两车之间轨迹点的距离，步长0.5s（不考虑车宽，因为是否碰撞上已经考虑了）
                                    if t_agent < t_neig:
                                        # agent先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_agent > 23:
                                            t_agent = 23
                                        else:
                                            t_agent = t_agent
                                        # try:
                                        time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_agent:
                                            time = np.arange(0, t_agent + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_agent, 0.5)
                                            time = np.append(time, t_agent)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离
                                    else:
                                        # neig先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_neig > 23:
                                            t_neig = 23
                                        else:
                                            t_neig = t_neig
                                        # try:
                                        time_n_0 = int(t_neig / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_neig:
                                            time = np.arange(0, t_neig + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_neig, 0.5)
                                            time = np.append(time, t_neig)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                elif dot_product_agent >= 0 and dot_product_neig < 0:
                                    # 2.2 agent把neig看做冲突对象，但是neig不把agent看做冲突对象，仍然需要判断在agent到冲突点的路程中，是否会发生碰撞，以及距离的大小
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录agent车走到交点的路程中，两车之间轨迹点的距离（不考虑车宽，因为是否碰撞上已经考虑了）
                                    # agent会到达冲突点，neig不会到达冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞

                                    if t_agent > 23:
                                        t_agent = 23
                                    else:
                                        t_agent = t_agent
                                    # try:
                                    time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_agent:
                                        time = np.arange(0, t_agent + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_agent, 0.5)
                                        time = np.append(time, t_agent)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t, Agent_y_t,
                                                                                                veh_length, veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t, Jiaohu_y_t,
                                                                                               veh_length, veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                else:
                                    # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                                    # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                                    GT_value = 0
                                    dis_min = 100000



                        else:
                            # neig不是前车 2.4.2
                            GT_value = 0  # 不看做交互，因为当前时刻已经没有碰撞了
                            dis_min = 100000


                else:
                    GT_value = 0  # 不交互
                    dis_min = 100000

                return GT_value, dis_min

            def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                       Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                       Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

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

            if self.disc_type == 'decentralized':
                rewards = [] # rewards = [] 和 report_rewards = []：创建用于存储判别器奖励的空列表。
                report_rewards = []
                for k in range(self.num_agents):
                    # 调用判别器的 get_reward 方法计算判别器奖励，将其添加到 rewards 列表中。这个奖励通常用于更新策略。
                    # 没整明白 !
                    # print('判别器的输入格式：', 're_obs_lstm:', np.shape(re_obs_lstm[k]), type(re_obs_lstm[k]))  # (10, 21, 46)
                    # print('判别器的输入格式：', 're_actions:', np.shape(re_actions[k]), type(re_actions[k]))  # (10, 2)
                    # print('判别器的输入格式：', 're_obs_next:', np.shape(re_obs_next_lstm[k]), type(re_obs_next_lstm[k]))  # (10, 21, 46)
                    # print('判别器的输入格式：', 're_path_prob:', np.shape(re_path_prob[k]), type(re_path_prob[k]))  # 0.0
                    if k <= 2:
                        direction_agent = 'left'
                    else:
                        direction_agent = 'straight'
                    batch_num = np.shape(re_obs_lstm[k])[0]  # np.shape(re_obs_lstm[k])[0] np.shape(re_obs[k])[0]
                    rew_social_allbatch = rew_n_social_generate[k]  # 存放这一个agent 所有batch的参数
                    canshu_social_allbatch_array = np.array(rew_social_allbatch)
                    # print('canshu_social_allbatch_array_generate:',np.shape(canshu_social_allbatch_array_generate))
                    # canshu_social_allbatch_array_generate: (10, 4)
                    score, pre = self.discriminator[k].get_reward(re_obs_lstm[k],  # re_obs[k]
                                                                  re_actions[k],  ##################
                                                                  re_obs_next_lstm[k],  # re_obs_next[k]
                                                                  re_path_prob[k],
                                                                  canshu_social_allbatch_array,
                                                                  discrim_score=False)

                    # # print('判别器的输入batch_num:',batch_num)
                    # # 计算利己和利他所需要的参数
                    # # 计算利己奖励和利他奖励，然后利用网络学习参数φ，cos(φ)=利己倾向，sin(φ)利他倾向
                    # rew_input_fuyuan = re_obs_lstm[k]  # re_obs_lstm[k]  re_obs[k]
                    # rew_social_allbatch = []  # 存放这一个agent 所有batch的参数
                    # # 利己性参数-速度, 针对每一个batch来计算
                    # for i_batch in range(batch_num):
                    #     # 改成当前时刻应该更好，因为是当前时刻的奖励，过去的已经无法改变了。过去的状态可以看做是影响社交倾向的因素
                    #     # 如果是考虑历史数据的话，对于一些当前时刻无效，但历史时刻有效的数据来说，奖励就没有实际含义了
                    #     # 其实也可以有实际含义。再想想。还是不考虑了
                    #     if rew_input_fuyuan[i_batch][20][0] != 0:
                    #         use_GT = []  # 存放这个ibatch的主要交互对象的GT
                    #         # speed = np.sqrt(rew_input_fuyuan[i_batch][20][2] ** 2 + rew_input_fuyuan[i_batch][20][3] ** 2)
                    #         pianyi_distance = rew_input_fuyuan[i_batch][20][-2]
                    #         # 计算和主要交互对象的GT
                    #         # 提取代理的状态和终点坐标
                    #         agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                    #         agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                    #         agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                    #         agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                    #         agent_angle_last = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # 上一个点的前进方向
                    #
                    #         agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)
                    #
                    #         # 避免碰撞
                    #         # 计算agent和周围最密切的三个交互对象的GT
                    #         # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
                    #         # 安全奖励
                    #         rew_GT = 0
                    #         # use_GT = []  # 存储和主要交互对象的GT  会有7+2个值，包括除主agent之外所有的agent和交互的landmark，即使没有这个对象，也会赋值为0
                    #         # 计算和除主车之外所有agent以及交互的landmark车辆的GT
                    #         # 把所有的agent都考虑
                    #         for agent_k_ in range(self.num_agents):
                    #             if agent_k_ != k:
                    #                 # 这个agent不是我们正在计算的k
                    #                 if agent_k_ <= 2:
                    #                     direction_jiaohu = 'left'
                    #                 else:
                    #                     direction_jiaohu = 'straight'
                    #
                    #                 rew_input_fuyuan_agent_k_ = re_obs_lstm[agent_k_]  # re_obs_lstm[agent_k_]  re_obs[agent_k_]
                    #
                    #                 if rew_input_fuyuan_agent_k_[i_batch][20][0] != 0:
                    #                     jiaohu_agent_x = rew_input_fuyuan_agent_k_[i_batch][20][0] * 38 - 4
                    #                     jiaohu_agent_y = rew_input_fuyuan_agent_k_[i_batch][20][1] * 23 + 14
                    #                     jiaohu_agent_vx = rew_input_fuyuan_agent_k_[i_batch][20][2] * 21 - 14
                    #                     jiaohu_agent_vy = rew_input_fuyuan_agent_k_[i_batch][20][3] * 12 - 2
                    #                     jiaohu_agent_angle_last = rew_input_fuyuan_agent_k_[i_batch][20][6] * 191 - 1  # 上一个点的前进方向
                    #
                    #                     jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                    #                                                             agent_angle_last, direction_agent,
                    #                                                             jiaohu_agent_x, jiaohu_agent_y,
                    #                                                             jiaohu_agent_vx, jiaohu_agent_vy,
                    #                                                             jiaohu_agent_angle_last,
                    #                                                             direction_jiaohu)
                    #                     # if same_jiaohu_agent_GT_value is not None:
                    #                     use_GT.append(jiaohu_agent_GT_value)
                    #                 else:  # 没有这个agent
                    #                     jiaohu_agent_x = -4
                    #                     jiaohu_agent_y = 14
                    #                     jiaohu_agentk_vx = -14
                    #                     jiaohu_agent_vy = -2
                    #                     jiaohu_agent_angle_last = -1
                    #                     jiaohu_agent_GT_value = None
                    #                     # dis_min = 100000
                    #                     use_GT.append(jiaohu_agent_GT_value)
                    #
                    #         # 左侧视野的landmark
                    #         if rew_input_fuyuan[i_batch][20][38] != 0:
                    #             delta_left_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][38] * 29 - 14
                    #             delta_left_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][39] * 30 - 15
                    #             delta_left_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][40] * 35 - 21
                    #             delta_left_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][41] * 16 - 5
                    #             left_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][53] * 360 - 90
                    #             left_jiaohu_landmark_x = agent_x - delta_left_jiaohu_landmark_x
                    #             left_jiaohu_landmark_y = agent_y - delta_left_jiaohu_landmark_y
                    #             left_jiaohu_landmark_vx = agent_vx - delta_left_jiaohu_landmark_vx
                    #             left_jiaohu_landmark_vy = agent_vy - delta_left_jiaohu_landmark_vy
                    #             left_jiaohu_landmark_angle_last = left_jiaohu_landmark_angle_last
                    #         else:
                    #             left_jiaohu_landmark_x = -5
                    #             left_jiaohu_landmark_y = -3
                    #             left_jiaohu_landmark_vx = -16
                    #             left_jiaohu_landmark_vy = -10
                    #             left_jiaohu_landmark_angle_last = -90
                    #
                    #         # 右侧视野的landmark
                    #         if rew_input_fuyuan[i_batch][20][42] != 0:
                    #             delta_right_jiaohu_landmark_x = rew_input_fuyuan[i_batch][20][42] * 35 - 15
                    #             delta_right_jiaohu_landmark_y = rew_input_fuyuan[i_batch][20][43] * 29 - 15
                    #             delta_right_jiaohu_landmark_vx = rew_input_fuyuan[i_batch][20][44] * 25 - 14
                    #             delta_right_jiaohu_landmark_vy = rew_input_fuyuan[i_batch][20][45] * 17 - 7
                    #             right_jiaohu_landmark_angle_last = rew_input_fuyuan[i_batch][20][54] * 360 - 90
                    #             right_jiaohu_landmark_x = agent_x - delta_right_jiaohu_landmark_x
                    #             right_jiaohu_landmark_y = agent_y - delta_right_jiaohu_landmark_y
                    #             right_jiaohu_landmark_vx = agent_vx - delta_right_jiaohu_landmark_vx
                    #             right_jiaohu_landmark_vy = agent_vy - delta_right_jiaohu_landmark_vy
                    #             right_jiaohu_landmark_angle_last = right_jiaohu_landmark_angle_last
                    #         else:
                    #             right_jiaohu_landmark_x = -5
                    #             right_jiaohu_landmark_y = -3
                    #             right_jiaohu_landmark_vx = -16
                    #             right_jiaohu_landmark_vy = -10
                    #             right_jiaohu_landmark_angle_last = -90
                    #
                    #         if left_jiaohu_landmark_x != -5:
                    #             direction_landmark = 'landmark'
                    #             left_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                    #                                                             agent_angle_last, direction_agent,
                    #                                                             left_jiaohu_landmark_x,
                    #                                                             left_jiaohu_landmark_y,
                    #                                                             left_jiaohu_landmark_vx,
                    #                                                             left_jiaohu_landmark_vy,
                    #                                                             left_jiaohu_landmark_angle_last,
                    #                                                             direction_landmark)
                    #
                    #             use_GT.append(left_jiaohu_landmark_GT_value)
                    #         else:
                    #             left_jiaohu_landmark_GT_value = None
                    #             # dis_min = 100000
                    #             use_GT.append(left_jiaohu_landmark_GT_value)
                    #
                    #         if right_jiaohu_landmark_x != -5:
                    #             direction_landmark = 'landmark'
                    #             right_jiaohu_landmark_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                    #                                                              agent_angle_last, direction_agent,
                    #                                                              right_jiaohu_landmark_x,
                    #                                                              right_jiaohu_landmark_y,
                    #                                                              right_jiaohu_landmark_vx,
                    #                                                              right_jiaohu_landmark_vy,
                    #                                                              right_jiaohu_landmark_angle_last,
                    #                                                              direction_landmark)
                    #
                    #             use_GT.append(right_jiaohu_landmark_GT_value)
                    #
                    #         else:
                    #             right_jiaohu_landmark_GT_value = None
                    #             # dis_min = 10000
                    #             use_GT.append(right_jiaohu_landmark_GT_value)
                    #
                    #         # 计算角度波动
                    #         # 计算一些rew
                    #         # # 计算上个时刻，上上个时刻，上上上时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                    #         # if rew_input_fuyuan[i_batch][18][0] == 0:
                    #         #     # 没有上上上时刻的角度，所以判断上上个时刻的角度（一开始无论左转还是直行几乎都是直行的角度）
                    #         #     if rew_input_fuyuan[i_batch][19][0] == 0:
                    #         #         # 也没有上上时刻的角度
                    #         #         heading_angle_last3_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #         #         heading_angle_last2_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #         #     else:
                    #         #         # 有上上时刻的角度，上上上时刻的角度也用上上时刻的角度来代替
                    #         #         heading_angle_last3_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #         #         heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #         # else:
                    #         #     # 有上上上时刻的角度，所以也有上上个时刻的角度
                    #         #     heading_angle_last3_real = rew_input_fuyuan[i_batch][18][6] * 191 - 1  # [0,1]
                    #         #     heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                    #         #
                    #         # heading_angle_last1_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                    #         #
                    #         # # 计算平均值
                    #         # mean_value = np.mean(
                    #         #     [heading_angle_last1_real, heading_angle_last2_real, heading_angle_last3_real])
                    #         # # 计算每个数据与平均值的差的平方
                    #         # squared_differences = [(x - mean_value) ** 2 for x in
                    #         #                        [heading_angle_last1_real, heading_angle_last2_real,
                    #         #                         heading_angle_last3_real]]
                    #         # # 计算平方差的平均值
                    #         # mean_squared_difference = np.mean(squared_differences)
                    #         # # 计算标准差
                    #         # std_dev = np.sqrt(mean_squared_difference)
                    #         # if std_dev > 3:
                    #         #     rew_heading_std_bodong = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                    #         # else:
                    #         #     rew_heading_std_bodong = 1
                    #
                    #         # 计算steering angle正负来回变化带来的惩罚
                    #         # 上一时刻的转角，若超过均值的1个标准差，则给予惩罚之类的
                    #         penalty = 1  # 惩罚系数
                    #         delta_angle_last1 = rew_input_fuyuan[i_batch][20][56]
                    #         comfort_adj = 0  # 初始化转向角过大惩罚
                    #         if direction_agent == 'left':
                    #             left_delta_angle_last1_real = ((2.8 * (delta_angle_last1 + 1)) / 2) - 0.3
                    #             left_delta_angle_last1_realmean = 1.085
                    #             left_delta_angle_last1_realstd = 0.702
                    #             if left_delta_angle_last1_realmean - left_delta_angle_last1_realstd <= left_delta_angle_last1_real and left_delta_angle_last1_real <= left_delta_angle_last1_realmean + left_delta_angle_last1_realstd:
                    #                 comfort_adj = 0  # 不做惩罚
                    #
                    #             else:
                    #                 dis_left_delta_angle_last1 = abs(
                    #                     left_delta_angle_last1_real - left_delta_angle_last1_realmean) - left_delta_angle_last1_realstd
                    #                 if dis_left_delta_angle_last1 > left_delta_angle_last1_realstd:
                    #                     comfort_adj = -1 * penalty
                    #                 else:
                    #                     comfort_adj = -(
                    #                             dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                    #                     # 越靠近left_delta_angle_last1_realstd，惩罚越接近-1
                    #         else:
                    #             right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                    #             right_delta_angle_last1_realmean = 0.001
                    #             right_delta_angle_last1_realstd = 0.076
                    #             if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                    #                 comfort_adj = 0  # 不做惩罚
                    #
                    #             else:
                    #                 dis_right_delta_angle_last1 = abs(right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                    #                 if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                    #                     comfort_adj = -1 * penalty
                    #                 else:
                    #                     comfort_adj = -(dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                    #                     # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1
                    #
                    #         # 利己-效率
                    #         rew_avespeed = agent_v / 6.8  # 除以85分位速度
                    #         # 利己-车道偏移
                    #         rew_lane_pianyi = pianyi_distance
                    #
                    #         # 利他-GT
                    #         # print("use_GT：",use_GT)  # 9个元素的list，例如[None, None, None, None, None, None, None, 0.32294667405015254, None]
                    #         use_GT_list_0 = [x for x in use_GT if x is not None] # 不为None的list，例如[0.32294667405015254]
                    #         use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                    #         rew_minGT_mapped = 0
                    #         print('use_GT_list:',use_GT_list)
                    #         if len(use_GT_list) != 0:
                    #             # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                    #             rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                    #             # print('rew_minGT:',rew_minGT)  # 最小值，数据0.32294667405015254
                    #             if rew_minGT <= 1.5:
                    #                 # 归一化
                    #                 normalized_data = (rew_minGT - 0) / (1.5 - 0)
                    #                 # 映射到目标范围
                    #                 rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
                    #             elif 1.5 < rew_minGT < 3:
                    #                 # 归一化
                    #                 normalized_data = (rew_minGT - 1.5) / (3 - 1.5)
                    #
                    #                 # 映射到目标范围
                    #                 rew_minGT_mapped = normalized_data * (0.5 - 0.25) + 0.25
                    #             elif 3 <= rew_minGT <= 4:
                    #                 # 归一化
                    #                 normalized_data = (rew_minGT - 3) / (4 - 3)
                    #
                    #                 # 映射到目标范围
                    #                 rew_minGT_mapped = normalized_data * (0.75 - 0.5) + 0.5
                    #             elif rew_minGT > 4:
                    #                 # 归一化
                    #                 normalized_data = np.exp(-(1 / (rew_minGT - 4)))
                    #
                    #                 # 映射到目标范围
                    #                 rew_minGT_mapped = normalized_data * (1 - 0.75) + 0.75
                    #
                    #         else:
                    #             rew_minGT_mapped = 0
                    #             social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0
                    #
                    #
                    #         # 下面的代码是用Cal_GT_crash计算的利他策略方式。用了碰撞，但是经常导致利他倾向是0.可能是因为这种方法计算出来的很多rew_aveGT_mapped都是0
                    #         # print('判别器训练use_GT:', use_GT)  # 应该是一个包含9个数字的list，1代表安全，-1代表未来可能碰撞，0代表无交互，-2代表当前碰撞
                    #         # count_very_danger = sum(1 for item in use_GT if item[0] == -2)  # 统计 -2 的个数
                    #         # count_danger = sum(1 for item in use_GT if item[0] == -1)  # 统计 -1 的个数
                    #         # count_safe = sum(1 for item in use_GT if item[0] == 1)  # 统计 1 的个数
                    #         # count_nojiaohu = sum(1 for item in use_GT if item[0] == 0)  # 统计 0 的个数
                    #         #
                    #         # if count_safe == 0 and count_danger == 0 and count_very_danger == 0:
                    #         #     rew_aveGT_mapped = 0  # 无交互对象
                    #         # else:
                    #         #     if count_safe != 0:
                    #         #         # 找到第一个值为1的元素并统计第二个值
                    #         #         selected_items = [item[1] for item in use_GT if item[0] == 1]
                    #         #         # 计算第二个值的平均值
                    #         #         average_min_disvalue = sum(selected_items) / len(selected_items)
                    #         #         lita_cof = count_safe / (
                    #         #                     count_very_danger + count_safe + count_danger)  # 在有冲突的对象中安全交互的比例
                    #         #         rew_aveGT = lita_cof * average_min_disvalue
                    #         #         rew_aveGT_mapped = 1 - np.exp(-(rew_aveGT / 3))  # 除以3的目的是，尽可能的减缓较小距离就有较大奖励的情况
                    #         #         # 消解冲突的奖励归一化 0-1, 1是最考虑所有人的安全，都不撞，并且最小距离比较远
                    #         #
                    #         #         # social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0
                    #         #     else:
                    #         #         rew_aveGT_mapped = -1  # 一点也不合作
                    #
                    #         print('生成器 rew_avespeed:', rew_avespeed,10*rew_avespeed,
                    #               'rew_lane_pianyi:', rew_lane_pianyi, -10 * rew_lane_pianyi,
                    #               'comfort_adj:', comfort_adj, 5*comfort_adj,
                    #               'rew_aveGT_mapped:', rew_minGT_mapped, 10*rew_minGT_mapped)
                    #         rew_social_allbatch.append([10*rew_avespeed, -10*rew_lane_pianyi, 5*comfort_adj, 10*rew_minGT_mapped])
                    #         # print('生成器 rew_social_allbatch:', rew_social_allbatch)
                    #     else:
                    #         # 此时刻是无效数据，历史时刻都已经考虑过了
                    #         rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])
                    #
                    # canshu_social_allbatch_array = np.array(rew_social_allbatch)
                    #
                    # # print('canshu_social_allbatch_array:',np.shape(canshu_social_allbatch_array),canshu_social_allbatch_array)  # (batch,4)
                    # # print('re_obs_lstm[k]:',np.shape(re_obs_lstm[k]),type(re_obs_lstm[k]))  # (batch,4)
                    #
                    # score, pre = self.discriminator[k].get_reward(re_obs_lstm[k],  # re_obs[k]
                    #                                  re_actions[k],  ##################
                    #                                  re_obs_next_lstm[k],  # re_obs_next[k]
                    #                                  re_path_prob[k], canshu_social_allbatch_array,
                    #                                  discrim_score=False)
                    # print('生成器在判别器的输出格式：','score:', np.shape(score), score[0])

                    if k <= 2:
                        print('左转车生成器在判别器的输出格式：','pre:', np.shape(pre), pre[0])
                    else:
                        print('直行车生成器在判别器的输出格式：','pre:', np.shape(pre), pre[0])

                    rewards.append(np.squeeze(score)) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                    # 类似地，将判别器的奖励添加到 report_rewards 列表中。这个奖励通常用于记录或报告。
                    score_report, pre_report = self.discriminator[k].get_reward(re_obs_lstm[k],  # re_obs[k]
                                                     re_actions[k],  ##################
                                                     re_obs_next_lstm[k],  # re_obs_next[k]
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
                # mb_rewards[k].append(rewards[k])
                mb_report_rewards[k].append(report_rewards[k])

            self.states = states


            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k]) # 将当前时间步骤的真实奖励添加到经验数据列表 mb_true_rewards[k] 中，用于记录每个智能体的真实奖励。
                mb_rewards[k].append(rewards[k] + true_rewards[k])  #  + true_rewards[k]   adddddddddddddd +true_rewards[k] 将当前时间步骤的判别器奖励添加到经验数据列表 mb_rewards[k] 中，用于记录每个智能体的判别器奖励。
                print('判别器的reward:',np.shape(rewards), rewards[k],'生成器的reward:',np.shape(true_rewards),true_rewards[k]) # (8, 10)
                # mb_rewards[k].append(true_rewards[k])
                # # print (rewards[k],true_rewards[k])

        # print('run过程中的多线程np.shape(mb_obs_lstm):',np.shape(mb_obs_lstm))  # (8, 5, 10, 21, 46)
        # print('run过程中的多线程np.shape(mb_obs):',np.shape(mb_obs))  # (8, 5, 10, 46)
        # print('run过程中的多线程np.shape(mb_actions):',np.shape(mb_actions))  # (8, 5, 10, 2)
        # print('run过程中的多线程np.shape(mb_values):',np.shape(mb_values))  # (8, 5, 10)
        # print('run过程中的多线程np.shape(mb_dones):',np.shape(mb_dones))   # (8, 5, 10)
        # print('run过程中的多线程np.shape(mb_rewards):', np.shape(mb_rewards))  # (8, 5, 10)

        # 这段代码的主要作用是整理收集到的数据，将其从一批步骤（batch of steps）的形式转换为一批轨迹（batch of rollouts）的形式，并计算每个智能体的回报（returns）。
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k]) # 将每个智能体在时间步骤上的完成状态（done）添加到对应智能体的完成状态列表中。这是为了记录每个智能体在每个时间步骤是否完成。
        # print('2run过程中的多线程np.shape(mb_dones):', np.shape(mb_dones))  # (8, 6, 10)
        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            '''
            np.asarray(mb_obs_lstm[k], dtype=np.float32): 将 mb_obs_lstm[k] 转换为 NumPy 数组，并指定数据类型为 np.float32。这确保了 mb_obs_lstm[k] 中的数据都是浮点数类型。
            .swapaxes(1, 0): 交换数组的轴。在这里，将轴 1 和轴 0 进行了交换。假设 mb_obs_lstm[k] 的原始形状为 (a, b, c)，那么经过这个操作后的数组形状变为 (b, a, c)。这种交换通常用于改变数组中数据的排列方式。
            '''
            mb_obs_lstm[k] = np.asarray(mb_obs_lstm[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape_lstm[k]) # (8,50,21,46)
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])  # (8,50,46)
            # 将观察数据（observations）从列表的形式转换为NumPy数组，然后进行轴交换和形状重塑，以将其变成一批轨迹的形式。mb_obs[k] 现在包含每个时间步骤上的观察数据。
            mb_obs_next_lstm[k] = np.asarray(mb_obs_next_lstm[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape_lstm[k]) # (8,50,21,46)
            mb_obs_next[k] = np.asarray(mb_obs_next[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)  # (8,10,5)
            mb_report_rewards[k] = np.asarray(mb_report_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ac_shape[k])
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)  # (8,10,5)
            mb_masks[k] = mb_dones[k][:, :-1] # 创建一个完成状态的掩码（mask），用于指示哪些时间步骤是终止状态，哪些是非终止状态。掩码是通过移除每个轨迹的最后一个时间步骤的完成状态来创建的。
            # print('掩码之前的mb_donesk:', np.shape(mb_dones[k][:, 1:]), mb_dones[k][:, 1:], np.shape(mb_dones[k]),mb_dones[k])
            mb_dones[k] = mb_dones[k][:, 1:] # 更新完成状态数据，移除每个轨迹的第一个时间步骤的完成状态。这是为了确保完成状态与后续时间步骤对齐。
            # print('掩码的mb_donesk:',np.shape(mb_dones[k]),mb_dones[k])
        # print('之后run过程中的多线程np.shape(mb_obs_lstm):', np.shape(mb_obs_lstm))  # (8, 50, 21, 46)
        # print('之后run过程中的多线程np.shape(mb_obs):', np.shape(mb_obs))  # (8, 50, 46)
        # print('之后run过程中的多线程np.shape(mb_actions):', np.shape(mb_actions))  # (8, 50, 2)
        # print('之后run过程中的多线程np.shape(mb_values):', np.shape(mb_values))  # (8, 10, 5)
        # print('之后run过程中的多线程np.shape(mb_dones):', np.shape(mb_dones))  # (8, 10, 5)
        # print('之后run过程中的多线程np.shape(mb_rewards):', np.shape(mb_rewards))  # (8, 10, 5)
        # print('之后run过程中的多线程np.shape(mb_masks):', np.shape(mb_masks))  # (8, 10, 5)

        # 这部分代码用于计算每个智能体的回报（returns）。
        # 它们的计算方式通常是通过对奖励信号进行折现（discounting）和值函数的估计来实现的，但具体的计算方式可能分布在其他代码段中。这些值是用于训练智能体的重要信息，通常用于计算损失函数和更新策略。
        # 这段代码的主要作用是计算每个智能体的回报（returns），并将收集到的数据重新组织成适合训练强化学习模型的格式。
        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)] # 创建一个包含每个智能体回报的列表，初始值为零，与之前收集的奖励数据的形状相同。
        mb_report_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)] # 创建一个包含每个智能体判别器报告的回报的列表，初始值为零，与之前收集的奖励数据的形状相同。
        # # print('mb_report_returns:',mb_report_returns,np.shape(mb_report_returns))
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)] # 创建一个包含每个智能体真实奖励的回报的列表，初始值为零，与之前收集的奖励数据的形状相同。
        last_values = self.model.value(self.obs_lstm, self.actions) # 获取最后一个时间步骤的值函数估计。这些值函数估计通常用于计算优势函数（advantages）或回报。
        # discount/bootstrap off value fn
        for k in range(self.num_agents): # 遍历每个智能体。
            # 遍历每个时间步骤上收集到的奖励、判别器报告的奖励、真实奖励、完成状态和最后一个时间步骤的值函数估计。
            for n, (rewards, report_rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_report_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                report_rewards = report_rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0: # 检查最后一个时间步骤的完成状态，如果不是终止状态（0表示非终止状态），则执行以下操作：
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1] # 函数将奖励信号进行折现，并将最后一个值（value）从奖励序列中移除。这是一种处理未结束轨迹的方式，以确保正确计算回报。
                    report_rewards = discount_with_dones(report_rewards + [value], dones + [0], self.gamma)[:-1] # 类似于上一步，对判别器报告的奖励进行折现并移除最后一个值。
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1] # 类似于上一步，对真实奖励进行折现并移除最后一个值。
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma) # 对奖励信号进行折现，不需要移除最后一个值，因为轨迹已经终止。
                    report_rewards = discount_with_dones(report_rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards # 将计算得到的回报存储在对应智能体的回报列表中。
                mb_report_returns[k][n] = report_rewards # 将计算得到的判别器报告的回报存储在对应智能体的回报列表中。
                mb_true_returns[k][n] = true_rewards # 将计算得到的真实奖励的回报存储在对应智能体的回报列表中。

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten() # 将每个智能体的回报列表展平，以便用于训练模型。
            mb_report_returns[k] = mb_report_returns[k].flatten() # 将每个智能体的判别器报告的回报列表展平。
            mb_masks[k] = mb_masks[k].flatten() # 将完成状态（masks）列表展平。
            mb_values[k] = mb_values[k].flatten() # 将值函数估计列表展平。
            # mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [mb_actions[k] for k in range(self.num_agents)] # 将智能体的动作列表存储在 mh_actions 中。(8, 50, 2)
        mb_all_obs_lstm = np.concatenate(mb_obs_lstm, axis=1)  # 将所有智能体的观察数据按轴1（列）进行连接，以形成一个包含所有智能体观察数据的大矩阵。  (50, 8 * 21, 46)
        mb_all_nobs_lstm = np.concatenate(mb_obs_next_lstm, axis=1)  # 类似于上一步，将所有智能体的下一步观察数据连接成一个大矩阵。 (50, 8 * 21, 46)
        # print('mb_obs:', np.shape(mb_obs))  # (8, 50, 46)
        # print('mh_actions:', np.shape(mh_actions))  # (8, 50, 2)
        mb_all_obs = np.concatenate(mb_obs, axis=1)  # 将所有智能体的观察数据按轴1（列）进行连接，以形成一个包含所有智能体观察数据的大矩阵。（50,8*46）
        mb_all_nobs = np.concatenate(mb_obs_next, axis=1) # 类似于上一步，将所有智能体的下一步观察数据连接成一个大矩阵。（50,8*46）
        mh_all_actions = np.concatenate(mh_actions, axis=1) # 将所有智能体的动作连接成一个大矩阵。(50, 8 * 2)
        # print('mb_all_obs:',np.shape(mb_all_obs))  # (50, 368)
        # print('mb_values:', np.shape(mb_values))  # (8, 50)
        # print('mb_returns:', np.shape(mb_returns))  # (8, 50)
        # print('mb_masks:', np.shape(mb_masks))  # (8, 50)
        if self.nobs_flag: # 检查是否需要返回包含"nobs"（下一步观察数据）的数据。
            # obs_lstm, states, rewards, masks, actions, values
            # mb_obs_lstm,  mb_states,  mb_returns, mb_masks, mb_actions, mb_values
            # (8, 50, 21, 46), 空, (8, 50), (8, 50), (8, 50, 2), (8, 50)
            return mb_obs_lstm, mb_obs_next_lstm, mb_obs, mb_obs_next, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions, \
                   mb_values, mb_all_obs, mb_all_nobs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns
        else:
            return mb_obs_lstm, mb_obs_next_lstm, mb_obs, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions,\
                   mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns

    # learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
    #       nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
    #       disc_type=disc_type, bc_iters=bc_iters, identical=identical, l2=l2, d_iters=d_iters,
    #       rew_scale=rew_scale)
def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=10, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, l2=0.1, d_iters=1, rew_scale=0.1):
    # print('运行的是nvn的learn')
    tf.reset_default_graph()
    # tf.reset_default_graph() # 用于清除 TensorFlow 默认的计算图，以确保每次运行都从一个干净的状态开始。
    set_global_seeds(seed) # 设置全局的随机种子，以确保实验的可重复性。这里使用了给定的 seed 值来设置随机种子。
    buffer = None  # 初始化 buffer 为 None。这个变量可能用于存储经验回放缓冲区。
    # 获取环境相关信息：
    # nenvs：环境的并行数，即同时运行多少个环境副本。
    # ob_space：环境的观察空间（状态空间）。
    # ac_space：环境的动作空间。
    nenvs = env.num_envs # 10个CPU
    ob_space = env.observation_space # 每一个agent都有46个
    ac_space = env.action_space # 2
    num_agents = (len(ob_space)) # 8
    # print('ob_space:',np.shape(ob_space))  # (8,46)
    # print('ac_space:', np.shape(ac_space))
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical) # 定义一个用于创建 模型的 lambda 函数。并返回一个用于创建强化学习模型的函数。
    # 如果设置了 save_interval（模型保存的间隔）并且存在日志目录 logger.get_dir()，则使用 cloudpickle 将 make_model 函数序列化，并保存为二进制文件 'make_model.pkl'。这样做的目的是为了在训练过程中保存创建模型的函数，以便在需要时重新加载。
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model() # 调用 make_model() 函数，创建一个强化学习模型，包括策略网络和价值网络。

    # 如果 disc_type（判别器类型）是 'decentralized' 或 'decentralized-all'，则创建一个名为 discriminator 的列表。
    # 每个判别器对象对应一个智能体，并用于执行对抗生成逆强化学习（GAIL）中的判别器任务。
    # 判别器的配置参数包括模型的会话、观察空间、动作空间等。
    # 如果 disc_type 不是 'decentralized' 或 'decentralized-all'，则断言失败，以确保判别器的类型正确。
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

    # 以下这段代码主要是对训练过程进行了初始化，包括了奖励正则化的设置、TensorFlow 变量的初始化、Runner 的创建以及协调器的初始化。
    # 它为接下来的训练做好了准备工作。

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
        )  # 计算奖励正则化项 reward_reg_loss。这个正则化项基于多个判别器的奖励信号，目的是对奖励进行正则化。
        # tf.square(discriminator[0].reward + discriminator[3].reward) 等表示了不同智能体的奖励信号之间的平方差。
        # tf.maximum(0.0, 1 - discriminator[0].reward) 等表示了对奖励信号的裁剪，确保奖励不小于1。这些项被组合起来，并通过 rew_scale 进行加权，得到最终的奖励正则化损失。

        reward_reg_lr = tf.placeholder(tf.float32, ()) # 创建了一个占位符 reward_reg_lr，用于设置奖励正则化的学习率。
        reward_reg_optim = tf.train.AdamOptimizer(learning_rate=reward_reg_lr) # 创建了一个 Adam 优化器 reward_reg_optim，用于最小化奖励正则化损失。
        reward_reg_train_op = reward_reg_optim.minimize(reward_reg_loss) # 创建了奖励正则化的训练操作 reward_reg_train_op，该操作通过优化器最小化奖励正则化损失。

    tf.global_variables_initializer().run(session=model.sess) # 执行全局变量初始化操作 tf.global_variables_initializer().run(session=model.sess)，以初始化所有 TensorFlow 变量，包括模型的参数。
    # log_dir = logger.get_dir()
    # writer = tf.summary.FileWriter(log_dir) if log_dir else None
    # # Create a unique TensorFlow summary writer
    # log_dir = osp.abspath(osp.join(logger.get_dir(), "tb"))
    # # log_dir = osp.join(logger.get_dir(), f"tb")
    # if not osp.exists(log_dir):
    #     os.makedirs(log_dir)
    # writer = tf.summary.FileWriter(log_dir)

    # print(f"TensorBoard logs will be saved to: {log_dir}")

    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type,
                    nobs_flag=True) # 创建一个 Runner 对象 runner，用于运行环境和模型的交互。Runner 用于生成训练样本，计算奖励等。
    nbatch = nenvs * nsteps # 计算 nbatch，即一个训练批次的数据量，等于环境数量 nenvs 乘以每个环境的时间步数 nsteps。
    
    tstart = time.time() # 记录当前时间 tstart，以便在训练结束后计算总训练时间。
    logger.record_tabular("time", tstart)
    
    coord = tf.train.Coordinator() # 创建 TensorFlow 协调器 coord，用于协调多个 TensorFlow 运行时的线程。协调器在多线程训练中用于管理线程的启动和停止。
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    for _ in range(bc_iters):
        e_obs, e_actions, e_nobs, _, _ = expert.get_next_batch(nenvs * nsteps)
        e_a = e_actions #[np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
        # lld_loss = model.clone(e_obs, e_a)
        # # print(lld_loss)

    update_policy_until = 0  # 10

    # 这段代码是一个主要的训练循环，用于训练强化学习模型和判别器
    # 累积数据
    rewards_data = []
    total_loss_data = []
    train_loss_out_data = []
    for update in range(1, 2000 + 1): # 开始一个训练更新周期，总共进行1000次更新。
        print('第几次大迭代:',update)
        # obs_lstm, states, rewards, masks, actions, values
        obs_lstm, obs_next_lstm, obs, obs_next, states, rewards, report_rewards, masks, actions, values, all_obs, all_nobs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns = runner.run(update)
        # 通过 runner 执行模拟游戏环境，并获取一批数据，包括当前观察、下一步观察、状态、奖励、判别器报告的奖励、完成状态、动作、值函数估计、所有观察数据、所有下一步观察数据、所有动作、所有动作（展平的形式）、智能体奖励和真实奖励。
        
        trun = time.time() # 记录当前时间，用于计算各个阶段的时间消耗。
        logger.record_tabular("time_1", trun-tstart) # 记录第一个时间段的时间消耗。

        total_loss = np.zeros((num_agents, d_iters)) # 创建一个数组，用于存储每个智能体的判别器总损失。

        idx = 0 # 初始化一个索引变量，用于对数据进行洗牌。
        # print('这里的all_obs为：',np.shape(all_obs))
        idxs = np.arange(len(all_obs)) # 创建一个包含数据索引的数组  all_obs的shape为（50,8*46） np.arange(50) 将生成一个包含 0 到 49 的整数的一维数组。结果为:array([0, 1, 2, 3, 4, 5, 6, 7])
        # print('这里的inxs为：',idxs) # 这里的inxs为： [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
        random.shuffle(idxs) # 随机打乱数据索引，以确保数据的随机性。打乱的是nbatch*nstep=50 所有并行环境得到的nstep行的排序
        # print('打乱之后这里的inxs为：', idxs)
        all_obs = all_obs[idxs] # 据打乱的索引重新排列所有观察数据。
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)] # 根据打乱的索引重新排列所有动作数据。 mh_actions的shape为（）
        mh_obs_lstm = [obs_lstm[k][idxs] for k in range(num_agents)]  # 根据打乱的索引重新排列智能体的观察数据。 obs_lstm的shape为（8,50,21,46）
        mh_obs_next_lstm = [obs_next_lstm[k][idxs] for k in range(num_agents)]  # 根据打乱的索引重新排列智能体的下一步观察数据。 obs_next_lstm的shape为（8,50,12,46）
        mh_obs = [obs[k][idxs] for k in range(num_agents)] # 根据打乱的索引重新排列智能体的观察数据。
        mh_obs_next = [obs_next[k][idxs] for k in range(num_agents)] # 根据打乱的索引重新排列智能体的下一步观察数据。
        mh_values = [values[k][idxs] for k in range(num_agents)] # 根据打乱的索引重新排列智能体的值函数估计。
        # print('打乱之后这里的mh_obs_lstm为：', np.shape(mh_obs_lstm)) #  (8, 50, 21, 46)
        # print('打乱之后这里的mh_obs_next_lstm为：', np.shape(mh_obs_next_lstm)) #  (8, 50, 21, 46)
        if buffer: # 检查缓冲区是否已创建。
            buffer.update(mh_obs_lstm, mh_actions, mh_obs_next_lstm, all_obs, mh_values) # 如果缓冲区已创建，则使用新的数据更新缓冲区，包括智能体的观察数据、动作、下一步观察数据、所有观察数据和值函数估计。
        else:
            buffer = Dset(mh_obs_lstm, mh_actions, mh_obs_next_lstm, all_obs, mh_values, randomize=True, num_agents=num_agents,
                          nobs_flag=True) # 如果缓冲区未创建，则创建一个新的缓冲区。

        d_minibatch = nenvs * nsteps # 计算每次判别器训练所使用的数据点数。

        d_iters_new = 1
        for d_iter in range(d_iters_new): # 开始判别器的训练迭代。 判别器迭代训练10次
            e_obs, e_actions, e_nobs, e_all_obs, _ = expert.get_next_batch(d_minibatch) # 从专家策略获取下一批数据，包括观察数据、动作、下一步观察数据和所有观察数据。 这是专家数据
            g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch) # 从缓冲区获取下一批数据，包括观察数据、动作、下一步观察数据和所有观察数据。 这是环境生成的数据

            e_a = e_actions# 将专家策略的动作数据存储在 e_a 变量中。 [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
            g_a = g_actions# 将缓冲区的动作数据存储在 g_a 变量中。 [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

            # g_obs: (8, 50, 21, 46) g_nobs (8, 50, 21, 46) g_a: (8, 50, 2) e_obs: (8, 50, 21, 46) e_nobs (8, 50, 21, 46) e_a: (8, 50, 2)
            '''
            一般来说，在生成对抗网络（GANs）或策略优化算法中，生成模型被训练来生成与专家策略相似的数据。
            在这里，g_log_prob 可以被认为是生成模型给定观察数据和动作数据的条件下，生成该动作的对数概率。
            这个值在算法的训练过程中可能会被用来计算损失函数、进行策略优化等。
            '''
            g_log_prob = model.get_log_action_prob(g_obs, g_a) # 计算环境生成的动作在当前模型下的对数概率。
            e_log_prob = model.get_log_action_prob(e_obs, e_a) # 计算专家策略的动作在当前模型下的对数概率。

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
            def Cal_GT_crash(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                             Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                             Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

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
                        GT_value = -2  # 非常不安全
                        dis_min = 0
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
                                GT_value = 1  # 安全
                                dis_min = np.sqrt((Agent_x - Jiaohu_x) ** 2 + (Agent_y - Jiaohu_y) ** 2)
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
                                    # 2.1 AGENT 会把交互对象看做 有冲突的对象;交互对象也会把agent看做 有冲突的对象
                                    # 判断当agent到交点的时候，neig在哪，如果撞了，GT_value=-1,否则None
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录快车走到交点的路程中，两车之间轨迹点的距离，步长0.5s（不考虑车宽，因为是否碰撞上已经考虑了）
                                    if t_agent < t_neig:
                                        # agent先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_agent > 23:
                                            t_agent = 23
                                        else:
                                            t_agent = t_agent
                                        # try:
                                        time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_agent:
                                            time = np.arange(0, t_agent + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_agent, 0.5)
                                            time = np.append(time, t_agent)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离
                                    else:
                                        # neig先到冲突点
                                        crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞
                                        if t_neig > 23:
                                            t_neig = 23
                                        else:
                                            t_neig = t_neig
                                        # try:
                                        time_n_0 = int(t_neig / 0.5)  # time_n_0不是无限大
                                        if time_n_0 * 0.5 == t_neig:
                                            time = np.arange(0, t_neig + 0.01, 0.5)
                                        else:
                                            time = np.arange(0, t_neig, 0.5)
                                            time = np.append(time, t_neig)  # 添加 t_agent 到
                                        # except:
                                        #     time_n_0 = 36
                                        #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                        for time_futurenow in time:
                                            # 走多少秒
                                            agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                            Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                                np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                            neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                            Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                            Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                                np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                            # 统计此时距离
                                            dis_time_futurenow = np.sqrt(
                                                (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                            dis_time.append(dis_time_futurenow)

                                            # 判断当前时刻辆车是否会相撞
                                            # 绘制两个矩形
                                            # 计算矩形的四个顶点坐标
                                            vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t,
                                                                                                    Agent_y_t,
                                                                                                    veh_length,
                                                                                                    veh_width,
                                                                                                    Agent_angle_last)
                                            vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t,
                                                                                                   Jiaohu_y_t,
                                                                                                   veh_length,
                                                                                                   veh_width,
                                                                                                   Jiaohu_angle_last)
                                            # 判断两个矩阵是否有交集
                                            intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                     vertices_neig_futurenow)
                                            if intersect_futurenow == True:
                                                # 说明agent和交互对象在未来交点相撞了
                                                GT_value = -1  # 不安全
                                                dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                                crash_futurenow = True
                                                break  # 退出这个循环，已经检测到未来会相撞
                                            else:
                                                GT_value = 1  # 安全
                                                crash_futurenow = False
                                        if crash_futurenow == False:
                                            dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                elif dot_product_agent >= 0 and dot_product_neig < 0:
                                    # 2.2 agent把neig看做冲突对象，但是neig不把agent看做冲突对象，仍然需要判断在agent到冲突点的路程中，是否会发生碰撞，以及距离的大小
                                    agent_dis = np.sqrt((Agent_x - jiaodianx) ** 2 + (Agent_y - jiaodiany) ** 2)
                                    t_agent = agent_dis / agent_v

                                    neig_dis = np.sqrt((Jiaohu_x - jiaodianx) ** 2 + (Jiaohu_y - jiaodiany) ** 2)
                                    t_neig = neig_dis / neig_v

                                    dis_time = []  # 记录agent车走到交点的路程中，两车之间轨迹点的距离（不考虑车宽，因为是否碰撞上已经考虑了）
                                    # agent会到达冲突点，neig不会到达冲突点
                                    crash_futurenow = False  # 有一时刻会碰撞的标签，False表示不会碰撞，True表示会碰撞

                                    if t_agent > 23:
                                        t_agent = 23
                                    else:
                                        t_agent = t_agent
                                    # try:
                                    time_n_0 = int(t_agent / 0.5)  # time_n_0不是无限大
                                    if time_n_0 * 0.5 == t_agent:
                                        time = np.arange(0, t_agent + 0.01, 0.5)
                                    else:
                                        time = np.arange(0, t_agent, 0.5)
                                        time = np.append(time, t_agent)  # 添加 t_agent 到
                                    # except:
                                    #     time_n_0 = 36
                                    #     time = np.arange(0, time_n_0 * 0.5 + 0.01, 0.5)

                                    for time_futurenow in time:
                                        # 走多少秒
                                        agent_dis_time_futurenow = time_futurenow * agent_v  # neig在t_agent内走的路程
                                        Agent_x_t = Agent_x + agent_dis_time_futurenow * np.cos(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Agent_y_t = Agent_y + agent_dis_time_futurenow * np.sin(
                                            np.deg2rad(Agent_angle_last))  # 计算下一个时刻的y

                                        neig_dis_time_now = time_futurenow * neig_v  # neig在t_agent内走的路程
                                        Jiaohu_x_t = Jiaohu_x + neig_dis_time_now * np.cos(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的x x0 + v0t+0.5at^2
                                        Jiaohu_y_t = Jiaohu_y + neig_dis_time_now * np.sin(
                                            np.deg2rad(Jiaohu_angle_last))  # 计算下一个时刻的y

                                        # 统计此时距离
                                        dis_time_futurenow = np.sqrt(
                                            (Agent_x_t - Jiaohu_x_t) ** 2 + (Agent_y_t - Jiaohu_y_t) ** 2)
                                        dis_time.append(dis_time_futurenow)

                                        # 判断当前时刻辆车是否会相撞
                                        # 绘制两个矩形
                                        # 计算矩形的四个顶点坐标
                                        vertices_agent_futurenow = calculate_rectangle_vertices(Agent_x_t, Agent_y_t,
                                                                                                veh_length, veh_width,
                                                                                                Agent_angle_last)
                                        vertices_neig_futurenow = calculate_rectangle_vertices(Jiaohu_x_t, Jiaohu_y_t,
                                                                                               veh_length, veh_width,
                                                                                               Jiaohu_angle_last)
                                        # 判断两个矩阵是否有交集
                                        intersect_futurenow = check_intersection(vertices_agent_futurenow,
                                                                                 vertices_neig_futurenow)
                                        if intersect_futurenow == True:
                                            # 说明agent和交互对象在未来交点相撞了
                                            GT_value = -1  # 不安全
                                            dis_min = 0  # 无意义的距离统计，只是为了形式上的统一
                                            crash_futurenow = True
                                            break  # 退出这个循环，已经检测到未来会相撞
                                        else:
                                            GT_value = 1  # 安全
                                            crash_futurenow = False
                                    if crash_futurenow == False:
                                        dis_min = min(dis_time)  # 有意义的距离统计，未来都没碰撞，就看最小距离

                                else:
                                    # 2.3 AGENT 不会把交互对象看做 有冲突的对象，但是交互对象可能会把agent看做有冲突的对象
                                    # 所以不看未来，只看现在。因为如果neig把agent看做交互对象的话，agent很大可能在历史时刻也把其看作为冲突对象过（此刻两车距离很近的情况下，很远就无所谓了）
                                    GT_value = 0
                                    dis_min = 100000



                        else:
                            # neig不是前车 2.4.2
                            GT_value = 0  # 不看做交互，因为当前时刻已经没有碰撞了
                            dis_min = 100000


                else:
                    GT_value = 0  # 不交互
                    dis_min = 100000

                return GT_value, dis_min
            def Cal_GT_gt(Agent_x, Agent_y, Agent_vx, Agent_vy, Agent_angle_last, Agent_direction,
                          Jiaohu_x, Jiaohu_y, Jiaohu_vx, Jiaohu_vy, Jiaohu_angle_last,
                          Jiaohu_direction):  # time_trj,neig_left均为1行的dataframe

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

            if disc_type == 'decentralized': # 检查判别器类型是否为"decentralized"。
                for k in range(num_agents):
                    if k <= 2:
                        direction_agent = 'left'
                    else:
                        direction_agent = 'straight'

                    obs_k = np.concatenate([g_obs[k], e_obs[k]],axis=0)
                    batch_num_ = np.shape(obs_k)[0]
                    print('判别器训练得到的obs_k:',np.shape(obs_k))

                    # 计算利己和利他所需要的参数
                    # 计算利己奖励和利他奖励，然后利用网络学习参数φ，cos(φ)=利己倾向，sin(φ)利他倾向
                    rew_input_fuyuan = obs_k
                    rew_social_allbatch = []  # 存放这一个agent 所有batch的参数
                    # 利己性参数-速度, 针对每一个batch来计算
                    for i_batch in range(batch_num_):
                        # 改成当前时刻应该更好，因为是当前时刻的奖励，过去的已经无法改变了。过去的状态可以看做是影响社交倾向的因素
                        # 如果是考虑历史数据的话，对于一些当前时刻无效，但历史时刻有效的数据来说，奖励就没有实际含义了
                        # 其实也可以有实际含义。再想想。还是不考虑了
                        if rew_input_fuyuan[i_batch][20][0] != 0:
                            use_GT = []  # 存放这个ibatch的主要交互对象的GT
                            # speed = np.sqrt(rew_input_fuyuan[i_batch][20][2] ** 2 + rew_input_fuyuan[i_batch][20][3] ** 2)
                            pianyi_distance = rew_input_fuyuan[i_batch][20][-2]
                            # 计算和主要交互对象的GT
                            # 提取代理的状态和终点坐标
                            agent_x = rew_input_fuyuan[i_batch][20][0] * 38 - 4
                            agent_y = rew_input_fuyuan[i_batch][20][1] * 23 + 14
                            agent_vx = rew_input_fuyuan[i_batch][20][2] * 21 - 14
                            agent_vy = rew_input_fuyuan[i_batch][20][3] * 12 - 2
                            agent_angle_last = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # 上一个点的前进方向

                            agent_v = np.sqrt((agent_vx) ** 2 + (agent_vy) ** 2)

                            # 避免碰撞
                            # 计算agent和周围最密切的三个交互对象的GT
                            # obs包括 front_delta_x, front_delta_y, front_delta_vx, front_delta_vy, left_delta_x, left_delta_y, left_delta_vx, left_delta_vy, right_delta_x, right_delta_y, right_delta_vx, right_delta_vy
                            # 安全奖励
                            rew_GT = 0
                            # use_GT = []  # 存储和主要交互对象的GT  会有7+2个值，包括除主agent之外所有的agent和交互的landmark，即使没有这个对象，也会赋值为0
                            # 计算和除主车之外所有agent以及交互的landmark车辆的GT
                            # 把所有的agent都考虑
                            for agent_k_ in range(num_agents):
                                if agent_k_ != k:
                                    # 这个agent不是我们正在计算的k
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
                                        jiaohu_agent_angle_last = rew_input_fuyuan_agent_k_[i_batch][20][6] * 191 - 1  # 上一个点的前进方向

                                        jiaohu_agent_GT_value = Cal_GT_gt(agent_x, agent_y, agent_vx, agent_vy,
                                                                                agent_angle_last, direction_agent,
                                                                                jiaohu_agent_x, jiaohu_agent_y,
                                                                                jiaohu_agent_vx, jiaohu_agent_vy,
                                                                                jiaohu_agent_angle_last,
                                                                                direction_jiaohu)
                                        # if same_jiaohu_agent_GT_value is not None:
                                        use_GT.append(jiaohu_agent_GT_value)
                                    else:  # 没有这个agent
                                        jiaohu_agent_x = -4
                                        jiaohu_agent_y = 14
                                        jiaohu_agentk_vx = -14
                                        jiaohu_agent_vy = -2
                                        jiaohu_agent_angle_last = -1
                                        jiaohu_agent_GT_value = None
                                        # dis_min = 100000
                                        use_GT.append(jiaohu_agent_GT_value)

                            # 左侧视野的landmark
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

                            # 右侧视野的landmark
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
                                # dis_min = 100000
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
                                # dis_min = 10000
                                use_GT.append(right_jiaohu_landmark_GT_value)

                            # 计算角度波动
                            # 计算一些rew
                            # # 计算上个时刻，上上个时刻，上上上时刻 三个heading的标准差，真实值，计算heading标准差过大带来的惩罚
                            # if rew_input_fuyuan[i_batch][18][0] == 0:
                            #     # 没有上上上时刻的角度，所以判断上上个时刻的角度（一开始无论左转还是直行几乎都是直行的角度）
                            #     if rew_input_fuyuan[i_batch][19][0] == 0:
                            #         # 也没有上上时刻的角度
                            #         heading_angle_last3_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                            #         heading_angle_last2_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                            #     else:
                            #         # 有上上时刻的角度，上上上时刻的角度也用上上时刻的角度来代替
                            #         heading_angle_last3_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                            #         heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                            # else:
                            #     # 有上上上时刻的角度，所以也有上上个时刻的角度
                            #     heading_angle_last3_real = rew_input_fuyuan[i_batch][18][6] * 191 - 1  # [0,1]
                            #     heading_angle_last2_real = rew_input_fuyuan[i_batch][19][6] * 191 - 1  # [0,1]
                            #
                            # heading_angle_last1_real = rew_input_fuyuan[i_batch][20][6] * 191 - 1  # [0,1]
                            #
                            # # 计算平均值
                            # mean_value = np.mean(
                            #     [heading_angle_last1_real, heading_angle_last2_real, heading_angle_last3_real])
                            # # 计算每个数据与平均值的差的平方
                            # squared_differences = [(x - mean_value) ** 2 for x in
                            #                        [heading_angle_last1_real, heading_angle_last2_real,
                            #                         heading_angle_last3_real]]
                            # # 计算平方差的平均值
                            # mean_squared_difference = np.mean(squared_differences)
                            # # 计算标准差
                            # std_dev = np.sqrt(mean_squared_difference)
                            # if std_dev > 3:
                            #     rew_heading_std_bodong = np.exp(-(std_dev - 3))  # 越大于3，奖励越小
                            # else:
                            #     rew_heading_std_bodong = 1

                            # 计算steering angle正负来回变化带来的惩罚
                            # 上一时刻的转角，若超过均值的1个标准差，则给予惩罚之类的
                            penalty = 1  # 惩罚系数
                            delta_angle_last1 = rew_input_fuyuan[i_batch][20][56]
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
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(
                                                dis_left_delta_angle_last1 / left_delta_angle_last1_realstd) * penalty
                                        # 越靠近left_delta_angle_last1_realstd，惩罚越接近-1
                            else:
                                right_delta_angle_last1_real = ((2.4 * (delta_angle_last1 + 1)) / 2) - 1.2
                                right_delta_angle_last1_realmean = 0.001
                                right_delta_angle_last1_realstd = 0.076
                                if right_delta_angle_last1_realmean - right_delta_angle_last1_realstd <= right_delta_angle_last1_real and right_delta_angle_last1_real <= right_delta_angle_last1_realmean + right_delta_angle_last1_realstd:
                                    comfort_adj = 0  # 不做惩罚

                                else:
                                    dis_right_delta_angle_last1 = abs(right_delta_angle_last1_real - right_delta_angle_last1_realmean) - right_delta_angle_last1_realstd
                                    if dis_right_delta_angle_last1 > right_delta_angle_last1_realstd:
                                        comfort_adj = -1 * penalty
                                    else:
                                        comfort_adj = -(dis_right_delta_angle_last1 / right_delta_angle_last1_realstd) * penalty
                                        # 越靠近right_delta_angle_last1_realstd，惩罚越接近-1

                            # 利己-效率
                            rew_avespeed = agent_v / 6.8  # 除以85分位速度
                            # 利己-车道偏移
                            rew_lane_pianyi = pianyi_distance

                            # 利他-GT
                            use_GT_list_0 = [x for x in use_GT if x is not None]  # 不为None的list，例如[0.32294667405015254]
                            use_GT_list = [x for x in use_GT_list_0 if not np.isnan(x)]
                            rew_minGT_mapped = 0
                            if len(use_GT_list) != 0:
                                # rew_aveGT = sum(use_GT_list) / len(use_GT_list)
                                rew_minGT = sum(use_GT_list) / len(use_GT_list)  # min(use_GT_list)
                                if rew_minGT <= 1.5:
                                    # 归一化
                                    normalized_data = (rew_minGT - 0) / (1.5 - 0)
                                    # 映射到目标范围
                                    rew_minGT_mapped = normalized_data * (0.25 + 0.5) - 0.5
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
                                rew_minGT_mapped = 0
                                social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0


                            # 下面的代码是用Cal_GT_crash计算的利他策略方式。用了碰撞，但是经常导致利他倾向是0.可能是因为这种方法计算出来的很多rew_aveGT_mapped都是0
                            # print('判别器训练use_GT:', use_GT)  # 应该是一个包含9个数字的list，1代表安全，-1代表未来可能碰撞，0代表无交互，-2代表当前碰撞
                            # count_very_danger = sum(1 for item in use_GT if item[0] == -2)  # 统计 -2 的个数
                            # count_danger = sum(1 for item in use_GT if item[0] == -1)  # 统计 -1 的个数
                            # count_safe = sum(1 for item in use_GT if item[0] == 1)  # 统计 1 的个数
                            # count_nojiaohu = sum(1 for item in use_GT if item[0] == 0)  # 统计 0 的个数
                            #
                            # if count_safe == 0 and count_danger == 0 and count_very_danger == 0:
                            #     rew_aveGT_mapped = 0  # 无交互对象
                            # else:
                            #     if count_safe != 0:
                            #         # 找到第一个值为1的元素并统计第二个值
                            #         selected_items = [item[1] for item in use_GT if item[0] == 1]
                            #         # 计算第二个值的平均值
                            #         average_min_disvalue = sum(selected_items) / len(selected_items)
                            #         lita_cof = count_safe / (
                            #                     count_very_danger + count_safe + count_danger)  # 在有冲突的对象中安全交互的比例
                            #         rew_aveGT = lita_cof * average_min_disvalue
                            #         rew_aveGT_mapped = 1 - np.exp(-(rew_aveGT / 3))  # 除以3的目的是，尽可能的减缓较小距离就有较大奖励的情况
                            #         # 消解冲突的奖励归一化 0-1, 1是最考虑所有人的安全，都不撞，并且最小距离比较远
                            #
                            #         # social_pre_ibatch = 0  # 在这种情况下就是无交互对象，我们认为是不需要利他的。φ为0
                            #     else:
                            #         rew_aveGT_mapped = -1  # 一点也不合作

                            # print('判别器 rew_avespeed:', rew_avespeed,10 * rew_avespeed,
                            #       'rew_lane_pianyi:', rew_lane_pianyi, -10 * rew_lane_pianyi,
                            #       'comfort_adj:',comfort_adj, 5 * comfort_adj,
                            #       'rew_aveGT_mapped:', rew_minGT_mapped, 10 * rew_minGT_mapped)
                            rew_social_allbatch.append([10 * rew_avespeed, -10 * rew_lane_pianyi, 5 * comfort_adj, 10 * rew_minGT_mapped])
                            # rew_social_allbatch.append([rew_avespeed, -5 * rew_lane_pianyi, comfort_adj, rew_minGT_mapped])
                            # print('生成器 rew_social_allbatch:', rew_social_allbatch)
                        else:
                            # 此时刻是无效数据，历史时刻都已经考虑过了
                            rew_social_allbatch.append([0.1, 0.1, 0.1, 0.1])


                    canshu_social_allbatch_array = np.array(rew_social_allbatch)

                    # print('total_losscanshu_social_allbatch_array:', np.shape(canshu_social_allbatch_array),canshu_social_allbatch_array)  # (batch,4)

                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs[k],
                        g_actions[k],
                        g_nobs[k],
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs[k],
                        e_actions[k],
                        e_nobs[k],
                        e_log_prob[k].reshape([-1, 1]), canshu_social_allbatch_array) # 循环遍历每个智能体，计算每个智能体的判别器损失并存储在 total_loss 中。
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

        tdistr = time.time() # 记录当前时间，用于计算各个阶段的时间消耗。
        logger.record_tabular("time_2", tdistr-trun) # 记录第二个时间段的时间消耗。 训练判别器的时间

        if update > update_policy_until:  # 10 检查是否达到更新策略网络的时机（默认在第10次更新后开始）。
            policy_loss, value_loss, policy_entropy, train_loss_out = model.train(obs_lstm, states, rewards, masks, actions, values) # 如果是的话，执行策略网络的训练，计算策略损失、值函数损失和策略熵。
        model.old_obs = obs # 将当前观察数据存储在 model 中的 old_obs 变量中，以备后续使用。
        nseconds = time.time() - tstart # 计算从训练开始到现在的总时间。
        
        tpoltr = time.time() # 记录当前时间，用于计算各个阶段的时间消耗。
        logger.record_tabular("time_3", tpoltr-tdistr) # 记录第三个时间段的时间消耗。更新策略网络和值函数网络的时间
        
        fps = int((update * nbatch) / nseconds) # 计算训练速度，即每秒钟处理的样本数。
        if update % log_interval == 0 or update == 1: # 检查是否需要记录日志，条件是每隔一定的更新周期或在第一个更新周期时。
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)] # 计算每个智能体的解释方差，并将其记录在日志中。
            # # 记录更新周期数、总时间步数、训练速度以及各项指标，如策略熵、策略损失、值函数损失、Pearson相关系数、Spearman秩相关系数和奖励。
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                if update > update_policy_until:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    logger.record_tabular("train_loss_out %d" % k, float(train_loss_out[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
                    except:
                        pass

            total_loss_m = np.mean(total_loss, axis = 1)  # 计算每个智能体的平均判别器损失。
            total_reward = (np.mean(rewards, axis = 1))  # 计算每个智能体的平均奖励。

            for k in range(num_agents):
                logger.record_tabular("total_loss %d" % k, total_loss_m[k])
            logger.record_tabular("total_loss", np.sum(total_loss_m))
            logger.record_tabular('rewards', np.sum(total_reward))
            logger.record_tabular('policy_loss', np.sum(policy_loss))
            logger.record_tabular('value_loss', np.sum(value_loss))
            if update > update_policy_until:
                logger.record_tabular('train_loss_out', np.sum(train_loss_out))

            # # 计算梯度范数
            # vanishing_threshold = 1e-6
            # exploding_threshold = 1e2
            #
            # with open("gradient_log.txt", "a") as f:  # 打开或创建一个日志文件
            #     f.write(f"Update: {update}\n")
            #
            #     for k in range(model.num_agents):
            #         # 计算每个智能体的梯度范数
            #         norms = [np.linalg.norm(g) if g is not None else 0.0 for g in grads_out[k]]
            #
            #         # 记录梯度范数
            #         f.write(f"Agent {k} Gradient Norms: {norms}\n")
            #
            #         # 检查是否发生梯度消失或爆炸
            #         if any(norm < vanishing_threshold for norm in norms):
            #             f.write(f"Warning: Agent {k} Gradient Vanishing Detected!\n")
            #         if any(norm > exploding_threshold for norm in norms):
            #             f.write(f"Warning: Agent {k} Gradient Exploding Detected!\n")
            #
            #     f.write("\n")  # 每次更新之间留一个空行
            #
            # # 在 logger 中记录梯度范数的统计信息
            # for k in range(model.num_agents):
            #     norms = [np.linalg.norm(g) if g is not None else 0.0 for g in grads_out[k]]
            #     avg_grad_norm = np.mean(norms)
            #     max_grad_norm = np.max(norms)
            #     min_grad_norm = np.min(norms)
            #
            #     logger.record_tabular(f"avg_grad_norm_{k}", avg_grad_norm)
            #     logger.record_tabular(f"max_grad_norm_{k}", max_grad_norm)
            #     logger.record_tabular(f"min_grad_norm_{k}", min_grad_norm)

            logger.dump_tabular()

            # 累积数据
            rewards_data.append(np.sum(total_reward))
            total_loss_data.append(np.sum(total_loss_m))
            if update > update_policy_until:
                train_loss_out_data.append(np.sum(train_loss_out))

            # Log to TensorFlow summary
            # if writer:
            #     for k in range(num_agents):
            #         summary = tf.Summary(value=[
            #             tf.Summary.Value(tag="explained_variance/%d" % k, simple_value=float(ev[k])),
            #             tf.Summary.Value(tag="policy_entropy/%d" % k, simple_value=float(policy_entropy[k])),
            #             tf.Summary.Value(tag="policy_loss/%d" % k, simple_value=float(policy_loss[k])),
            #             tf.Summary.Value(tag="value_loss/%d" % k, simple_value=float(value_loss[k]))
            #         ])
            #         writer.add_summary(summary, global_step=update)
            #
            #     writer.flush()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir(): # 检查是否需要保存模型和判别器的检查点，条件是每隔一定的更新周期或在第一个更新周期时，以及是否有可用的日志目录。
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            # print('Saving to', savepath)
            model.save(savepath)  # 保存策略模型的检查点。
            if disc_type == 'decentralized' or disc_type == 'decentralized-all': # 如果判别器类型是"decentralized"或"decentralized-all"，则保存每个判别器的检查点。
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update)) #
                    discriminator[k].save(savepath)
            else:
                assert False


    df = pd.DataFrame({
        'update': range(1, 2001),
        'rewards_data': rewards_data,
        'total_loss_data': total_loss_data,
        'train_loss_out_data': train_loss_out_data
    })

    # 设置 'update' 作为行号
    df.set_index('update', inplace=True)
    df.to_csv(r'rewards_loss_data.csv')
    coord.request_stop() # 请求停止多线程协调器。
    # coord.join(enqueue_threads)
    env.close() # 关闭游戏环境。
