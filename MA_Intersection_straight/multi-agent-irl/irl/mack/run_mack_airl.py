#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore")


import logging
import os
import itertools
import click
import gym


import make_env
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.airl_con_ac import learn
from sandbox.mack.policies import MaskedCategoricalPolicy, GaussianPolicy, LSTMGaussianPolicy, MASKATTGaussianPolicy
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, l2=0.1, d_iters=1,
          rew_scale=0.1):
    # 这个函数的作用是根据给定的 rank 创建并配置一个环境，并返回该环境的实例。在分布式设置中，通常会为不同的进程创建不同的环境实例，以确保随机性和隔离性。
    def create_env(rank):
        def _thunk(): # 内部定义了一个名为 _thunk 的函数，这个函数将在外部函数 create_env 中返回。它用于实际创建和配置环境
            scenario_test_name = 0
            training_label = True
            env = make_env.make_env(env_id, scenario_test_name, training_label)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True) # 将创建的环境包装在 bench.Monitor 中，用于记录环境的性能数据。这将允许你收集关于环境的各种信息，例如奖励、步数等，并将其记录到文件中以进行后续分析。
            gym.logger.setLevel(logging.WARN) # 将 Gym 库的日志级别设置为 WARN，以减少输出的日志信息，通常用于减少不必要的输出。
            return env # 返回配置好的环境对象。
        return _thunk

    # identical=make_env.get_identical(env_id)
    # identical=[False] + [True for _ in range(6)] + [False] + [True for _ in range(2)] + [False] + [True for _ in range(3)] + [False] + [True for _ in range(3)]
    identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]
    # 根据policy来设置identical，[False] + [True for _ in range(4)] 是一个policy里的5个agent，第一个是false，后四个是True，意思是和前面这个agent的policy一致； [False] + [True for _ in range(33)] 是一个policy里的34个agent，第一个是false，意思是和前面的agent的policy不一致，后33个是True，意思是和前面这个agent的policy一致
    # identical = [False] + [True for _ in range(146)] + [False] + [True for _ in range(819)]
    # identical=[False] + [True for _ in range(8)] + [False] + [True for _ in range(6)]
    print('identical:',identical)

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    print('num_cpu的个数为：',num_cpu)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)  # 环境
    print('并行环境已完成')

    policy_fn = MASKATTGaussianPolicy  # GaussianPolicy  # 策略
    print('策略已完成')
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation, nobs_flag=True)
    print('专家数据已完成')
    # learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
    #       nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
    #       disc_type=disc_type, bc_iters=bc_iters, identical=identical, l2=l2, d_iters=d_iters,
    #       rew_scale=rew_scale)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.01, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=identical, l2=l2, d_iters=d_iters,
          rew_scale=rew_scale)

    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='multi-agent-trj/logger')
@click.option('--env', type=click.STRING, default='trj_intersection_4')
@click.option('--expert_path', type=click.STRING,
              default=r'D:\Study\同济大学\博三\面向自动驾驶测试的仿真\sinD_nvn_xuguan'
                      r'\ATT-social-iniobs_rlyouhua\MA_Intersection_straight'
                      r'\AV_test\DATA\sinD_nvnxuguan_9jiaohu_social_dayu1_v2.pkl')
@click.option('--seed', type=click.INT, default=13)
@click.option('--traj_limitation', type=click.INT, default=90) # 场景限制数 原来是100
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)  # 0.001 10-4
@click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']),
              default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--l2', type=click.FLOAT, default=0.1)
@click.option('--d_iters', type=click.INT, default=10)
@click.option('--rew_scale', type=click.FLOAT, default=0)
def main(logdir, env, expert_path, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, l2, d_iters,
         rew_scale):
    env_ids = [env]
    lrs = [0.001]
    seeds = [seed]
    batch_sizes = [100]
    print('env_ids',env_ids)
    cishu = 0
    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        cishu = cishu +1
        print('第：:',cishu,'次',env_id, seed, lr, batch_size)
        train(logdir + '/airl/' + env_id + '/' + disc_type + '/s-{}/seed-{}'.format(
              traj_limitation, seed),
              env_id, 5e5, lr, batch_size, seed, 10, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters, l2=l2, d_iters=d_iters,
              rew_scale=rew_scale)



if __name__ == "__main__":
    main()
