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
from sandbox.mack.policies import MaskedCategoricalPolicy, GaussianPolicy, MASKATTGaussianPolicy


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, l2=0.1, d_iters=1,
          rew_scale=0.1):
    def create_env(rank):
        def _thunk():
            scenario_test_name = 0
            training_label = True
            env = make_env.make_env(env_id, scenario_test_name, training_label)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]
    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)  # 环境
    policy_fn = MASKATTGaussianPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation, nobs_flag=True)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.01, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=identical, l2=l2, d_iters=d_iters,
          rew_scale=rew_scale)

    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='multi-agent-trj/logger')
@click.option('--env', type=click.STRING, default='trj_intersection_4')
@click.option('--expert_path', type=click.STRING,
              default=r'\Data\sinD_nvnxuguan_9jiaohu_social_dayu1.pkl')
@click.option('--seed', type=click.INT, default=13)
@click.option('--traj_limitation', type=click.INT, default=90)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
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
    cishu = 0
    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        cishu = cishu + 1
        train(logdir + '/airl/' + env_id + '/' + disc_type + '/s-{}/seed-{}'.format(traj_limitation, seed),
              env_id, 5e5, lr, batch_size, seed, 10, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters, l2=l2, d_iters=d_iters,
              rew_scale=rew_scale)


if __name__ == "__main__":
    main()
