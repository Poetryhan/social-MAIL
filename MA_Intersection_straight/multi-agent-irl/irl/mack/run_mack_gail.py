#!/usr/bin/env python
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
from irl.mack.gail_con_ac import learn
from sandbox.mack.policies import CategoricalPolicy,GaussianPolicy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    
    # identical=[False] + [True for _ in range(12)] + [False] + [True for _ in range(12)] + [False] + [True for _ in range(3)] + [False] + [True for _ in range(1)]
    identical = [False] + [True for _ in range(2)] + [False] + [True for _ in range(4)]


    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = GaussianPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=identical)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='multi-agent-trj/logger')
@click.option('--env', type=click.STRING, default='trj_intersection')
@click.option('--expert_path', type=click.STRING,
              default='multi-agent-trj/expert_trjs/intersection_131_str.pkl')
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--seed', type=click.INT, default=4)
@click.option('--traj_limitation', type=click.INT, default=100)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'centralized', 'single']), default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
def main(logdir, env, expert_path, atlas, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters):
    env_ids = [env]
    lrs = [0.0001]
    seeds = [seed]
    batch_sizes = [50]

    logdir = 'multi-agent-trj/logger'

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/gail/' + env_id + '/' + disc_type + '/s-{}/l-{}-b-{}-d-{}-c-{}/seed-{}'.format(
              traj_limitation, lr, batch_size, dis_lr, bc_iters, seed),
              env_id, 5e6, lr, batch_size, seed, 10, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters)


if __name__ == "__main__":
    main()
