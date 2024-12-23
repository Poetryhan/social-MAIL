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
from sandbox.mack.acktr_disc import learn
from sandbox.mack.policies import CategoricalPolicy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #  #屏蔽通知信息、警告信息和报错信（INFO\WARNING\FATAL）


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu):
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
    
    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)  # 四个环境的子进程，并行
    policy_fn = CategoricalPolicy # 分类策略
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=make_env.get_identical(env_id))
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/logger')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener',
                                          'simple_crypto', 'simple_push',
                                          'simple_tag', 'simple_spread', 'simple_adversary']),default='simple')
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
@click.option('--atlas', is_flag=True, flag_value=True)

def main(logdir, env, lr, seed, batch_size, atlas):
    env_ids = [env]
    print(logdir)
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]

    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        print('itertools.product(env_ids, seeds, lrs, batch_sizes):',itertools.product(env_ids, seeds, lrs, batch_sizes))
        print('第一次训练：', env_id, seed, lr, batch_size)
        train(logdir + env_id, env_id, 5e7, lr, batch_size, seed, batch_size // 250)


if __name__ == "__main__":
    main()


'''

这段代码是一个 Python 脚本，用于训练强化学习模型，特别是使用了 ACKTR (Actor-Critic using Kronecker-Factored Trust Region) 算法。它主要的功能包括设置训练环境，配置训练参数，创建模型，然后开始训练模型。

下面是代码的主要逻辑：

导入必要的库和模块，包括 Gym 环境、日志记录、多进程环境（SubprocVecEnv）、模型训练函数（learn）等。

设置一些环境变量，包括限制 TensorFlow 日志级别和指定可见的 GPU。

定义一个 train 函数，该函数用于执行训练的主要逻辑。它接受一些参数，如日志目录（logdir）、环境 ID（env_id）、训练步数（num_timesteps）、学习率（lr）、每个批次的时间步数（timesteps_per_batch）、随机数种子（seed）和 CPU 核心数量（num_cpu）。

在 train 函数内部，它首先创建一个多进程的环境（SubprocVecEnv）来并行地运行多个环境实例。每个环境实例都通过 create_env 函数创建，并设置了随机数种子以确保可重复性。

定义了策略函数（policy_fn）为 CategoricalPolicy，这是一个分类策略，通常用于离散动作空间的问题。

最后，它调用 learn 函数来进行模型的训练。learn 函数接受上述参数，并在训练过程中记录训练进度和性能指标。

main 函数是脚本的入口点，它接受一些命令行参数，如日志目录、环境 ID、学习率、随机数种子、批次大小、以及一个名为 atlas 的标志。然后，它使用 itertools.product 来组合不同的参数组合，以便多次运行训练。

总的来说，这段代码是一个用于训练强化学习模型的训练脚本，通过在多个并行环境中采集经验数据，并使用 ACKTR 算法进行优化，以提高模型性能。它支持对不同的训练参数组合进行多次训练实验。
'''