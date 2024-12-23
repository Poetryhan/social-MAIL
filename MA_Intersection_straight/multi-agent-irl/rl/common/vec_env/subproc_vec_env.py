import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
# from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, env_fn_wrapper, is_multi_agent):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        # print('cmd:', cmd, 'data:',data)
        if cmd == 'step':
            reset_info = [False]
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs, actions_new_n
            # obs_n_lstm, obs_n, reward_n, done_n, info_n, ini_step_n, ini_obs, actions_new_n, rew_n_social_generate, collide_situation
            # print('step data:',np.shape(data))  # 10个(8, 62)
            obs_lstm, ob, reward, done, info, ini_step, ini_ob, actions_new_n, rew_n_social_generate, collide_situation = env.step(data)
            ini_obs_lstm = np.zeros((8, 21, 57))
            # print('并行环境代码这里的done:',done)
            if is_multi_agent:
                # print('并行环境中的done:',np.shape(done),done,done[0])
                if done == [True, True, True, True, True, True, True, True]:  # == [True, True, True, True]:  #[0]:   #done == [True, True, True, True]: # 现在的写法是所有车都不在继续生成了，就重置  #done[0]:
                    print('重置环境啦')
                    # obs_n_lstm, obs_n, ini_steps, ini_obs, reset_infos, ini_obs_lstm
                    # obs_n_lstm, obs_n, ini_steps, ini_obs, reset_infos, ini_obs_lstm, rew_n_social_generate, collide_situation
                    reset_data = [0, True]
                    obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm, rew_n_social_generate, collide_situation = env.reset(reset_data)  # , reset_info
                    actions_new_n = np.zeros((8, 2))  # 重置环境的这一时刻，动作都是0
            else:
                if done:
                    reset_data = [0, True]
                    obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm, rew_n_social_generate, collide_situation = env.reset(reset_data)

            remote.send((obs_lstm, ob, reward, done, info, ini_step, ini_ob, reset_info, ini_obs_lstm, actions_new_n, rew_n_social_generate, collide_situation))  # , reset_info
            # obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos
        elif cmd == 'reset':
            # ob = env.reset()  # , reset_info
            # remote.send(ob)  # ,reset_info

            obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm, rew_n_social_generate, collide_situation = env.reset(data)
            # 这里相当于10个并行的cpu都会执行这个操作（在reset函数里有for remotes循环），所以会有10个print，每个remote的结果都会放到相应的cpu中
            # 并行环境中的ob： (8, 18) ini_steps: (8, 1) ini_obs: (8, 18)
            # print('并行环境中的obs_lstm, ：',np.shape(obs_lstm), 'ob:', np.shape(ob),'ini_steps:',np.shape(ini_step),
            #       'ini_obs:',np.shape(ini_ob),'reset_infos:',np.shape(reset_info),'ini_obs_lstm:', np.shape(ini_obs_lstm))
            # 并行环境中的obs_lstm, ： (8, 21, 57) ob: (8, 57) ini_steps: (8, 1) ini_obs: (8, 57) reset_infos: (1,) ini_obs_lstm: (8, 21, 57)
            #  obs_lstm：(8, 21, 56) ob: (8, 56) ini_steps: (8, 1) ini_obs: (8, 56) reset_infos: (1,) ini_obs_lstm: (8, 21, 56)
            remote.send((obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm, rew_n_social_generate, collide_situation))
        elif cmd == 'ini_obs_update':
            # ob = env.reset()  # , reset_info
            # remote.send(ob)  # ,reset_info
            # print('ini_obs_update data:',np.shape(data), data)  # (8, 63)
            obs_lstm, ob, collide_situation = env.ini_obs_update(data)
            # 这里相当于10个并行的cpu都会执行这个操作（在reset函数里有for remotes循环），所以会有10个print，每个remote的结果都会放到相应的cpu中
            # 并行环境中的ob： (8, 18) ini_steps: (8, 1) ini_obs: (8, 18)
            # print('并行环境中的obs_lstm, ：',np.shape(obs_lstm), 'ob:', np.shape(ob)) # 并行环境中的obs_lstm, ： (8, 21, 57) ob: (8, 57)
            #  obs_lstm：(8, 21, 56) ob: (8, 56) ini_steps: (8, 1) ini_obs: (8, 56) reset_infos: (1,) ini_obs_lstm: (8, 21, 56)
            remote.send((obs_lstm, ob, collide_situation))

        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'render':
            env.render()
            remote.send(0)
        else:
            raise NotImplementedError

# 这个类应该是创建子进程的过程，可能这样高效一些。
# 具体来说是异步发送指令信号，同步接受并执行，最后父进程依次收取子进程的信息并打包起来。 self.remotes和self.work_remotes数据的收发。
class SubprocVecEnv(VecEnv): # 这个类用于管理并行化地运行多个环境，以提高训练效率。
    def __init__(self, env_fns, is_multi_agent=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        print('运行的是nvn的SubprocVecEnv')
        print('env_fns',env_fns,nenvs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)]) # 创建 Pipe 对象的元组，用于在主进程和子进程之间传递数据和通信。
        print('len(self.remotes):',len(self.remotes),'self.remotes:',self.remotes,'self.work_remotes:', len(self.work_remotes))
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), is_multi_agent))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)] # ：创建一个包含多个进程的列表 self.ps，每个进程负责运行一个环境。worker 函数将在子进程中执行，接受一个工作管道、环境生成函数以及一个指示是否是多智能体环境的参数。
        #### 在for循环里面有依次使能一个智能体这样。
        # start 10 processes to interact with environments.
        for p in self.ps: # 遍历进程列表。
            p.daemon = True # 将每个进程设置为守护进程，以确保它们在主进程结束时自动终止。
            p.start() # 启动每个进程，开始运行工作。
        for remote in self.work_remotes: # 遍历工作管道列表。
            remote.close() # 关闭工作管道，因为它们将在子进程中使用。

        self.remotes[0].send(('get_spaces', None)) # 通过主进程中的第一个管道发送消息，请求环境的动作空间和观察空间。
        self.action_space, self.observation_space = self.remotes[0].recv() # 从第一个管道接收动作空间和观察空间的信息，并将其分配给相应的变量。
        print('self.remotes[0]:',self.remotes[0])
        print('self.action_space:',self.action_space,'self.observation_space:',self.observation_space)
        self.is_multi_agent = is_multi_agent # 将 is_multi_agent 参数存储在类的属性中，以表示环境是否是多智能体的。
        self.num_agents = None
        if is_multi_agent:
            try:
                n = len(self.action_space)  # 获取动作空间的长度，用于确定多智能体的数量。
                print('try成功：',n)
            except:
                n = len(self.action_space.spaces) # 获取动作空间中每个智能体的子空间的数量，用于确定多智能体的数量。
                print('try不成功：', n)
            self.num_agents = n # 将确定的多智能体数量存储在 self.num_agents 中，以供后续使用。
            print('这里的num_agents:',self.num_agents)

    def step_async(self, actions):
        # if self.is_multi_agent:
        #     remote_action = []
        #     for i in range(len(self.remotes)):
        #         remote_action.append([action[i] for action in actions])
        #     actions = remote_action

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if self.is_multi_agent:
            # obs_lstm, ob, reward, done, info, ini_step, ini_ob, reset_info, ini_obs_lstm
            # obs_lstm, ob, reward, done, info, ini_step, ini_ob, reset_info, ini_obs_lstm, actions_new_n
            obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n, rew_n_social_generate, collide_situation = [], [], [], [], [], [], [], [], [], [], [], []
            for k in range(self.num_agents):
                # print('报错的地方的results：', np.shape(results),results)
                obs_lstm.append([result[0][k] for result in results])
                obs.append([result[1][k] for result in results])
                rews.append([result[2][k] for result in results])
                dones.append([result[3][k] for result in results])
                ini_step_n.append([result[5][k] for result in results])
                ini_obs.append([result[6][k] for result in results])
                ini_obs_lstm.append([result[8][k] for result in results])
                actions_new_n.append([result[9][k] for result in results])
                rew_n_social_generate.append([result[10][k] for result in results])
                collide_situation.append([result[11][k] for result in results])
            reset_infos.append([result[7] for result in results]) # reset_infos (1, 10, 1) [[[False], [False], [False], [False], [False], [False], [False], [False], [False], [False]]]
            try:
                infos = [result[4] for result in results]
            except:
                infos = None

            obs_lstm = [np.stack(ob_lstm) for ob_lstm in obs_lstm]  # 8*10*21*46
            obs = [np.stack(ob) for ob in obs]  # 8*10*46
            rews = [np.stack(rew) for rew in rews]
            dones = [np.stack(done) for done in dones]
            ini_step_n = [np.stack(ini_step) for ini_step in ini_step_n]
            ini_obs = [np.stack(ini_ob) for ini_ob in ini_obs]
            reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            ini_obs_lstm = [np.stack(ini_ob_lstm) for ini_ob_lstm in ini_obs_lstm]
            actions_new_n = [np.stack(action_new_n) for action_new_n in actions_new_n]
            rew_n_social_generate = [np.stack(rew_social_generate) for rew_social_generate in
                                     rew_n_social_generate]  # 8*10*46
            collide_situation = [np.stack(collide_one) for collide_one in
                                 collide_situation]  # 8*10*46
            # reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            print('并行环境中的step的结果：',np.shape(obs_lstm), np.shape(obs), np.shape(rews), np.shape(dones), np.shape(infos),
                                        np.shape(ini_step_n), np.shape(ini_obs), np.shape(reset_infos), np.shape(ini_obs_lstm),
                                        np.shape(actions_new_n))
            return obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n, rew_n_social_generate, collide_situation
        else:
            obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n, rew_n_social_generate, collide_situation = zip(*results)  # , ini_step_n
            return np.stack(obs_lstm), np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(ini_step_n), np.stack(
                ini_obs), np.stack(reset_infos), np.stack(ini_obs_lstm), np.stack(actions_new_n), np.stack(rew_n_social_generate), np.stack(collide_situation)
            # obs, rews, dones, infos = zip(*results)  # , reset_infos
            # return np.stack(obs), np.stack(rews), np.stack(dones), infos  #, reset_infos

    def reset(self, infs):
        for remote, inf in zip(self.remotes, infs):
            # print('这里的一个环境的inf:', inf)
            remote.send(('reset', inf))
        # for remote in self.remotes:
        #     remote.send(('reset', None))
        if self.is_multi_agent:
            results = [remote.recv() for remote in self.remotes]  # 对于每一个cpu的remote来说，得到它的results，也就是并行环境中的ob： (8, 18) ini_steps: (8, 1) ini_obs: (8, 18),这里是把10个cpu的放到一起
            # print('线程这里的results', np.shape(results), results)
            # obs = [[result[k] for result in results] for k in range(self.num_agents)]
            # obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm
            obs_lstm = [[result[0][k] for result in results] for k in
                   range(self.num_agents)]  # result【0】是一个cpu的obs结果，这里的shape应该是【8，10，46】
            obs = [[result[1][k] for result in results] for k in
                   range(self.num_agents)]  # result【0】是一个cpu的obs结果，这里的shape应该是【8，10，46】
            ini_steps = [[result[2][k] for result in results] for k in
                         range(self.num_agents)]  # result【1】是一个cpu的ini_steps结果，这里的shape应该是【8，10，1】
            ini_obs = [[result[3][k] for result in results] for k in
                       range(self.num_agents)]  # result【3】是一个cpu的ini_obs结果，这里的shape应该是【8，10，46】
            reset_infos = [[result[4] for result in
                            results]]  # result【3】是一个cpu的reset_infos结果，这里的shape应该是【1,10，1】是判断这个cpu的环境是否被重置的一个标签
            ini_obs_lstm = [[result[5][k] for result in results] for k in
                       range(self.num_agents)] # result【5】是一个cpu的ini_obs_lstm结果，这里的shape应该是【8，10，21,46】
            rew_n_social_generate = [[result[6][k] for result in results] for k in
                                     range(self.num_agents)]  # result【0】是一个cpu的obs结果，这里的shape应该是【8，10，46】
            collide_situation = [[result[7][k] for result in results] for k in
                                 range(self.num_agents)]
            # reset_infos = [[result[1] for result in results]]
            # for ob in obs:
            #     print([np.size(o) for o in ob ])
            # print([np.size(ob) for ob in obs])
            # obs = [np.stack(ob) for ob in obs]
            # 将结果堆叠成数组
            obs_lstm = [np.stack(ob_lstm) for ob_lstm in obs_lstm]
            obs = [np.stack(ob) for ob in obs]
            ini_steps = [np.stack(ini_step) for ini_step in ini_steps]
            ini_obs = [np.stack(ini_ob) for ini_ob in ini_obs]
            reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            ini_obs_lstm = [np.stack(ini_ob_lstm) for ini_ob_lstm in ini_obs_lstm]
            rew_n_social_generate = [np.stack(rew_social_generate) for rew_social_generate in rew_n_social_generate]
            collide_situation = [np.stack(collide_one) for collide_one in collide_situation]
            # reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            print('10个并行环境中的obs_lstm：', np.shape(obs_lstm), 'obs', np.shape(obs), 'ini_steps:', np.shape(ini_steps), 'ini_obs:', np.shape(ini_obs),
                  'reset_infos:', np.shape(reset_infos), 'ini_obs_lstm:', np.shape(ini_obs_lstm))
            # obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm
            return obs_lstm, obs, ini_steps, ini_obs, reset_infos, ini_obs_lstm, rew_n_social_generate, collide_situation # obs
        else:
            return np.stack([remote.recv() for remote in self.remotes])

    def ini_obs_update(self, ini_obs_old_list):
        for remote, ini_obs_old in zip(self.remotes, ini_obs_old_list):
            remote.send(('ini_obs_update', ini_obs_old))

        if self.is_multi_agent:
            results = [remote.recv() for remote in self.remotes]  # 对于每一个cpu的remote来说，得到它的results，也就是并行环境中的ob： (8, 18) ini_steps: (8, 1) ini_obs: (8, 18),这里是把10个cpu的放到一起
            # print('线程这里的results', np.shape(results), results)  # (10, 2, 8)
            # obs = [[result[k] for result in results] for k in range(self.num_agents)]
            # obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm
            obs_lstm = [[result[0][k] for result in results] for k in
                   range(self.num_agents)]  # result【0】是一个cpu的obs结果，这里的shape应该是【8，10，46】
            obs = [[result[1][k] for result in results] for k in
                   range(self.num_agents)]  # result【0】是一个cpu的obs结果，这里的shape应该是【8，10，46】
            collide_situation = [[result[2][k] for result in results] for k in
                   range(self.num_agents)]

            # reset_infos = [[result[1] for result in results]]
            # for ob in obs:
            #     print([np.size(o) for o in ob ])
            # print([np.size(ob) for ob in obs])
            # obs = [np.stack(ob) for ob in obs]
            # 将结果堆叠成数组
            obs_lstm = [np.stack(ob_lstm) for ob_lstm in obs_lstm]
            obs = [np.stack(ob) for ob in obs]
            collide_situation = [np.stack(collide) for collide in collide_situation]

            # reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            # print('10个并行环境中的obs_lstm：', np.shape(obs_lstm), 'obs', np.shape(obs))  # 10个并行环境中的obs_lstm： (8, 10, 21, 57) obs (8, 10, 57)
            # obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm
            return obs_lstm, obs, collide_situation # obs
        else:
            return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)



if __name__ == '__main__':
    from make_env import make_env

    def create_env(rank):
        def _thunk():
            env = make_env('simple_push')
            env.seed(rank)
            return env
        return _thunk

    env = SubprocVecEnv([create_env(i) for i in range(0, 4)], is_multi_agent=True)  # 四条车道，四个环境
    env.reset()
    obs, rews, dones, _ = env.step(
        [[np.array([0, 1, 0, 0, 0]), np.array([2, 0, 0, 0, 0])] for _ in range(4)]
    )
    print(env.observation_space)
    print(obs)
    print(rews[0].shape)
    print(dones[1].shape)
    env.close()
