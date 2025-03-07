import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
# from baselines.common.vec_env import VecEnv, CloudpickleWrapper

def worker(remote, env_fn_wrapper, is_multi_agent):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reset_info = [False]
            obs_lstm, ob, reward, done, info, ini_step, ini_ob, actions_new_n = env.step(data)
            ini_obs_lstm = np.zeros((8, 21, 57))
            if is_multi_agent:
                if done == [True, True, True, True, True, True, True, True]:
                    obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm = env.reset()
                    actions_new_n = np.zeros((8, 2))
            else:
                if done:
                    obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm = env.reset()

            remote.send((obs_lstm, ob, reward, done, info, ini_step, ini_ob, reset_info, ini_obs_lstm, actions_new_n))
        elif cmd == 'reset':
            obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm = env.reset()
            remote.send((obs_lstm, ob, ini_step, ini_ob, reset_info, ini_obs_lstm))
        elif cmd == 'ini_obs_update':
            obs_lstm, ob = env.ini_obs_update(data)
            remote.send((obs_lstm, ob))

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

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, is_multi_agent=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), is_multi_agent))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        # start 10 processes to interact with environments.
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.is_multi_agent = is_multi_agent
        self.num_agents = None
        if is_multi_agent:
            try:
                n = len(self.action_space)
            except:
                n = len(self.action_space.spaces)
            self.num_agents = n

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if self.is_multi_agent:
            obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n = [], [], [], [], [], [], [], [], [], []
            for k in range(self.num_agents):
                obs_lstm.append([result[0][k] for result in results])
                obs.append([result[1][k] for result in results])
                rews.append([result[2][k] for result in results])
                dones.append([result[3][k] for result in results])
                ini_step_n.append([result[5][k] for result in results])
                ini_obs.append([result[6][k] for result in results])
                ini_obs_lstm.append([result[8][k] for result in results])
                actions_new_n.append([result[9][k] for result in results])
            reset_infos.append([result[7] for result in results])
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
            return obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n
        else:
            obs_lstm, obs, rews, dones, infos, ini_step_n, ini_obs, reset_infos, ini_obs_lstm, actions_new_n = zip(*results)
            return np.stack(obs_lstm), np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(ini_step_n), np.stack(
                ini_obs), np.stack(reset_infos), np.stack(ini_obs_lstm), np.stack(actions_new_n)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        if self.is_multi_agent:
            results = [remote.recv() for remote in self.remotes]
            obs_lstm = [[result[0][k] for result in results] for k in
                   range(self.num_agents)]
            obs = [[result[1][k] for result in results] for k in
                   range(self.num_agents)]
            ini_steps = [[result[2][k] for result in results] for k in
                         range(self.num_agents)]
            ini_obs = [[result[3][k] for result in results] for k in
                       range(self.num_agents)]
            reset_infos = [[result[4] for result in
                            results]]
            ini_obs_lstm = [[result[5][k] for result in results] for k in
                       range(self.num_agents)]
            obs_lstm = [np.stack(ob_lstm) for ob_lstm in obs_lstm]
            obs = [np.stack(ob) for ob in obs]
            ini_steps = [np.stack(ini_step) for ini_step in ini_steps]
            ini_obs = [np.stack(ini_ob) for ini_ob in ini_obs]
            reset_infos = [np.stack(reset_info) for reset_info in reset_infos]
            ini_obs_lstm = [np.stack(ini_ob_lstm) for ini_ob_lstm in ini_obs_lstm]
            return obs_lstm, obs, ini_steps, ini_obs, reset_infos, ini_obs_lstm # obs
        else:
            return np.stack([remote.recv() for remote in self.remotes])

    def ini_obs_update(self, ini_obs_old_list):
        for remote, ini_obs_old in zip(self.remotes, ini_obs_old_list):
            remote.send(('ini_obs_update', ini_obs_old))
        if self.is_multi_agent:
            results = [remote.recv() for remote in self.remotes]
            obs_lstm = [[result[0][k] for result in results] for k in
                   range(self.num_agents)]
            obs = [[result[1][k] for result in results] for k in
                   range(self.num_agents)]
            obs_lstm = [np.stack(ob_lstm) for ob_lstm in obs_lstm]
            obs = [np.stack(ob) for ob in obs]

            return obs_lstm, obs
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
    env.close()
