import pickle as pkl
import numpy as np
from rl import logger
from tqdm import tqdm

# 这段代码定义了一个名为 Dset 的类，用于创建数据集对象，该对象用于存储和管理用于训练的数据。
class Dset(object):
    # Dset(mh_obs_lstm, mh_actions, mh_obs_next_lstm, all_obs, mh_values, randomize=True, num_agents=num_agents,
    #                           nobs_flag=True)  # 如果缓冲区未创建，则创建一个新的缓冲区。
    def __init__(self, inputs, labels, nobs, all_obs, rews, randomize, num_agents, nobs_flag=False):
        # 输入包括智能体的观察数据,动作,下一个时刻的观察数据(通常是智能体在执行动作后观察到的数据),一个包含所有智能体观察数据的大矩阵,值函数,一个布尔值，指示是否要随机打乱数据,智能体的数量,一个布尔值，指示是否存储下一步观察数据
        self.inputs = inputs.copy() # 将输入数据复制到类属性 inputs 中，以确保原始数据不会被修改。
        self.labels = labels.copy()
        self.nobs_flag = nobs_flag
        if nobs_flag:
            self.nobs = nobs.copy()
        self.all_obs = all_obs.copy()
        self.rews = rews.copy()
        self.num_agents = num_agents
        assert len(self.inputs[0]) == len(self.labels[0]) # 使用断言检查输入数据和标签数据的长度是否一致。这是一个验证步骤，用于确保数据的正确性。
        self.randomize = randomize # 将 randomize 参数的值存储在类属性 randomize 中，以指示是否需要随机打乱数据。
        self.num_pairs = len(inputs[0]) # 计算数据集中的数据对数，通常等于输入数据和标签数据的样本数量。inputs (8, 50, 21, 46)
        self.init_pointer() # 调用 init_pointer 方法，用于初始化数据集指针，后续可以用于遍历数据集。

    # 这段代码定义了名为 init_pointer 的方法，其作用是初始化数据集的指针以便后续可以用于遍历数据集。
    # init_pointer 方法的主要作用是初始化数据集的指针，并且如果需要，将数据集中的数据按随机顺序重新排列，以增加训练的随机性。这对于深度学习模型的训练通常是有益的，因为它可以避免模型对数据的顺序产生过于依赖。
    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs) # 0-49的array
            np.random.shuffle(idx) # 随机打乱索引数组 idx，以便后续数据点的访问是随机的。
            for k in range(self.num_agents):
                self.inputs[k] = self.inputs[k][idx, :]
                self.labels[k] = self.labels[k][idx, :]
                if self.nobs_flag:
                    self.nobs[k] = self.nobs[k][idx, :]
                self.rews[k] = self.rews[k][idx]
            self.all_obs = self.all_obs[idx, :]  # （50，8*46）

    # 这段代码定义了一个 get_next_batch 方法，其作用是从数据集中获取下一个批次（batch）的数据。
    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0: # 检查 batch_size 是否为负数。如果 batch_size 是负数，表示需要返回整个数据集，那么执行以下操作：
            return self.inputs, self.labels, self.all_obs, self.rews
        if self.pointer + batch_size >= self.num_pairs:
            # 检查如果当前指针位置加上 batch_size 大于等于数据集中数据点的总数 (self.num_pairs)，则表示下一个批次会超出数据集的范围，需要重新初始化指针位置，以确保可以循环使用数据集。
            self.init_pointer()
        end = self.pointer + batch_size
        inputs, labels, rews, nobs = [], [], [], []
        for k in range(self.num_agents):
            inputs.append(self.inputs[k][self.pointer:end, :])
            labels.append(self.labels[k][self.pointer:end, :])
            rews.append(self.rews[k][self.pointer:end])
            if self.nobs_flag:
                nobs.append(self.nobs[k][self.pointer:end, :])
        all_obs = self.all_obs[self.pointer:end, :]
        self.pointer = end
        if self.nobs_flag:
            return inputs, labels, nobs, all_obs, rews
        else:
            return inputs, labels, all_obs, rews

    def update(self, inputs, labels, nobs, all_obs, rews, decay_rate=0.9):
        idx = np.arange(self.num_pairs)
        np.random.shuffle(idx)
        l = int(self.num_pairs * decay_rate)
        # decay
        for k in range(self.num_agents):
            self.inputs[k] = self.inputs[k][idx[:l], :]
            self.labels[k] = self.labels[k][idx[:l], :]
            if self.nobs_flag:
                self.nobs[k] = self.nobs[k][idx[:l], :]
            self.rews[k] = self.rews[k][idx[:l]]
        self.all_obs = self.all_obs[idx[:l], :]
        # update
        for k in range(self.num_agents):
            self.inputs[k] = np.concatenate([self.inputs[k], inputs[k]], axis=0)
            self.labels[k] = np.concatenate([self.labels[k], labels[k]], axis=0)
            if self.nobs_flag:
                self.nobs[k] = np.concatenate([self.nobs[k], nobs[k]], axis=0)
            self.rews[k] = np.concatenate([self.rews[k], rews[k]], axis=0)
        self.all_obs = np.concatenate([self.all_obs, all_obs], axis=0)
        self.num_pairs = len(inputs[0])
        self.init_pointer()


class MADataSet(object):

    def __init__(self, expert_path, train_fraction=0.7, ret_threshold=None, traj_limitation=np.inf, randomize=True,
                 nobs_flag=False):
        self.nobs_flag = nobs_flag
        with open(expert_path, "rb") as f:  # 加载专家数据
            traj_data_ = pkl.load(f)
        traj_data1 = traj_data_[:49]
        traj_data2 = traj_data_[50:69]
        traj_data3 = traj_data_[70:90]
        traj_data = np.concatenate([traj_data1, traj_data2], axis=0)
        traj_data = np.concatenate([traj_data, traj_data3], axis=0)
        num_agents = len(traj_data[0]["ob"])
        obs = []
        acs = []
        rets = []
        lens = []
        rews = []
        obs_next = []
        max_time_step = 185

        all_obs = []
        for k in range(num_agents):
            obs.append([])
            acs.append([])
            rews.append([])
            rets.append([])
            obs_next.append([])

        # np.random.shuffle(traj_data) # 随机打乱专家轨迹数据，以确保数据的随机性。

        for traj in tqdm(traj_data):
            if len(lens) >= traj_limitation: # traj_limitation是场景限制数
                break
            for k in range(num_agents):
                # 把obs填充为有历史数据的 每一个时刻的obs为【21,46】
                agent_ob = traj["ob"][k][:,:57]  # 获取当前 agent 的观察数据，形状为 [185, 46]
                # 遍历每个时刻
                obs_k_t = []
                for t in range(max_time_step):
                   # 提取当前时刻和前 20 个时刻的数据，不够 20 个时刻的部分用零填充
                    obs_t = np.zeros((21, 57))
                    if t >= 20:
                        obs_t = agent_ob[t - 20:t + 1]
                    else:
                        obs_t[20 - t:] = agent_ob[:t + 1]
                    obs_k_t.append(obs_t)
                obs[k].append(obs_k_t)

                # # extracted_data = np.concatenate((traj["ob"][k][:,:46], traj["ob"][k][:,47:]), axis=0)
                # obs[k].append(traj["ob"][k][:,:57])
                acs[k].append(traj["ac"][k])
                rews[k].append(traj["rew"][k])
                rets[k].append(traj["ep_ret"][k])
            lens.append(len(traj["ob"][0]))
            all_obs.append(traj["all_ob"])
        # 原来的obs的shape为【8 85 185 46】，现在的应该为【8 85 185 21 46】 (8, 85, 185, 21, 46)
        print("observation shape:", np.shape(obs), len(obs), len(obs[0]), len(obs[0][0]), len(obs[0][0][0]), len(obs[0][0][0]))  # (8, 89, 185, 21, 57) 8 89 185 21 21
        print("action shape:", len(acs), len(acs[0]), len(acs[0][0]), len(acs[0][0][0]))
        print("reward shape:", len(rews), len(rews[0]), len(rews[0][0]))
        print("return shape:", len(rets), len(rets[0]))
        print("all observation shape:", len(all_obs), len(all_obs[0]), len(all_obs[0][0]))
        self.num_traj = len(rets[0]) # 场景数
        self.avg_ret = np.sum(rets, axis=1) / len(rets[0])
        self.avg_len = sum(lens) / len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        self.obs = obs
        self.acs = acs
        self.rews = rews


        for k in range(num_agents):
            self.obs[k] = np.concatenate(self.obs[k])
            self.acs[k] = np.concatenate(self.acs[k])
            self.rews[k] = np.concatenate(self.rews[k])
        self.all_obs = np.concatenate(all_obs)

        # get next observation
        for k in range(num_agents):
            nobs = self.obs[k].copy()
            nobs[:-1] = self.obs[k][1:] # 这一行代码将 nobs 的前n-1个时间步的观察值设置为 self.obs[k] 的后n-1个时间步的观察值。这实际上是将观察值向前滚动了一个时间步，以模拟下一个时间步的观察值。
            nobs[-1] = self.obs[k][0] # 这一行代码将 nobs 的最后一个时间步的观察值设置为 self.obs[k] 的第一个时间步的观察值，以确保环境在一个周期内保持连续。
            obs_next[k] = nobs # 将更新后的 nobs 赋值给 obs_next[k]，表示智能体k的下一个时间步的观察值。
        self.obs_next = obs_next

        if len(self.acs) > 2: # 如果动作维度大于2，这一行代码使用 np.squeeze 函数将动作数据 self.acs 压缩为更低维度，以便后续的处理。这通常用于处理维度不匹配的情况，以确保数据结构正确。
            self.acs = np.squeeze(self.acs)

        assert len(self.obs[0]) == len(self.acs[0])
        self.num_transition = len(self.obs[0]) # 这一行代码将数据集中每个智能体的观察值和动作序列的长度（时间步数）存储在 self.num_transition 变量中。
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                         nobs_flag=self.nobs_flag) # 创建一个数据集对象 self.dset，该对象包含了观察值、动作、下一个观察值、奖励等数据，用于后续的训练和数据采样。
        # for behavior cloning
        self.train_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                              nobs_flag=self.nobs_flag)
        self.val_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                            nobs_flag=self.nobs_flag)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average episode length: %f" % self.avg_len)
        logger.log("Average returns:", str(self.avg_ret))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, ret_threshold, traj_limitation):
    dset = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    a, b, c, d = dset.get_next_batch(64)
    print(a[0].shape, b[0].shape, c.shape, d[0].shape)
    # dset.plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str,
                        default="/Users/LantaoYu/PycharmProjects/multiagent-irl/models/mack-simple_tag-checkpoint20000-20tra.pkl")
    parser.add_argument("--ret_threshold", type=float, default=-9.1)
    parser.add_argument("--traj_limitation", type=int, default=200)
    args = parser.parse_args()
    test(args.expert_path, args.ret_threshold, args.traj_limitation)
