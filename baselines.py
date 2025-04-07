import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from itertools import permutations
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)

class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        self.data = torch.stack(data, dim=0).to(device)

        self.size = len(self.data)
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
def get_inner_model(model):
    return model
def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)



def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)
class Baseline(object):
    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss

class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v[:,0], c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    def __init__(self, model, opts, epoch=0):
        super(Baseline, self).__init__()
        self.opts = opts
        self.last = 0
        self._update_model(model, epoch)

    """
        RolloutBaseline更新会伴随着RolloutBaseline使用的数据集发生变化，但要注意的是只要不更新，做validate用的是同一个数据集
        用的同一个数据集才能验证是否当前模型优于之前的模型
    """
    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        # 这一步的目的是能够让模型训练中断后继续训练时用回原本的基线（包括数据集）
        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif dataset[0].size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None: # 为基线生成验证数据集
            self.dataset = TSPDataset(size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")

        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy() # 用新生成的数据集去跑当前基线中的模型
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def _reload_model(self, model):
        # model = copy.deepcopy(self.model)
        # 我替换成以下代码, 源代码的reload应该是有问题的（没有验证过）
        model.load_state_dict(copy.deepcopy(self.model.state_dict()))



    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x, self.model1, self.model2)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        # 用基线中的数据集跑一次model
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        """
            这段代码的逻辑对应Algorithm 1的逻辑
             if OneSidedPairedTTest(zθ,z∗θ)<0.05 then 
                θ∗←θ, ep←epoch
             else
                if epoch − ep > M then
                    θ←θ∗
                end if
             end if
        """
        if candidate_mean - self.mean < 0: # 平均值小，意味着total_travel_time小(reward大), DRL最重要的就是尽可能使reward大
            # Calc p value 单侧T检验
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))

            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch) # TODO 文档标注此处为θ*更新完毕

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
