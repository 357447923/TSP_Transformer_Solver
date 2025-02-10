import os
import time
import torch
import math
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import numpy as np
from transformer import AttentionModel
from scipy.interpolate import CubicSpline

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger


from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
import warnings
import pprint as pp
warnings = warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Cities:
    def __init__(self, n_cities = 100):
        self.n_cities = n_cities
        self.cities = torch.rand((n_cities, 2))
    def __getdis__(self,i, j):
        return torch.sqrt(torch.sum(torch.pow(torch.sub(self.cities[i], self.cities[j]), 2)))

class DistanceMatrix:
    # DistanceMatrix类用于模拟城市间，并实现基于时间变化的距离矩阵
    def __init__(self, ci, max_time_step = 100, load_dir = None):
        self.n_c = ci.n_cities
        self.max_time_step = max_time_step
        with torch.no_grad():
            self.mat = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m2 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m3 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m4 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.var = torch.full((ci.n_cities * ci.n_cities, 1), 0.00, device = device).view(-1)
            if (load_dir is not None):
                temp = np.loadtxt(load_dir, delimiter=',', skiprows=0)
                x = np.arange(max_time_step + 1)
                # 遍历数组，拟合并存储数据
                for k in range(self.n_c):
                    for j in range(self.n_c):
                        i = k * self.n_c + j
                        # 三次样条插值
                        cs = CubicSpline(x, np.concatenate((temp[i], [temp[i,0]]), axis=0), bc_type='periodic')
                        self.m4[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[0], device=device)
                        self.m3[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[1], device=device)
                        self.m2[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[2], device=device)
                        self.mat[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[3], device=device)
    # 与getddd 都用于获取在某一特定时间t下，由状态向量st中指定的城市a和b的距离估计
    # 但getd针对单个时间点和一对城市计算距离，而getddd是批量处理计算
    def __getd__(self, st, a, b, t):
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        tt = torch.floor(t * self.max_time_step) % self.max_time_step
        zz = (torch.floor(t * self.max_time_step) + 1) % self.max_time_step
        c = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + tt.squeeze().long()
        d = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + zz.squeeze().long() 
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)
        z = (t.squeeze() * self.max_time_step - torch.floor(t.squeeze() * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)
        return res
    def __getddd__(self, st, a, b, t):
        s0, s1 = a.size(0), a.size(1)
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        tt = torch.round(t * self.max_time_step) % self.max_time_step
        zz = (torch.round(t * self.max_time_step) + 1) % self.max_time_step 
        c = a * self.n_c * self.max_time_step + b * self.max_time_step + tt.long()
        c = c.view(-1)
        d = a * self.n_c * self.max_time_step + b * self.max_time_step + zz.long()
        d = d.view(-1)
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)
        tt = tt.view(-1)
        ttt = t.expand(s0, s1).contiguous().view(-1)
        z = (ttt * self.max_time_step - torch.floor(ttt * self.max_time_step)) / self.max_time_step 
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)
        return res.view(s0, s1)
# rollout 和 roll分别用于执行模型的评估过程，遍历数据集并在贪心解码模式下计算每批数据的成本
def rollout(mat, model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = model(mat, move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)
def roll(mat, model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    c = []
    p = []
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(mat, move_to(bat, opts.device))
        return cost.data.cpu(), pi.data.cpu()
    
    for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
        cost, pi = eval_model_bat(bat)
        for z in range(cost.size(0)):
            c.append(cost[z])
            p.append(pi[z])
    return torch.stack(p), torch.stack(c)
def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)
def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)


class TSPDataset(Dataset):
    # AI: 构建出了包含城市坐标信息、访问顺序等相关数据的数据集对象
    def __init__(self, ci, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        l = torch.rand((num_samples, ci.n_cities - 1)) # 随机生成城市坐标
        sorted, ind = torch.sort(l)
        ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
        ind = ind[:,:size,:] + 1
        ff = ci.cities.unsqueeze(0)
        ff = ff.expand(num_samples, ci.n_cities, 2) # 此时ff中有num_samples个城市坐标(ci.cities)
        f = torch.gather(ff, dim = 1, index = ind)
        f = f.permute(0,2,1) # 把形状(1000, 19, 2) 调为 (1000,2,19)
        depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1) # 看到这块，我感觉应该得结合论文的第四部分的Part A看
        self.static = torch.cat((depot, f), dim = 2)
        depot = torch.zeros(num_samples, 1, 1, dtype=torch.long)
        ind = ind[:,:,0:1]
        ind = torch.cat((depot, ind), dim=1)
        self.data = torch.zeros(num_samples, size+1, ci.n_cities)
        self.data = self.data.scatter_(2, ind, 1.)
        # q = torch.randn((num_samples, size, 1), dtype=torch.float)  # 起点的容量为0
        # for i in range(num_samples):    # 保证走完全程后负载非负
        #     if q[i].sum() < 0:
        #         q[i] = -q[i]
        # self.quality = torch.cat((depot, q), dim=1)
        self.size = len(self.data)
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return self.data[idx], self.quality[idx]
        return self.data[idx]


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def validate(mat, model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(mat, model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def train_batch(
        mat,
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # 解包批次数据，并将其移动到指定设备上
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # 前向传播计算成本和对数似然
    cost, log_likelihood,_ = model(mat, x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss 计算损失函数
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    # 做梯度下降
    optimizer.zero_grad()   # 清除梯度
    loss.backward() # 反向传播
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step() # 更新模型参数

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

def train_epoch(mat, ci, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch) # 调整学习率

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # 每一个epoch都会生成不同的dataset进行训练
    training_dataset = baseline.wrap_dataset(TSPDataset(ci, size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling") # train为sampling解码， validate为greedy解码
    # 一批有512个， 一共有250批
    for batch_id, batch in enumerate(training_dataloader):
        # 处理每个批次的训练
        train_batch(
            mat,
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    # 每隔指定的epoch或在最后一个epoch时保存状态模型
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    # 验证平均回报有效性
    avg_reward = validate(mat, model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)



def run(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    # 默认不使用tensorboard记录日志
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    import subprocess
    def start_tensorboard():
        subprocess.Popen(
            ["start", "cmd", "/K", "tensorboard --logdir=logs --port 6006"],
            shell=True)
        start_tensorboard()
        print("please goto explorer localhost:6006")
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Figure out what's the problem

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    # 初始化距离向量，城市默认100个节点
    ci = Cities()
    # mat包含了ci，这一步之后，才把data.csv数据读入，并且分为12个时间段
    # 并且在该模型中，距离是用时间来进行评估的，生成距离向量中采用三次样条插值
    # 的目的在于得到估计的旅行时间函数f_i_j(t)
    mat = DistanceMatrix(ci, load_dir='./data.csv', max_time_step = 12)
    np.savetxt('var.txt', mat.var.cpu().numpy(), fmt='%.6f')
    np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')
    # Initialize model
    # 选择注意力模型类
    model_class = AttentionModel
    # 实例化模型（用于训练）
    model = model_class(
        opts.embedding_dim, # 嵌入维度
        opts.hidden_dim,    # 隐藏层维度
        n_encode_layers=opts.n_encode_layers,   # 编码器层数
        mask_inner=True,    # 内部掩码
        mask_logits=True,   # Logits掩码
        normalization=opts.normalization,   # 归一化方法
        tanh_clipping=opts.tanh_clipping,   # tanh裁剪值
        checkpoint_encoder=opts.checkpoint_encoder, # 编码器检查点
        shrink_size=opts.shrink_size,   # 放缩大小
        input_size=opts.graph_size+1,   # 输入大小
        max_t=12,    # 最大时间步长
        beam_width=opts.beam_width
    ).to(opts.device)

    # Overwrite model parameters by parameters to load
    # 加载预训练模型
    model_ = get_inner_model(model)
    # 加载状态字典，后面这个是预训练后的参数，load_data中的state与当前的state在key上冲突，则覆盖
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    # 初始化baseline模型，根据不同的baseline选择对应的baseline类
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta) #指数基线
    
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(mat, ci, model, opts) # Rollout基线，Rollout貌似都是使用greedy策略
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()
    # 如果需要，初始化预热baseline
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    # 这里将模型参数和基线参数添加到优化器中，当基线有可学习参数时，把基线的参数和学习率也添加进来
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    # 如果有预训练的优化器state，则进行加载
    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        # 将优化器的状态数据移动到指定设备
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    # 初始化学习率调度器，根据每个epoch对学习率进行衰减。另外创建验证数据集，数据集大小由图的大小和样本数量决定
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    # Start the actual training loop
    val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    _,ind = torch.max(val_dataset.data, dim=2)
    np.savetxt('valid_data.txt', ind.numpy(), fmt='%d')
    # 继续上次训练没完成的模型
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # 初始化和有些参数的保存到此处结束，下面是正式开始
    if opts.eval_only:
        # 只评估
        validate(mat, model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            # 模型训练方法
            train_epoch(
                mat, # 距离矩阵
                ci, # 城市对象
                model,  # 模型
                optimizer,  # 优化器
                baseline,   # 基线
                lr_scheduler, # 学习率调度器
                epoch,
                val_dataset,    # 验证数据集
                tb_logger,
                opts
            )
    # 得出最终算法，并且给出成本
    model2 = baseline.baseline.model
    ans, cost = roll(mat, model2, val_dataset, opts)
    print('Avg cost:', torch.mean(cost)*1440)
    np.savetxt('answer.txt', ans.numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.numpy(), fmt='%.6f')
if __name__ == "__main__":
    run(get_options())
