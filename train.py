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
    def __init__(self, n_cities=100, load_dir=None):
        def read_tsp_coordinates(file_name):
            coordinates = []
            with open(file_name, 'r') as f:
                read_data = False
                for line in f:
                    line = line.strip()
                    if line == "NODE_COORD_SECTION":
                        read_data = True
                        continue
                    if line == "EOF":
                        break
                    if read_data:
                        parts = line.split()
                        x, y = float(parts[1]), float(parts[2])
                        coordinates.append((x, y))
            return np.array(coordinates)
        self.n_cities = n_cities
        if load_dir is None:
            self.cities = torch.rand((n_cities, 2), device=device) * 1.
        else:
            self.cities = torch.from_numpy(read_tsp_coordinates(load_dir))
    def __getdis__(self,i, j):
        return torch.sqrt(torch.sum(torch.pow(torch.sub(self.cities[i], self.cities[j]), 2)))

# rollout 和 roll分别用于执行模型的评估过程，遍历数据集并在贪心解码模式下计算每批数据的成本
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
def roll(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    c = []
    p = []
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(move_to(bat, opts.device))
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
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        # self.data_set = []
        # l = torch.rand((num_samples, ci.n_cities - 1)) # 随机生成城市坐标
        # _, ind = torch.sort(l)
        # ind = ind.to(device)
        # ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
        # ind = ind[:,:size,:] + 1
        # ff = ci.cities.unsqueeze(0)
        # ff = ff.expand(num_samples, ci.n_cities, 2) # 此时ff中有num_samples个城市坐标(ci.cities)
        # f = torch.gather(ff, dim = 1, index = ind)
        # f = f.permute(0,2,1) # 把形状(1000, 19, 2) 调为 (1000,2,19)
        # depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1) # 看到这块，我感觉应该得结合论文的第四部分的Part A看
        # self.static = torch.cat((depot, f), dim = 2)
        # depot = torch.zeros(num_samples, 1, 1, dtype=torch.long, device=device)
        # ind = ind[:,:,0:1]
        # ind = torch.cat((depot, ind), dim=1)
        # self.data = torch.zeros(num_samples, size+1, ci.n_cities, device=device)
        # self.data = self.data.scatter_(2, ind, 1.)
        # self.size = len(self.data)
        data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        self.data = torch.stack(data, dim=0).to(device)

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


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def train_batch(
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
    cost, log_likelihood,_ = model(x)

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

def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch) # 调整学习率

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # 每一个epoch都会生成不同的dataset进行训练
    training_dataset = baseline.wrap_dataset(TSPDataset(size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling") # train为sampling解码， validate为greedy解码
    # 一批有512个， 一共有250批
    for batch_id, batch in enumerate(training_dataloader):
        # 处理每个批次的训练
        train_batch(
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
    avg_reward = validate(model, val_dataset, opts)

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
    # ci = Cities()
    # mat包含了ci，这一步之后，才把data.csv数据读入，并且分为12个时间段
    # 并且在该模型中，距离是用时间来进行评估的，生成距离向量中采用三次样条插值
    # 的目的在于得到估计的旅行时间函数f_i_j(t)
    # mat = DistanceMatrix(ci, load_dir='./data.csv', max_time_step = 12)
    # mat = DistanceMatrix(ci)
    # np.savetxt('var.txt', mat.var.cpu().numpy(), fmt='%.6f')
    # np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    # np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    # np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    # np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')
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
        input_size=opts.graph_size,   # 输入大小
        max_t=12,    # 最大时间步长
        beam_width=opts.beam_width,
        max_seq_len=20
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
        # baseline = RolloutBaseline(mat, ci, model, opts) # Rollout基线，Rollout貌似都是使用greedy策略
        baseline = RolloutBaseline(model, opts)
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
    val_dataset = TSPDataset(size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    np.savetxt('valid_data.txt', val_dataset.data.reshape(-1, 2).cpu().numpy(), fmt='%f')
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
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            # 模型训练方法
            train_epoch(
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
    ans, cost = roll(model2, val_dataset, opts)
    print('Avg cost:', torch.mean(cost))
    np.savetxt('answer.txt', ans.cpu().numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.cpu().numpy(), fmt='%.6f')
if __name__ == "__main__":
    run(get_options())
