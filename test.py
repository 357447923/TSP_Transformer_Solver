import os
import pickle
import time
import torch
import math

from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import numpy as np
from transformer import AttentionModel

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger


from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
import warnings
import pprint as pp
warnings = warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    assert opts.decode_type == "beam_search" or opts.decode_type == "greedy", "Unknown decode type: {}".format(opts.decode_type)
    set_decode_type(model, opts.decode_type)
    model.eval()
    c = []
    p = []
    step = 0
    max_step = dataset.size // opts.eval_batch_size
    if dataset.size % opts.eval_batch_size != 0:
        max_step += 1
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(move_to(bat, opts.device))
        return cost.data.cpu(), pi.data.cpu()
    for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
        print(f"正在处理第{step + 1}批, 共{max_step}批")
        cost, pi = eval_model_bat(bat)
        step += 1
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

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        if filename is None:
            data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        else:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        self.data = torch.stack(data, dim=0).to(device)
        self.size = len(self.data)
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]



def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    #print(len(param_groups[0]['params']))
    #print('param_groups', param_groups)
    #print('group[params]', [group['params'] for group in param_groups])
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    #print(len(param_groups[0]['params']))
    #print('ss', [g_norm for g_norm in grad_norms])
    #print('grad_norms', grad_norms)
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    #print('grad_norms_clipped', grad_norms_clipped)
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
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
   # print(x.size())
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood,_ = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

def train_epoch(ci, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(TSPDataset(ci, size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(training_dataloader):

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

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)



def run(opts):
    # Pretty print the run args
    print(123)
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
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
    # Initialize model
    model_class = AttentionModel
    model = model_class(
        opts.embedding_dim,  # 嵌入维度
        opts.hidden_dim,  # 隐藏层维度
        n_encode_layers=opts.n_encode_layers,  # 编码器层数
        n_decode_layers=opts.n_decode_layers,
        mask_inner=True,  # 内部掩码
        mask_logits=True,  # Logits掩码
        normalization=opts.normalization,  # 归一化方法
        tanh_clipping=opts.tanh_clipping,  # tanh裁剪值
        beam_width=opts.beam_width,
        max_len_pe=1000
    ).to(opts.device)


    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    # Start the actual training loop
    val_dataset = TSPDataset(size=opts.graph_size, num_samples=1000, filename='./data/tsp/tsp20_test_seed1234.pkl', distribution=opts.data_distribution)
    _,ind = torch.max(val_dataset.data, dim=2)
    #np.savetxt('valid_data.txt', ind.numpy(), fmt='%d')
    if opts.resume:
        epoch_resume = 999

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    model2 = baseline.baseline.model
    ans, cost = roll(model2, val_dataset, opts)
    print('Avg cost:', torch.mean(cost))
    np.savetxt('answer.txt', ans.numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.numpy(), fmt='%.6f')
if __name__ == "__main__":
    run(get_options())
