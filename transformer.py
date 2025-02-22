import copy
import heapq

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from typing import NamedTuple
import warnings

from torch.distributions import Categorical
from torch.onnx.symbolic_opset9 import tensor
from torch.onnx.utils import attr_pattern

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.bool).view(*mask.size()[:-1], -1)[..., :n]

def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0

def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            # 计算一个标准差值，让参数的初始值满足均匀分布，避免梯度爆炸或者梯度消失
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

# 论文中提到的AttentionLayer
class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection( # 论文中提到的MHA子层
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection( # 前馈子层
                nn.Sequential(
                    # 输入映射层
                    nn.Linear(embed_dim, feed_forward_hidden),
                    # 用来增加非线性性，使得神经网络能够拟合更复杂的函数
                    # 激活层
                    nn.ReLU(),
                    # 输出映射层
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        # 创建了一个由n_layers个MultiHeadAttentionLayer实例组成的序列
        # 每个实例都使用相同的参数。他们作为Sequential实例的参数
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        # x 是h^(0)_i
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
    Inputs : Q of size (bsz, dim_emb, 1)                batch of queries
             K of size (bsz, dim_emb, nb_nodes+1)       batch of keys
             V of size (bsz, dim_emb, nb_nodes+1)       batch of values
             mask of size (bsz, nb_nodes+1)             batch of masks of visited cities
             clip_value is a scalar
    Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
              attn_weights of size (bsz, 1, nb_nodes+1) batch of attention weights
    """
    bsz, nb_nodes, emd_dim = K.size()  # dim_emb must be divisable by nb_heads
    if nb_heads > 1:
        # PyTorch view requires contiguous dimensions for correct reshaping
        Q = Q.transpose(1, 2).contiguous()  # size(Q)=(bsz, dim_emb, 1)
        Q = Q.view(bsz * nb_heads, emd_dim // nb_heads, 1)  # size(Q)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1, 2).contiguous()  # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1, 2).contiguous()  # size(K)=(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # size(K)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1, 2).contiguous()  # size(K)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1, 2).contiguous()  # size(V)=(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz * nb_heads, emd_dim // nb_heads,
                   nb_nodes)  # size(V)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1, 2).contiguous()  # size(V)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1, 2)) / Q.size(
        -1) ** 0.5  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)  # size(mask)=(bsz*nb_heads, nb_nodes+1)
        # attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-inf')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1),
                                                float('-1e9'))  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_weights = torch.softmax(attn_weights, dim=-1)  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V)  # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous()  # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1)  # size(attn_output)=(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1, 2).contiguous()  # size(attn_output)=(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1,
                                         nb_nodes)  # size(attn_weights)=(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1)  # mean over the heads, size(attn_weights)=(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights


class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :
      h_t of size      (bsz, 1, dim_emb)          batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention keys
      V_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention values
      mask of size     (bsz, nb_nodes+1)          batch of masks of visited cities
    Output :
      h_t of size (bsz, nb_nodes+1)               batch of transformed queries
    """

    def __init__(self, dim_emb, nb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        self.BN_selfatt = nn.LayerNorm(dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        self.K_sa = None
        self.V_sa = None

    def reset_selfatt_keys_values(self):
        self.K_sa = None
        self.V_sa = None

    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # embed the query for self-attention
        q_sa = self.Wq_selfatt(h_t)  # size(q_sa)=(bsz, 1, dim_emb)
        k_sa = self.Wk_selfatt(h_t)  # size(k_sa)=(bsz, 1, dim_emb)
        v_sa = self.Wv_selfatt(h_t)  # size(v_sa)=(bsz, 1, dim_emb)
        # concatenate the new self-attention key and value to the previous keys and values
        if self.K_sa is None:
            self.K_sa = k_sa  # size(self.K_sa)=(bsz, 1, dim_emb)
            self.V_sa = v_sa  # size(self.V_sa)=(bsz, 1, dim_emb)
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1)
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1)
        # compute self-attention between nodes in the partial tour
        h_t = h_t + self.W0_selfatt(myMHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)[0])  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze())  # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
        q_a = self.Wq_att(h_t)  # size(q_a)=(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att(myMHA(q_a, K_att, V_att, self.nb_heads, mask)[0])  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze())  # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # MLP
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1))  # size(h_t)=(bsz, dim_emb)
        return h_t

class GraphAttentionDecoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, n_decode_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_decode_layers = n_decode_layers
        self.decoder_layers = nn.ModuleList(
            [AutoRegressiveDecoderLayer(embedding_dim, n_heads) for _ in range(n_decode_layers - 1)])
        self.Wq_final = nn.Linear(embedding_dim, embedding_dim)

    def reset_self_att_keys_values(self):
        for l in range(self.n_decode_layers - 1):
            self.decoder_layers[l].reset_selfatt_keys_values()

    def forward(self, h_t, K_att, V_att, mask):
        for l in range(self.n_decode_layers):
            K_att_l = K_att[:, :,
                      l * self.embedding_dim:(l + 1) * self.embedding_dim].contiguous()  # size(K_att_l)=(bsz, nb_nodes+1, dim_emb)
            V_att_l = V_att[:, :,
                      l * self.embedding_dim:(l + 1) * self.embedding_dim].contiguous()  # size(V_att_l)=(bsz, nb_nodes+1, dim_emb)
            if l < self.n_decode_layers - 1:  # decoder layers with multiple heads (intermediate layers)
                h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)
            else:  # decoder layers with single head (final layer)
                q_final = self.Wq_final(h_t)
                bsz = h_t.size(0)
                q_final = q_final.view(bsz, 1, self.embedding_dim)
                attn_weights = myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1]
        prob_next_node = attn_weights.squeeze(1)
        return prob_next_node

def generate_positional_encoding(d_model, max_len):
    """
    Create standard transformer PEs.
    Inputs :
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
    """
    pe = torch.zeros(max_len, d_model).to(device)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)).to(device)
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)
    return pe

def get_costs(dataset, pi):
    bsz = dataset.size(0)
    nb_nodes = dataset.size(1)
    arange_vec = torch.arange(bsz, device=dataset.device)
    first_cities = dataset[arange_vec, pi[:, 0], :]  # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    cost = torch.zeros(bsz, device=dataset.device)
    with torch.no_grad():
        for i in range(1, nb_nodes):
            current_cities = dataset[arange_vec, pi[:, i], :]
            cost += torch.sum((current_cities - previous_cities) ** 2, dim=1) ** 0.5  # dist(current, previous node)
            previous_cities = current_cities
        cost += torch.sum((current_cities - first_cities) ** 2, dim=1) ** 0.5  # dist(last, first node)
    return cost, None
    # assert (
    #         torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
    #         pi.data.sort(1)[0]
    # ).all(), "Invalid tour"
    #
    # # Gather dataset in order of tour
    # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
    #
    # # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor   # 出发节点
    prev_a: torch.Tensor    # 上一次访问的节点
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor   # 总距离
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property

    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateTSP, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.bool):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                ) # 第一维是batch_size的原因是方便使用
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def addmask(self):
        visited_ = self.visited_.scatter(-1, self.first_a[:, :, None], 1)
        return self._replace(visited_=visited_)        

    def update(self, selected, mat, input):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        # ind = torch.arange(0, input.size(1) - 1).repeat(input.size(0), 1, 1)

        # bdd = mat.var[self.prev_a.squeeze() * mat.n_c + next_node.squeeze()].unsqueeze(1)
        # add = mat.__getd__(ind, self.prev_a, next_node, self.lengths).unsqueeze(1)
        # first_a = prev_a if self.i.item() == 0 else self.first_a
        # add = mat.__getd__(ind, self.prev_a, prev_a).reshape(-1, 1)
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, lengths=lengths,
                             cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class DistanceMatrix:
    # DistanceMatrix类用于模拟城市间，并实现基于时间变化的距离矩阵
    # def __init__(self, ci, max_time_step = 100, load_dir = None):
    def __init__(self, ci):
        self.cities = ci
        self.cities_distance = torch.cdist(self.cities, self.cities).to(device)
    # 与getddd 都用于获取在某一特定时间t下，由状态向量st中指定的城市a和b的距离估计
    # 但getd针对单个时间点和一对城市计算距离，而getddd是批量处理计算
    def __getd__(self, st, a, b):
        # a = torch.gather(st, 1, a)
        # b = torch.gather(st, 1, b)
        cities = self.cities.repeat(st.size(0), 1, 1)
        city_a = torch.gather(cities, 1, a.unsqueeze(-1).expand(-1, -1, 2))
        city_b = torch.gather(cities, 1, b.unsqueeze(-1).expand(-1, -1, 2))
        res = torch.cdist(city_a, city_b) # 计算二维欧氏距离
        return res
    def __getddd__(self, st, a, b):
        s0, s1 = a.size(0), a.size(1) * b.size(1)
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        cities = self.cities.repeat(st.size(0), 1, 1).to(st.device)
        cities_a = torch.gather(cities, 1, a.unsqueeze(-1).expand(-1, -1, 2))
        cities_b = torch.gather(cities, 1, b.unsqueeze(-1).expand(-1, -1, 2))
        res = torch.cdist(cities_a, cities_b)
        return res.view(s0, s1)



class AttentionModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 n_decode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 beam_width=2,
                 max_len_pe=1000):
        
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.n_decode_layers = n_decode_layers
        self.decode_type = None
        self.beam_width = beam_width
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        # step_context_dim = 4 * embedding_dim  # Embedding of first and last node
        # node_dim = 100
        node_dim = 2
        self.W_placeholder = nn.Parameter(torch.Tensor(4 * embedding_dim))
        # 作用是生成服从-1到1的服从均匀分布的随机数
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        # 定义了应该全连接层，作用是将100维的数据转成128维
        # 这样可以获得数据的更多特征，学习更多的信息
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        # 看论文，这些都是与论文息息相关的
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        # self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        # self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        # self.embed_static_traffic = nn.Linear(node_dim * max_t, embedding_dim)
        # self.embed_static_traffic = nn.Linear(input_size, embedding_dim)
        # self.embed_static = nn.Linear(2 * embedding_dim, embedding_dim)
        self.PE = generate_positional_encoding(embedding_dim, max_len_pe)
        assert embedding_dim % n_heads == 0
        self.start_placeholder = nn.Parameter(torch.randn(embedding_dim))

        # decoder layer
        self.decoder = GraphAttentionDecoder(embedding_dim, n_heads, self.n_decode_layers)
        self.WK_att_decoder = nn.Linear(embedding_dim, self.n_decode_layers * embedding_dim)
        self.WV_att_decoder = nn.Linear(embedding_dim, self.n_decode_layers * embedding_dim)
        self.PE = generate_positional_encoding(embedding_dim, max_len_pe).to(device)
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        # self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.project_traffic = nn.Linear(input_size*input_size, embedding_dim, bias=False)
        # self.project_visit = nn.Linear(input_size, embedding_dim, bias=False)
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
    
    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        batch_size, node_size = input.size(0), input.size(1)
        x = self._init_embed(input) # 将每个客户节点i转换为ID嵌入x_i
        # mat = DistanceMatrix(input)
        # y = self.embed_static_traffic(mat.cities_distance) # 将估计的流量状况(mat)转化为流量嵌入y_i  # TODO 论文指出在DPDP数据集中还嵌入了运输量q，但我还没发现相关代码
        # 编码器执行，输出每个节点的特以及整个图
        # embeddings, _ = self.embedder(self.embed_static(torch.cat((x, y), dim = 2))) # 将x和y连接起来，产生{h^(0)_0},并且作为图注意力编码器的前向传播的输入
        # 此处embeddings应该为(bsz, nb_nodes+1, dim_emb)
        embeddings, _ = self.embedder(torch.cat([x, self.start_placeholder.repeat(batch_size, 1, 1)], dim=1))
        # 解码器运行，log_p就是论文中解码器的输出p(每一个下一步的所有可能的概率)， pi是访问序列，state是论文中Fig.1中的state
        _log_p, pi = self._inner(input, embeddings)
        cost, mask = get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        return cost, ll, pi

    def _init_embed(self, x):
        return self.init_embed(x)
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _sampling(self, input, fixed, mat, state):
        outputs = []
        sequences = []

        while not (state.all_finished()):
            # 获取20个节点的选择概率，以及掩码（visited）
            log_p, mask = self._get_log_p(fixed, state, mat, input)  # self-attention feed_back mask

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            # 一次性选择为每个(1, 20)的tensor选择出一个节点, 并且update后，mask中对应的该节点会为True
            state = state.update(selected, mat, input)  # state.lengths在此处进行修改

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)  # TODO 保存sequence结果
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state

    def _beam_search(self, input, fixed, mat, state, beam_width = 2):
        class BeamState:
            def __init__(self, init_state, init_seq, init_outputs):
                self.state = init_state
                self.sequences = init_seq
                self.outputs = init_outputs
        beam_states = [(BeamState(state, [], []), 0.0)] # (BeamState, score 概率和) 用于剪枝
        while not beam_states[0][0].state.all_finished():
            new_beam_states = []
            for i in range(len(beam_states)):
                    beam_state = beam_states[i][0]

                    log_p, mask = self._get_log_p(fixed, beam_state.state, mat, input)
                    topk_probs, topk_nodes = torch.topk(log_p[:, 0, :], beam_width)
                    for j in range(beam_width):
                        sequences = beam_state.sequences  + [topk_nodes[:, j]]
                        outputs = beam_state.outputs + [log_p[:, 0, :]]
                        init_state = beam_state.state.update(topk_nodes[:, j], mat, input)
                        probs = topk_probs[:,  j]
                        if torch.isinf(probs).all():
                            break
                        msk = ~torch.isinf(probs)
                        new_beam_states.append((BeamState(init_state, sequences, outputs),
                                            beam_states[i][1] + torch.mean(probs[msk]).item()))

            # 剪枝，取最大的几个
            new_beam_states = heapq.nlargest(beam_width, new_beam_states, key=lambda x: x[1])
            beam_states = new_beam_states

        return (torch.stack(beam_states[0][0].outputs, 1),
                torch.stack(beam_states[0][0].sequences, 1),
                beam_states[0][0].state)

    # 解码器相关函数
    def _inner(self, input, embeddings):
        """
        :param input: 客户的访问状态
        :param mat: 当前交通状况（距离）
        :param embeddings: 解码器计算出的图
        :output output: p0, ..., pc
        :output sequences: 节点访问选择顺序
        """
        outputs = []
        sequences = []
        batch_size, node_size = input.size(0), input.size(1)
        zero_to_bsz = torch.arange(batch_size, device=device)

        # fixed = self._precompute(embeddings)
        k_att = self.WK_att_decoder(embeddings)
        v_att = self.WV_att_decoder(embeddings)
        idx_start_placeholder = (torch.Tensor([node_size]).long()
                                 .repeat(batch_size).to(device))
        h_start = embeddings[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(batch_size, 1)

        mask = torch.zeros((batch_size, node_size + 1), device=device).bool()
        mask[zero_to_bsz, node_size] = True
        self.decoder.reset_self_att_keys_values()
        h_t = h_start
        for i in range(node_size):
            probs = self.decoder(h_t, k_att, v_att, mask)
            selected = self._select_node(probs, mask)
            log_p = torch.log(probs[zero_to_bsz, :]).view(batch_size, 1, -1)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            h_t = embeddings[zero_to_bsz, selected, :]
            h_t = h_t + self.PE[i + 1].expand(batch_size, self.embedding_dim)
            mask = mask.clone()
            mask[zero_to_bsz, selected] = True

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)


        # if self.decode_type == "greedy":
        #     return self._beam_search(input, fixed, mat, state, self.beam_width)
        # elif self.decode_type == "sampling":
        #     return self._sampling(input, fixed, mat, state)
        #
        # assert False, "Unknown decode type"

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency 为提高效率，graph embedding的fixed context投影仅计算一次。
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    def _get_log_p(self, fixed, state, mat, input, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, input))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p) softmax的输入常称为logits
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"
        # if self.test_decode_type == "greedy": # TODO 不需要的时候记得删
        #     _, selected = probs.max(1)
        #     assert not mask.gather(1, selected.unsqueeze(
        #         -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        #     return selected
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability" # 不能重复访问访问节点
            # 贪心解码是一种简单的解码策略，它在每一步决策时，都会选择当前状态下看起来最优的那个选项，而不考虑这个选择对后续步骤的影响。
        elif self.decode_type == "sampling": # 采样编码策略, 即使对同样的输入，采样每次获得的数据都不同
            """
                我的疑问：为什么强化学习一定要用sampling？
                AI回答：
                    1. 动作选择的随机性需求
                        在强化学习中，智能体需要根据策略网络输出的动作概率分布来选择动作。
                        这种选择不能总是确定性的，因为如果智能体总是选择概率最高的动作，就会导致探索不足。
                        例如，在一个迷宫探索游戏中，如果智能体总是朝着它认为最有可能获得奖励的方向移动，可能会错过其他潜在的、更优的路径。
                        torch.multinomial函数可以根据给定的动作概率分布进行随机抽样，使得智能体能够以一定的概率选择不同的动作，从而实现探索与利用的平衡。
                        假设策略网络输出了动作概率分布action_probs = [0.1, 0.3, 0.6]，这表示有 3 个动作，它们被选中的概率分别为 0.1、0.3 和 0.6。
                        用torch.multinomial(action_probs, 1)（假设只选择一个动作），
                        智能体就可以以这些概率随机地选择一个动作，有时候会选择概率较低的动作，这就增加了探索环境的机会。
                    2. 符合概率分布的动作选择
                        强化学习的策略本质上是一个概率分布，它定义了在每个状态下选择每个动作的概率。
                        torch.multinomial能够准确地从这个概率分布中进行抽样，保证动作选择的概率符合策略网络所学习到的分布。
                        这对于正确地训练策略网络非常重要，因为训练过程依赖于智能体按照策略所规定的概率进行动作选择，然后根据反馈来更新策略。
                        例如，在基于策略梯度的算法中，如 A2C 或 A3C 算法，
                        策略网络输出动作概率，智能体使用torch.multinomial选择动作并与环境交互，得到奖励。
                        然后根据奖励和动作概率来计算策略梯度，以更新策略网络，使得策略网络输出的概率分布能够朝着获得更多奖励的方向调整。
                        如果动作选择不按照正确的概率分布进行，策略梯度的计算就会出现偏差，导致训练效果不佳。
                    """


            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected
    def _get_parallel_step_context(self, embeddings, state, mat, input):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        b_s, i_s = embeddings.size(0), embeddings.size(1)
        current_node = state.get_current_node()
        if state.i.item() == 0:
            # First and only step, ignore prev_a (this is a placeholder)
            return self.W_placeholder[None, None, :].expand(b_s, 1, self.W_placeholder.size(-1))
        # else:
        #     return embeddings.gather(
        #         1,
        #         torch.cat((state.first_a, current_node), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1))
        #     ).view(b_s, 1, -1)

        # ind = torch.arange(0, i_s - 1).repeat(b_s, 1, 1)
        # current_traffic = self.project_traffic(mat.__getddd__(ind, self.xx.repeat(b_s, 1, 1).view(b_s, i_s*i_s), self.yy.repeat(b_s, 1, 1).view(b_s, i_s*i_s), state.lengths).view(b_s, 1, i_s*i_s))
        # current_traffic = self.project_traffic(mat.__getddd__(ind, self.xx.repeat(b_s, 1).view(b_s, i_s), self.yy.repeat(b_s, 1).view(b_s, i_s)).view(b_s, 1, i_s * i_s))
        current_traffic = self.project_traffic(mat.cities_distance.view(b_s, 1, i_s * i_s))
        current_visit = self.project_visit(state.visited_.float())
        ss = embeddings.gather(1, torch.cat((state.first_a, state.prev_a), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1)))
        return torch.cat((ss.view(b_s, 1, -1), current_traffic, current_visit), dim=2)
        
    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)