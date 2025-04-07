import copy

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

# BatchNorm
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


def MHA(Q, K, V, nb_heads, mask=None, clip_value=None):
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
    attn_weights = torch.bmm(Q, K.transpose(1, 2)) / Q.size(-1) ** 0.5  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
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

    # For beam search
    def reorder_selfatt_keys_values(self, t, idx_top_beams):
        bsz, B = idx_top_beams.size()
        zero_to_B = torch.arange(B, device=idx_top_beams.device) # [0,1,...,B-1]
        B2 = self.K_sa.size(0)// bsz
        self.K_sa = self.K_sa.view(bsz, B2, t+1, self.dim_emb) # size(self.K_sa)=(bsz, B2, t+1, dim_emb)
        K_sa_tmp = self.K_sa.clone()
        self.K_sa = torch.zeros(bsz, B, t+1, self.dim_emb, device=idx_top_beams.device)
        for b in range(bsz):
            self.K_sa[b, zero_to_B, :, :] = K_sa_tmp[b, idx_top_beams[b], :, :]
        self.K_sa = self.K_sa.view(bsz*B, t+1, self.dim_emb) # size(self.K_sa)=(bsz*B, t+1, dim_emb)
        self.V_sa = self.V_sa.view(bsz, B2, t+1, self.dim_emb) # size(self.K_sa)=(bsz, B, t+1, dim_emb)
        V_sa_tmp = self.V_sa.clone()
        self.V_sa = torch.zeros(bsz, B, t+1, self.dim_emb, device=idx_top_beams.device)
        for b in range(bsz):
            self.V_sa[b, zero_to_B, :, :] = V_sa_tmp[b, idx_top_beams[b], :, :]
        self.V_sa = self.V_sa.view(bsz*B, t+1, self.dim_emb) # size(self.K_sa)=(bsz*B, t+1, dim_emb)

    # For beam search
    def repeat_selfatt_keys_values(self, B):
        self.K_sa = torch.repeat_interleave(self.K_sa, B, dim=0) # size(self.K_sa)=(bsz.B, t+1, dim_emb)
        self.V_sa = torch.repeat_interleave(self.V_sa, B, dim=0) # size(self.K_sa)=(bsz.B, t+1, dim_emb)

    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # embed the query for self-attention
        # 计算出Q,K,V
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
        h_t = h_t + self.W0_selfatt(MHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)[0])  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze())  # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
        q_a = self.Wq_att(h_t)  # size(q_a)=(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att(MHA(q_a, K_att, V_att, self.nb_heads, mask)[0])  # size(h_t)=(bsz, 1, dim_emb)
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

    # For beam search
    def reorder_selfatt_keys_values(self, t, idx_top_beams):
        for l in range(self.n_decode_layers - 1):
            self.decoder_layers[l].reorder_selfatt_keys_values(t, idx_top_beams)

    # For beam search
    def repeat_selfatt_keys_values(self, B):
        for l in range(self.n_decode_layers - 1):
            self.decoder_layers[l].repeat_selfatt_keys_values(B)

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
                # 公式中之所以没有写V，是因为在这里虽然V用到了，但是并不参与概率的运算
                attn_weights = MHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1]
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
        # 计算每一步的移动距离
        for i in range(1, nb_nodes):
            current_cities = dataset[arange_vec, pi[:, i], :]
            cost += torch.sum((current_cities - previous_cities) ** 2, dim=1) ** 0.5  # dist(current, previous node)
            previous_cities = current_cities
        # 形成回路
        cost += torch.sum((current_cities - first_cities) ** 2, dim=1) ** 0.5  # dist(last, first node)
    return cost, None

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
        node_dim = 2

        # 定义了应该全连接层，作用是将100维的数据转成128维
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        # 保证多头注意力能够平均分掉嵌入维度
        assert embedding_dim % n_heads == 0
        # 起始位置
        self.start_placeholder = nn.Parameter(torch.randn(embedding_dim))

        # decoder layer
        self.decoder = GraphAttentionDecoder(embedding_dim, n_heads, self.n_decode_layers)
        self.WK_att_decoder = nn.Linear(embedding_dim, self.n_decode_layers * embedding_dim)
        self.WV_att_decoder = nn.Linear(embedding_dim, self.n_decode_layers * embedding_dim)
        self.PE = generate_positional_encoding(embedding_dim, max_len_pe).to(device)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
    
    def forward(self, input):
        """
        :param input: (batch_size, graph_size, 2) 每一个样本都有graph_size个城市坐标，坐标为二维欧几里得坐标(x,y)
        :return costs: (batch_size, 1) 每一个样本都会有一个对应的cost
        """
        batch_size = input.size(0)
        x = self._init_embed(input) # 将城市坐标进行嵌入

        # 将城市坐标嵌入与起始token嵌入进行合并后交给编码器处理，合并后的tensor: (batch_size, graph_size + 1, embedding_dim)
        # 编码器执行，输出每个节点的特征以及整个图
        embeddings, _ = self.embedder(torch.cat([x, self.start_placeholder.repeat(batch_size, 1, 1)], dim=1))
        # 跑实验结果时使用，训练不要设置beam_search解码方式
        if self.decode_type == "beam_search":
            return self.beam_search(input, embeddings)
        # 解码器运行，log_p就是自回归解码器输出每一个动作的概率， pi是访问序列
        log_p, pi = self._inner(input, embeddings)
        # 核心逻辑是将访问序列进行闭环。旅行商问题的定义所决定必须这样做
        costs, mask = get_costs(input, pi)
        ll = log_p.sum(dim=1)
        # 建议学习一下下面这个函数的逻辑对log_p进行断言，防止_inner出现有负无穷
        # ll = self._calc_log_likelihood(_log_p, pi, mask)
        return costs, ll, pi

    def _init_embed(self, x):
        return self.init_embed(x)

    def beam_search(self, input, embeddings):
        # beam_search的score设计为score_t = score_(t-1) + log_p(第t个选择的动作的概率)
        batch_size, node_size = input.size(0), input.size(1)
        zero_to_bsz = torch.arange(batch_size, device=device)
        zero_to_beam = torch.arange(self.beam_width, device=device)

        self.decoder.reset_self_att_keys_values()
        k_att = self.WK_att_decoder(embeddings)
        v_att = self.WV_att_decoder(embeddings)
        k_att_tmp = k_att
        v_att_tmp = v_att
        # 将旅行商问题视作为序列生成问题
        for token in range(node_size):
            if token == 0:
                b_t0 = min(self.beam_width, node_size) # 第一步最多有node_size个动作，即最多有node_size个候选路径
                # 动作选择从起始token开始
                idx_start_placeholder = torch.Tensor([node_size]).long().repeat(batch_size).to(device)
                h_start = embeddings[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(batch_size, 1)
                h_t = h_start
                mask = torch.zeros((batch_size, node_size + 1), device=device).bool()
                mask[zero_to_bsz, node_size] = True
                probs = self.decoder(h_t, k_att, v_att, mask)
                # 第一个动作选择时，所有候选路径 score=0+log_p
                score_t = torch.log(probs)
                # 剪枝，只取beam_width个（当小于beam_width个时取node_size个）
                top_score, top_idx = torch.topk(score_t, b_t0, dim=1)
                # 计算beam_width覆盖到的几条候选路径的总得分
                sum_scores = top_score
                zero_to_beam_t0 = torch.arange(b_t0, device=device)
                # 目的：让每条候选路径都维护一个mask，并且对mask进行更新
                mask = mask.unsqueeze(1)
                mask = torch.repeat_interleave(mask, b_t0, dim=1) # 扩展成beam_width个mask
                for b in range(batch_size):
                    mask[b, zero_to_beam_t0, top_idx[b]] = True
                # 每条候选路径也会记录自己的访问序列
                tours = torch.zeros(batch_size, b_t0, node_size, device=device).long()
                tours[:,:,token] = top_idx
                h_t = torch.zeros(batch_size, b_t0, self.embedding_dim, device=device)
                # 为每条候选路径各自选择下一个动作做准备（准备输入数据）
                for b in range(batch_size):
                    h_t[b, :, :] = embeddings[b, top_idx[b], :]
                h_t = h_t + self.PE[token+1].expand(batch_size, b_t0, self.embedding_dim)
                self.decoder.repeat_selfatt_keys_values(b_t0)
                k_att = torch.repeat_interleave(k_att_tmp, b_t0, dim=0)
                v_att = torch.repeat_interleave(v_att_tmp, b_t0, dim=0)

            elif token == 1:
                # 当前的输入数据是(batch_size, beam_width, embedding_dim)格式
                # 而解码器的输入格式应该是(batch_size, embedding_dim)格式
                # 我们可以将输入数据视为是batch_size * beam_width个批次的数据
                # 这样就解决了输入格式不符的问题
                h_t = h_t.view(batch_size*b_t0, self.embedding_dim)
                mask = mask.view(batch_size*b_t0, node_size+1)
                probs = self.decoder(h_t, k_att, v_att, mask)
                probs = probs.view(batch_size, b_t0, node_size+1)
                mask = mask.view(batch_size, b_t0, node_size+1)
                # 还原输入，便于下次处理
                h_t = h_t.view(batch_size, b_t0, self.embedding_dim)
                score_t = torch.log(probs)
                # score_t = score_(t-1) + log_p
                sum_scores = score_t + sum_scores.unsqueeze(2)
                sum_scores_flatten = sum_scores.view(batch_size, -1)
                top_score, top_idx = torch.topk(sum_scores_flatten, self.beam_width, dim=1)
                idx_top_beams = top_idx // (node_size+1) # 用于判断属于哪一个候选序列
                idx_in_beams = top_idx - idx_top_beams * (node_size + 1) # 用于标记应该保留的候选序列的本次动作是选择了哪个节点
                sum_scores = top_score
                mask_tmp = mask.clone()
                mask = torch.zeros(batch_size, self.beam_width, node_size+1, device=device).bool()
                # 更新mask
                for b in range(batch_size):
                    mask[b, zero_to_beam, :] = mask_tmp[b, idx_top_beams[b], :] # 拷贝原mask到新mask中
                for b in range(batch_size):
                    mask[b, zero_to_beam, idx_in_beams[b]] = True
                tours_tmp = tours.clone()
                tours = torch.zeros(batch_size, self.beam_width, node_size, device=device).long()
                for b in range(batch_size):
                    tours[b, zero_to_beam, :] = tours_tmp[b, idx_top_beams[b], :] # 防止要拷贝的数据，提前被覆盖了，逻辑类似a[1]和a[2]交换
                tours[:,:,token] = idx_in_beams # 更新tours
                h_t = torch.zeros(batch_size, self.beam_width, self.embedding_dim, device=device)
                for b in range(batch_size):
                    h_t[b, :, :] = embeddings[b, idx_in_beams[b], :]
                h_t = h_t + self.PE[token+1].expand(batch_size, self.beam_width, self.embedding_dim)
                self.decoder.reorder_selfatt_keys_values(token, idx_top_beams) # 使decoder中上下文数据符合idx_top_beams的顺序
                # 这里没有考虑第二次解码后候选序列个数仍旧小于beam_width的情况，但是由于beam_width设置不宜过大
                # 过大无法平衡搜索质量与搜索效率，故此处不修正该问题
                # k_att和v_att更新都是提供给下一次解码
                k_att = torch.repeat_interleave(k_att_tmp, self.beam_width, dim=0)
                v_att = torch.repeat_interleave(v_att_tmp, self.beam_width, dim=0)

            else:
                # 逻辑与token=1（除了最后两行）的基本一致，参考上面的注释
                h_t = h_t.view(batch_size*self.beam_width,self.embedding_dim)
                mask = mask.view(batch_size*self.beam_width, node_size+1)
                probs = self.decoder(h_t, k_att, v_att, mask)
                probs = probs.view(batch_size, self.beam_width, node_size+1)
                mask = mask.view(batch_size, self.beam_width, node_size+1)
                h_t = h_t.view(batch_size, self.beam_width, self.embedding_dim)
                score_t = torch.log(probs)
                sum_scores = score_t + sum_scores.unsqueeze(2)
                sum_scores_flatten = sum_scores.view(batch_size, -1)
                top_score, top_idx = torch.topk(sum_scores_flatten, self.beam_width, dim=1)
                idx_top_beams = top_idx // (node_size+1)
                idx_in_beams = top_idx - idx_top_beams * (node_size+1)
                sum_scores = top_score
                mask_tmp = mask.clone()
                for b in range(batch_size):
                    mask[b,zero_to_beam, :] = mask_tmp[b, idx_top_beams[b], :]
                for b in range(batch_size):
                    mask[b, zero_to_beam, idx_in_beams[b]] = True
                tours_tmp = tours.clone()
                for b in range(batch_size):
                    tours[b, zero_to_beam, :] = tours_tmp[b, idx_top_beams[b], :]
                tours[:, :, token] = idx_in_beams
                for b in range(batch_size):
                    h_t[b,:,:]=embeddings[b, idx_in_beams[b], :]
                h_t = h_t + self.PE[token+1].expand(batch_size, self.beam_width, self.embedding_dim)
                self.decoder.reorder_selfatt_keys_values(token, idx_top_beams)

        tours_beam_search = tours
        x = input.repeat_interleave(self.beam_width, dim=0)
        # 获取cost最少的一条候选路径
        all_tours_costs, _ = get_costs(x, tours_beam_search.view(batch_size*self.beam_width, node_size))
        all_tours_costs = all_tours_costs.view(batch_size, self.beam_width)
        # 取出每个样本对应的候选序列中cost最小的
        min_tours_costs, idx_min = all_tours_costs.min(dim=1)
        sequence = torch.zeros((batch_size, node_size), dtype=torch.int64)
        # 把每一个cost最小的序列作为每一个样本的输出序列
        sequence[:, :] = tours_beam_search[torch.arange(batch_size), idx_min, :]

        return min_tours_costs, None, sequence

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
        # 预处理
        k_att = self.WK_att_decoder(embeddings)
        v_att = self.WV_att_decoder(embeddings)
        # 序列从起始token开始
        idx_start_placeholder = (torch.Tensor([node_size]).long()
                                 .repeat(batch_size).to(device))
        # 让解码器能够注意到起始token是序列的第0号
        h_start = embeddings[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(batch_size, 1)
        # 初始化掩码，并且把起始token标记为已访问
        mask = torch.zeros((batch_size, node_size + 1), device=device).bool()
        mask[zero_to_bsz, node_size] = True
        self.decoder.reset_self_att_keys_values()
        h_t = h_start
        # 开始node_size次自回归解码
        for i in range(node_size):
            probs = self.decoder(h_t, k_att, v_att, mask)
            # 根据概率和解码策略进行动作选择
            selected = self._select_node(probs, mask)
            # 记录选择的动作的概率以及选择了哪一个城市
            outputs.append(torch.log(probs[zero_to_bsz, selected]))
            sequences.append(selected)
            # 更新解码器的输入
            h_t = embeddings[zero_to_bsz, selected, :]
            h_t = h_t + self.PE[i + 1].expand(batch_size, self.embedding_dim)
            mask = mask.clone()
            mask[zero_to_bsz, selected] = True

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability" # 不能重复访问访问节点
            # 贪心解码是一种简单的解码策略，它在每一步决策时，都会选择当前状态下看起来最优的那个选项，而不考虑这个选择对后续步骤的影响。
        elif self.decode_type == "sampling": # 采样编码策略, 即使对同样的输入，采样每次获得的数据都不同
            """
                为什么强化学习一定要用sampling？
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

            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)