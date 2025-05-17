import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector

        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) * mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim

        return attn_pool, alpha

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        # score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class FullyConnection(nn.Module):
    def __init__(self):
        super(FullyConnection, self).__init__()

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 300)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 100)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc5(x)
        # x = self.fc3(x)
        return x
    
class Emoformer_t(nn.Module):
    def __init__(self, D_m, dropout=0.5,  mode1=0, norm=0):
        super(Emoformer_t, self).__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mode1 = mode1
        self.norm_strategy = norm
        
        norm_train = True
        self.norm1a = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1b = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(D_m, elementwise_affine=norm_train)

        self.norm3a = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(D_m, affine=norm_train)
        
        self.attention_1 = Attention(D_m, n_head=1)
        self.attention_2 = Attention(D_m, n_head=1)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 500)
        
        self.norm_1 = nn.LayerNorm(D_m)
        self.norm_2 = nn.LayerNorm(D_m)
        
        self.fc1 = FullyConnection()
        self.fc2 = FullyConnection()
        
    def forward(self, r1,r2,r3,r4):
        seq_len, batch, feature_dim = r1.size()

        if self.norm_strategy == 1:
            r1 = self.norm1a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm1b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm1c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm1d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        elif self.norm_strategy == 3:
            r1 = self.norm3a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm3b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm3c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm3d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        if self.mode1 == 0:
            r = (r1 + r2 + r3 + r4)/4
        elif self.mode1 == 1:
            r = torch.cat([r1, r2], axis=-1)
        elif self.mode1 == 2:
            r = torch.cat([r1, r2, r3, r4], axis=-1)
        elif self.mode1 == 3:
            r = r1
        elif self.mode1 == 4:
            r = r2
        elif self.mode1 == 5:
            r = r3
        elif self.mode1 == 6:
            r = r4
        elif self.mode1 == 7:
            r = self.r_weights[0]*r1 + self.r_weights[1]*r2 + self.r_weights[2]*r3 + self.r_weights[3]*r4   
        
        r = self.l1(r)
        r_u = r
        output_t1, score_t1 = self.attention_1(r, r)
        r_1= self.norm_1(r + output_t1)
        output_t2, score_t2 = self.attention_2(r_1, r_1)
        r = self.norm_2(r + output_t2)    
        r = self.fc1(r+r_u)
        
        r_u = self.l2(r_u)
        
        r = self.dropout1(r)
        r_u = self.dropout2(r_u)
        
        return r_u,r

class CommonsenseattentionCell(nn.Module):
    def __init__(self, D_u, D_k, D_h,N_s,dropout=0.5):
        super(CommonsenseattentionCell, self).__init__()

        self.D_u = D_u
        self.D_k = D_k
        self.D_h = D_h    

        self.state11 = Attention(D_k)
        self.state12 = Attention(D_k)
        self.state21 = Attention(D_k)
        self.state22 = Attention(D_k)
        self.speaker_embedding = nn.Embedding(N_s, 10)
        
        self.norm1 = nn.LayerNorm(D_k)
        self.norm2 = nn.LayerNorm(D_k)

        # 使用 ModuleDict 为每个说话人分配单独的 GRUCell
        self.g_cells = nn.ModuleDict()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.l1 = nn.Linear(600, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 300)
        self.l4 = nn.Linear(300, 100)
        # self.ll = nn.Linear(self.D_k + 300, self.D_k)

    def _get_gru_cell(self, speaker_id):
        if str(speaker_id) not in self.g_cells:
            self.g_cells[str(speaker_id)] = nn.GRUCell(self.D_k, self.D_h)
        return self.g_cells[str(speaker_id)]

    def _select_parties(self, X, indices):
        q0_sel = [X[i, idx].unsqueeze(0) for i, idx in enumerate(indices)]
        return torch.cat(q0_sel, 0)

    def mapping(self, u):
        u = F.gelu(self.l1(u))
        u = F.gelu(self.l2(u))
        u = self.l3(u)
        return u

    def forward(self, u, sk, nu, lk, a, s0, qmask, qmask_n):
        qm_idx = torch.argmax(qmask, 1)  # 当前说话人
        qm_n_idx = torch.argmax(qmask_n, 1)  # 下一个说话人

        s0_s_sel = self._select_parties(s0, qm_idx)
        s0_l_sel = self._select_parties(s0, qm_n_idx)

        s_0, _ = self.state11(sk.view(-1, self.D_k), u.view(-1, self.D_u))
        s_s, _ = self.state12(sk.view(-1, self.D_k), s_0.view(-1, self.D_u))
        
        s_1, _ = self.state21(lk.view(-1, self.D_k), nu.view(-1, self.D_u))
        s_l, _ = self.state22(lk.view(-1, self.D_k), s_1.view(-1, self.D_u))

        s_s = s_s.squeeze(1).expand(qmask.size(0), -1)
        s_l = s_l.squeeze(1).expand(qmask.size(0), -1)

        # s_s = self.ll(torch.cat((a, s_s), dim=-1))
        

        speaker_states = []
        listener_states = []

        device = s0.device  # 统一设备
        for i, speaker_id in enumerate(qm_idx):
            gru_cell = self._get_gru_cell(speaker_id.item()).to(device)
            state = gru_cell(s_s[i].unsqueeze(0).to(device), 
                             s0_s_sel[i].unsqueeze(0).to(device))
            speaker_states.append(state)

        for i, listener_id in enumerate(qm_n_idx):
            gru_cell = self._get_gru_cell(listener_id.item()).to(device)
            state = gru_cell(s_l[i].unsqueeze(0).to(device), 
                             s0_l_sel[i].unsqueeze(0).to(device))
            listener_states.append(state)

        s = torch.stack(speaker_states, dim=0)#[batch_size, 1, hiddensize]
        l = torch.stack(listener_states, dim=0)

        # Dropout
        s = self.dropout1(s)
        l = self.dropout2(l)

        # Mask 处理
        qmask_ = qmask.unsqueeze(2)
        qmask_n = qmask_n.unsqueeze(2)
        
        qm_idx = qm_idx.unsqueeze(-1).unsqueeze(1).expand(-1, 1, s.shape[-1])  
        qm_n_idx = qm_n_idx.unsqueeze(-1).unsqueeze(1).expand(-1, 1, l.shape[-1])
            

        # 创建与 s0 相同形状的 0 张量
        s_full = torch.zeros_like(s0)
        l_full = torch.zeros_like(s0)


        s_full.scatter_(1, qm_idx, s)
        l_full.scatter_(1, qm_n_idx, l)


        s0 = s0 * (1 - qmask_ - qmask_n) + s_full * qmask_ + l_full * qmask_n

        return s0



class CommonsenseRNN(nn.Module):
    def __init__(self, D_u, D_k, D_h, N_s, dropout=0.5):
        super(CommonsenseRNN, self).__init__()

        self.D_u = D_u
        self.D_k = D_k
        self.D_h = D_h  
        self.N_s = N_s
        self.dropout = nn.Dropout(dropout)
        self.dialogue_cell = CommonsenseattentionCell(self.D_u, self.D_k, self.D_h, self.N_s, dropout)

    def _select_parties(self, X, indices):
        q0_sel = [X[i, idx].unsqueeze(0) for i, idx in enumerate(indices)]
        return torch.cat(q0_sel, 0)

    def forward(self, U, SK, NU, LK, A, qmask):
        s = torch.zeros(qmask.size(1), qmask.size(2), self.D_h).type(U.type())  # batch, party, D_h
        s_ = torch.zeros(1, qmask.size(1), self.D_h).type(U.type())  # 初始化输出
        qmask_n = [qmask[i+1] if i < len(qmask)-1 else qmask[i] for i in range(len(U))]

        for u, sk, nu, lk, a, qm, qm_n in zip(U, SK, NU, LK, A, qmask, qmask_n):
            qm_idx = torch.argmax(qm, 1)  # 当前说话人索引
            s = self.dialogue_cell(u, sk, nu, lk, a, s, qm, qm_n)
            s_ = torch.cat([s_, self._select_parties(s, qm_idx).unsqueeze(0)], 0)

        return s_  # seq_len, batch, D_h
    
    
class Lsattention(nn.Module):
    def __init__(self, D_m, n_head=1, dropout=0.5):
        super(Lsattention, self).__init__() 
        self.attention = Attention(D_m, n_head=n_head, dropout=dropout)  
        self.norm = nn.LayerNorm(D_m)
        self.D_m = D_m
    def forward(self, hidden, S):
        h,_= self.attention(S, hidden)
        h = self.norm(h+ hidden)
        return h
            
        
    
class LSTMModel(nn.Module):

    def __init__(self, D_u, D_k, D_r, D_m, D_e, D_h, N_s, n_classes=6, dropout=0.5, attention=False):

        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm =nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.common = CommonsenseRNN(D_u,D_k,D_r,N_s,dropout=0.5)
        self.emo = Emoformer_t(1024)
        if self.attention:
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')

        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, r1,r2,r3,r4, a ,qmask, umask, UU, SK, NU, LK):
        
        r_u, r = self.emo(r1,r2,r3,r4)
        s = self.common(UU, SK, NU, LK,a,qmask)
        s = s[1:, :, :]
        U = U.float()
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        
        """
        emotions, hidden = self.lstm(torch.cat((U,s), dim=-1))
        
        alpha, alpha_f, alpha_b = [], [], []

        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b
    
    

class MELD3LSTMModel(nn.Module):
    def __init__(self, D_u, D_k, D_r, D_m, D_e, D_h, N_s, n_classes=3, dropout=0.1, attention=False, mode1=0, norm=0):
        super(MELD3LSTMModel, self).__init__()
        self.mode1 = mode1
        self.norm_strategy = norm
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.common = CommonsenseRNN(D_u,D_k,D_r,N_s,dropout=0.1)
        self.emo = Emoformer_t(1024)
        self.attention_1 = Attention(D_h, n_head=1)
        if self.attention:
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")
        
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, r1,r2,r3,r4, a, qmask,  umask, UU, SK, NU, LK):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        r_u, r = self.emo(r1,r2,r3,r4)
        s = self.common(UU, SK, NU, LK,a,qmask)
        s = s[1:, :, :]
        # emotions, hidden = self.lstm(torch.cat((r_u,r,s,a), dim=-1))
        emotions, hidden = self.lstm(torch.cat((r_u,r), dim=-1))
        alpha, alpha_f, alpha_b = [], [], []

        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        
        # hidden = F.relu(self.l1(hidden))
        # hidden = self.l2(hidden)
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden+s), 2)
        # print("log_prob.size() = ", log_prob.size()) # log_prob.size() =  torch.Size([94, 32, 6])
        return log_prob, alpha, alpha_f, alpha_b
    
class MELD7LSTMModel(nn.Module):
    def __init__(self, D_u, D_k, D_r, D_m, D_e, D_h, N_s,n_classes=7, dropout=0.1, attention=False, mode1=0, norm=0):
        super(MELD7LSTMModel, self).__init__()
        self.mode1 = mode1
        self.norm_strategy = norm
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.common = CommonsenseRNN(D_u,D_k,D_r,N_s,dropout=0.1)
        self.emo = Emoformer_t(512)
        self.attention_1 = Attention(D_h, n_head=1)
        if self.attention:
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")
        # self.l4 = nn.Linear(300, 100)
                                                                                                                    
        self.linear = nn.Linear(2 * D_e, D_h)
        self.lsattention = Lsattention(D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self,r1,r2,r3,r4, a ,qmask, umask, UU, SK, NU, LK):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        r_u, r = self.emo(r1,r2,r3,r4)
        s = self.common(UU, SK, NU, LK,a,qmask)
        s = s[1:, :, :]
        # a = F.gelu(self.l4(a))    
        emotions, hidden = self.lstm(torch.cat((r_u,r), dim=-1))
        # emotions, hidden = self.lstm(torch.cat((r,s,a), dim=-1))
        alpha, alpha_f, alpha_b = [], [], []

        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
            
        # print(hidden.shape,s.shape)
        hidden = self.dropout(hidden)
        # hidden = self.lsattention(hidden,s)
        # print(hidden.shape)
        
        log_prob = F.log_softmax(self.smax_fc(hidden+s), 2)
        # print("log_prob.size() = ", log_prob.size()) # log_prob.size() =  torch.Size([94, 32, 6])
        return log_prob, alpha, alpha_f, alpha_b
    


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss

class MaskedFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=5.5, reduction='sum'):
        super(MaskedFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels, mask):
        """
        preds: 预测结果，未经softmax处理的logits
        labels: 真实标签
        mask: 掩码，标记哪些数据点有效
        """
        # 将preds通过softmax获取预测概率
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = torch.log(preds_softmax + 1e-8)

        # 构建一个与preds相同形状的张量，只包含对应labels类别的概率
        preds_softmax_selected = preds_softmax.gather(1, labels.unsqueeze(1)).squeeze()
        preds_logsoft_selected = preds_logsoft.gather(1, labels.unsqueeze(1)).squeeze()

        # 计算Focal Loss
        focal_weight = torch.pow(1 - preds_softmax_selected, self.gamma)
        loss = -1 * focal_weight * preds_logsoft_selected

        # 应用掩码和权重
        if self.weight is not None:
            weight = self.weight[labels]
            loss = loss * weight
        loss = loss * mask.view(-1)  # 确保掩码与loss形状匹配

        if self.reduction == 'mean':
            return torch.mean(loss.float()) / torch.mean(mask.float())
        elif self.reduction == 'sum':
            return torch.sum(loss.float())/torch.sum(mask.float())
        else:
            return loss



class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor



class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss
    










