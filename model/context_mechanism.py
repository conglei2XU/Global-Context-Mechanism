import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContext(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.3):
        super(GlobalContext, self).__init__()
        self.input_size = hidden_size
        self.dropout_rate = dropout_rate
        self.gate = Gate(input_size=hidden_size * 2, output_size=hidden_size)
        # self.gate = GateAttention(input_size=hidden_size * 2, output_size=hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, forward_global_cell, backward_global_cell):
        max_len = x.size(1)

        forward_info = forward_global_cell.unsqueeze(1).repeat(1, max_len, 1)
        backward_info = backward_global_cell.unsqueeze(1).repeat(1, max_len, 1)

        # concat forward infor, backward info as its origin directions
        # global_info = torch.cat([forward_info, backward_info], dim=-1)

        # concat forward information, backward information reverse.
        global_info = torch.cat([backward_info, forward_info], dim=-1)
        i_g, i_c = self.gate(torch.cat([global_info, x], dim=-1))
        # macbert-base-chinese on weibo not use dropout
        global_info = self.dropout(global_info)
        output = global_info * i_g + x * i_c
        # output = global_info + x
        return output, (i_g, i_c)


class Gate(nn.Module):
    def __init__(self, input_size, output_size=None):
        super(Gate, self).__init__()
        if output_size is None:
            output_size = input_size
        self.gate_ = nn.Linear(input_size, output_size)
        self.gate_g = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.dropout(x)
        i_g = torch.sigmoid(self.gate_(x))
        i_c = torch.sigmoid(self.gate_g(x))
        return i_g, i_c


class GateAttention(nn.Module):
    def __init__(self, input_size, output_size=None):
        super(GateAttention, self).__init__()
        self.inner_layer = nn.Linear(input_size, input_size)
        self.weight_layer = nn.Linear(input_size, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.dropout(x)
        intermediate_state = torch.relu(self.inner_layer(x))
        weights = self.weight_layer(intermediate_state)
        normal_weights = torch.softmax(weights, dim=-1)
        i_g, i_c = normal_weights[:, :, 0], normal_weights[:, :, 1]
        return i_g.unsqueeze(-1), i_c.unsqueeze(-1)




class MySequential(nn.Module):
    def __init__(self, input_size):
        super(MySequential, self).__init__()
        self.input_size = input_size
        self.sequential = nn.Sequential(nn.Dropout(p=0.1),
                                        nn.Linear(self.input_size * 2, self.input_size),  # 2/3
                                        nn.Tanh(),
                                        nn.Linear(self.input_size, self.input_size // 2),
                                        nn.Tanh(),
                                        nn.Linear(self.input_size // 2, 1),
                                        )

    def forward(self, x):
        return self.sequential(x)


class SAN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: (batch_size, seq_length, hidden_size)
        :param src_mask: (batch_size, seq_length)
        :param src_key_padding_mask:
        :return:
        """
        src_mask = src_mask.to(torch.bool)
        # attn_mask = src_mask.repeat(self.nhead, 1, 1,)
        src2, _ = self.self_attn(src, src, src)
        # print(src2.size())
        # print(src.size())
        # src = src + self.dropout(src2)
        src = src + src2
        # src = torch.cat((src, src2), dim=-1)
        # print(src.shape())
        # apply layer normalization
        src = self.norm(src)
        return src, []
