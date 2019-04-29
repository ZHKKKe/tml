import numpy as numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils import export


@export
def baseline(**kwargs):
    return Baseline(**kwargs)


@export
def bl_p2a(**kwargs):
    return BL_P2A(**kwargs)


@export
def bl_p2a_dropout(**kwargs):
    return BL_P2A_Dropout(**kwargs)


@export
def p2b_rnn(**kwargs):
    return RNN(**kwargs)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.l1 = nn.Linear(9, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)

        self.l5l = nn.Linear(256, 128)
        self.l6l = nn.Linear(128, 2)

        self.l5t = nn.Linear(256, 128)
        self.l6t = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))

        xl = self.relu(self.l5l(x))
        xl = self.l6l(xl)

        xt = self.relu(self.l5t(x))
        xt = self.l6t(xt)

        return xl, xt


class BL_P2A(nn.Module):
    def __init__(self):
        super(BL_P2A, self).__init__()

        self.l1 = nn.Linear(9, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.l2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.l3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # self.drop = nn.Dropout(0.5)

        self.l5l = nn.Linear(256, 128)
        self.bn5l = nn.BatchNorm1d(128)
        self.l6l = nn.Linear(128, 2)

        self.l5t = nn.Linear(256, 128)
        self.bn5t = nn.BatchNorm1d(128)
        self.l6t = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))
        x = self.relu(self.bn3(self.l3(x)))

        # x = self.drop(x)

        xl = self.relu(self.bn5l(self.l5l(x)))
        xl = self.l6l(xl)

        xt = self.relu(self.bn5t(self.l5t(x)))
        xt = self.l6t(xt)

        return xl, xt


class BL_P2A_Dropout(nn.Module):
    def __init__(self):
        super(BL_P2A_Dropout, self).__init__()

        self.l1 = nn.Linear(9, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.l2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.l3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.drop = nn.Dropout(0.5)

        self.l5l = nn.Linear(256, 128)
        self.bn5l = nn.BatchNorm1d(128)
        self.l6l = nn.Linear(128, 2)

        self.l5t = nn.Linear(256, 128)
        self.bn5t = nn.BatchNorm1d(128)
        self.l6t = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))
        x = self.relu(self.bn3(self.l3(x)))

        x = self.drop(x)

        xl = self.relu(self.bn5l(self.l5l(x)))
        xl = self.l6l(xl)

        xt = self.relu(self.bn5t(self.l5t(x)))
        xt = self.l6t(xt)

        return xl, xt

def model_str(module):
    row_format = "{name:<40} {shape:>20} = {total_size:>12,d}"
    lines = ["", "model parameters:",
             "=========================",]

    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(name=name, 
            shape=" * ".join(str(p) for p in param.size()), total_size=param.numel()))

    lines.append("=" * 75)
    lines.append(row_format.format(name="all parameters", shape="sum of above", 
        total_size=sum(int(param.numel()) for name, param in params)))
    lines.append("")

    return "\n".join(lines)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.l1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(192)

        self.l2 = nn.Linear(192, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.l5l = nn.Linear(128, 128)
        self.bn5l = nn.BatchNorm1d(128)
        self.l6l = nn.Linear(128, 2)

        self.l5t = nn.Linear(128, 128)
        self.bn5t = nn.BatchNorm1d(128)
        self.l6t = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.l1(r_out[:, time_step, :]))
        
        x = torch.cat((outs[0], outs[1]), dim=1)
        x = torch.cat((x, outs[2]), dim=1)
        x = self.relu(self.bn1(x))
        # x = self.relu(self.bn1(self.l1(x)))
        x = self.relu(self.bn2(self.l2(x)))

        xl = self.relu(self.bn5l(self.l5l(x)))
        xl = self.l6l(xl)

        xt = self.relu(self.bn5t(self.l5t(x)))
        xt = self.l6t(xt)

        return xl, xt
