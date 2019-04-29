import os
import sys
import time
import logging
import datetime
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


def add_gaussian_noise(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise


class CheckInDataset(Dataset):
    def __init__(self, path, map_size, transform=None, noise=False):
        self.path = path
        self.map_size = map_size
        self.transform = transform
        self.noise = noise

        self.datas = []
        df = pd.read_csv(self.path)
        for index, row in df.iterrows():
            rlist = row.tolist()
            sample = [
                [rlist[0]], 
                [rlist[1] / self.map_size[0], rlist[2] / self.map_size[1], self._str2time(rlist[3]), 
                 rlist[4] / self.map_size[0], rlist[5] / self.map_size[1], self._str2time(rlist[6]),
                 rlist[7] / self.map_size[0], rlist[8] / self.map_size[1], self._str2time(rlist[9])],
            ]

            if len(rlist) == 13:
                sample.append(
                    [rlist[10] / self.map_size[0], rlist[11] / self.map_size[1], self._str2time(rlist[12])])
            self.datas.append(sample)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        sample = self.datas[idx]

        if self.transform is not None:
            pass

        sample[0] = torch.FloatTensor(sample[0])
        sample[1] = torch.FloatTensor(sample[1])
        sample[2] = torch.FloatTensor(sample[2])

        if self.noise:
            for jdx in [0, 1, 3, 4, 6, 7]:
                sample[1][jdx] = add_gaussian_noise(sample[1][jdx], 0, 0.01)
            for jdx in [2, 5, 8]:
                sample[1][jdx] = add_gaussian_noise(sample[1][jdx], 0, 1/24)

        return sample[0], sample[1], sample[2]
            

    def _str2time(self, sv):
        slist = sv.split(':')
        stime = int(slist[0]) * 3600 + int(slist[1]) * 60 + int(slist[2])
        ttime = 24 * 3600

        return stime / ttime



if __name__ == '__main__':
    dpath = './canvas/social-checkin-prediction/train.csv'
    map_size = (2915, 1982)
    dataset = CheckInDataset(dpath, map_size)