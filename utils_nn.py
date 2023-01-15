#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:21:44 2020

@author: crowpeter
"""

import numpy as np
import os
from scipy.io import wavfile
def read_audio(filename):
    rate, audio_data = wavfile.read(filename)
    return audio_data


def stack_features(data, context):
    if context == 0:
        return data
    padded = np.r_[np.repeat(data[0][None], context, axis=0), data,
                   np.repeat(data[-1][None], context, axis=0)]
    stacked_features = np.zeros((len(data), (2 * context + 1) *
                                 data.shape[1])).astype(np.float32)
    for i in range(context, len(data) + context):
        sfea = padded[i - context: i + context + 1]
        stacked_features[i - context] = sfea.reshape(-1)
    return stacked_features

def mu_law_encode(x):
    sign = np.sign(x)
    x_temp = sign*(np.log(1+255*np.abs(x))/np.log(1+255))
    return x_temp

def mu_law_decode(x):
    sign = np.sign(x)
    x_temp = sign*(np.power(1+255,np.abs(x))-1)*(1/255)
    return x_temp


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def generate_random_sample(data, label, index):
    while True:
        batch1 = [data[i][:] for i in index]
        batch1 = np.array(batch1)
        batch2 = [label[i] for i in index]
        batch2 = np.array(batch2)
        yield batch1, batch2


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler)//self.batch_size
        else:
            return (len(self.sampler) + self.batch_size -1)//self.batch_size