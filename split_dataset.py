# This file is intended to analyze and generate datasets;

import argparse
import os
from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params


dataset = Dataset('icews14')

print(len(dataset.data["train"]))
print(len(dataset.data["valid"]))
print(len(dataset.data["test"]))

ds_name = "icews14"
ds_path = "datasets/" + ds_name.lower() + "/"

with open(ds_path + "train.txt", "r", encoding='UTF-8') as f:
    originalTrainData = f.readlines()  # data in original dataset;

print(len(originalTrainData))
print(originalTrainData[0])
print(originalTrainData[len(originalTrainData)-1])

if not os.path.exists('datasets/split_icews14'):
    os.makedirs('datasets/split_icews14')

numTrainTriple = int(len(originalTrainData)/2)
with open('datasets/split_icews14/' + 'train.txt', 'w', encoding='UTF-8') as f:
    for i in range(numTrainTriple):
        f.write("%s" % originalTrainData[i])





# test whether it is correct;
with open('datasets/split_icews14/' + 'train.txt', "r", encoding='UTF-8') as f:
    data = f.readlines()

print('datasets/split_icews14/' + 'train.txt')
print(len(data))
print(data[0])
print(data[len(data)-1])

