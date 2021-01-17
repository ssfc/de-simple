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
filename = ds_path + "train.txt"  # this number of data in icews14 is 72826;

with open(filename, "r", encoding='UTF-8') as f:
    originalTrainData = f.readlines()  # data in original dataset;

print(filename)
print(len(originalTrainData))
print(originalTrainData[0])
print(originalTrainData[len(originalTrainData)-1])

if not os.path.exists('datasets/split_icews14'):
    os.makedirs('datasets/split_icews14')

writeFileName = 'datasets/split_icews14/' + 'train.txt'

numTrainTriple = int(len(originalTrainData)/2)
with open(writeFileName, 'w', encoding='UTF-8') as f:
    for i in range(numTrainTriple):
        f.write("%s" % originalTrainData[i])

with open(writeFileName, "r", encoding='UTF-8') as f:
    data = f.readlines()

print(writeFileName)
print(len(data))
print(data[0])
print(data[len(data)-1])

