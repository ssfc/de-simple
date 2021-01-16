# This file is intended to analyze and generate datasets;

import argparse
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
filename = ds_path + "train.txt"

with open(filename, "r", encoding='UTF-8') as f:
    data = f.readlines()

print(len(data))
print(data[0])
print(data[1])






