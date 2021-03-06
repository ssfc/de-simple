# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import torch
import torch.nn as nn
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE


class Trainer:
    def __init__(self, dataset, params):
        self.model_name = params.model_type[0]
#        instance_gen = globals()[self.model_name]
#        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.model = nn.DataParallel(globals()[self.model_name](dataset=dataset, params=params))  # 这里建立的是DE_SimplE的实例;
        self.dataset = dataset
        self.params = params

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"  # directory to save models;
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")

    def train(self, early_stop=False):
        self.model.train()

        loss_f = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )  # weight_decay corresponds to L2 regularization

        content = []

        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()

            while not last_batch:

                # (1) 从数据集中取出标签;
                heads, relations, tails, years, months, days = self.dataset.get_next_batch(self.params.bsize,
                                                                                           neg_ratio=self.params.neg_ratio)

                last_batch = self.dataset.was_last_batch()

                scores = self.model(heads, relations, tails, years, months, days)  # (2) forward propagation;

                # Added for softmax
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio + 1)
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)  # (3) 计算代价函数;
                optimizer.zero_grad()  # (4) 清零梯度准备计算;
                loss.backward()  # (5) back propagation;
                optimizer.step()  # (6) 更新训练参数;
                total_loss += loss.cpu().item()

            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")

            content.append(total_loss)

            if epoch % self.params.save_each == 0:
                self.save_model(epoch)

        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"  # directory to save models;
        with open(directory + self.params.str_() + "_loss.txt", "w", encoding='UTF-8') as f:
            for element in content:
                f.write("%s\n" % element)
