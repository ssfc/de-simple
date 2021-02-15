# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class Params:

    def __init__(self,
                 ne=500,
                 bsize=512,  # batch size;
                 lr=0.001,  # learning rate;
                 reg_lambda=0.0,
                 emb_dim=100,
                 neg_ratio=20,
                 dropout=0.4,
                 save_each=50,
                 se_prop=0.9):
        self.ne = ne  # number of epochs;
        self.bsize = bsize  # batch size;
        self.lr = lr  # learning rate;
        self.reg_lambda = reg_lambda  # L2 regularization parameter;
        self.static_emb_dim = int(se_prop * emb_dim)
        self.temporal_emb_dim = emb_dim - int(se_prop * emb_dim)
        self.save_each = save_each  # save model after certain epochs;
        self.neg_ratio = neg_ratio  # negative ratio;
        self.dropout = dropout  # dropout probability
        self.se_prop = se_prop  # static embedding proportion

    def str_(self):
        return str(self.ne) + "_" + str(self.bsize) + "_" + str(self.lr) + "_" + str(self.reg_lambda) + "_" + str(
            self.static_emb_dim) + "_" + str(self.neg_ratio) + "_" + str(self.dropout) + "_" + str(
            self.temporal_emb_dim) + "_" + str(self.save_each) + "_" + str(self.se_prop)
