# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class Params:

    def __init__(self,
                 kg_name = 'icews14',  # Parameter 1;
                 model_type = 'DE_DistMult',  # Parameter 2;
                 ne=500,  # Parameter 3;
                 bsize=512,  # Parameter 4;
                 lr=0.001,  # Parameter 5;
                 reg_lambda=0.0,  # Parameter 6;
                 emb_dim=100,  # Parameter 7;
                 neg_ratio=20,  # Parameter 8;
                 dropout=0.4,  # Parameter 9;
                 save_each=50,  # Parameter 10;
                 se_prop=0.9):  # Parameter 11;

        self.kg_name = kg_name,  # Parameter 1;
        self.model_type = model_type,  # Parameter 2;
        self.ne = ne  # Parameter 3, number of epochs;
        self.bsize = bsize  # Parameter 4, batch size;
        self.lr = lr  # Parameter 5, learning rate;
        self.reg_lambda = reg_lambda  # Parameter 6, L2 regularization parameter;
        self.static_emb_dim = int(se_prop * emb_dim)
        self.temporal_emb_dim = emb_dim - int(se_prop * emb_dim)

        self.neg_ratio = neg_ratio  # Parameter 8, negative ratio;
        self.dropout = dropout  # Parameter 9, dropout probability
        self.save_each = save_each  # Parameter 10, save model after certain epochs;
        self.se_prop = se_prop  # Parameter 11, static embedding proportion

    def str_(self):
        return str(self.ne) + "_" + str(self.bsize) + "_" + str(self.lr) + "_" + str(self.reg_lambda) + "_" + str(
            self.static_emb_dim) + "_" + str(self.neg_ratio) + "_" + str(self.dropout) + "_" + str(
            self.temporal_emb_dim) + "_" + str(self.save_each) + "_" + str(self.se_prop)
