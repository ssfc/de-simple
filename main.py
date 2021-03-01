# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse  # Part 1;
from dataset import Dataset  # Part 2;
from params import Params  # Part 3;
from trainer import Trainer  # Part 4;
from tester import Tester  # Part 5;


# --------------------------------------------- 1. this part use argparse --------------------------------------
parser = argparse.ArgumentParser(description='Temporal KG Completion methods')

# Parameter 1;
parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14','split_icews14', 'icews05-15', 'gdelt'])
# Parameter 2;
parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices = ['DE_DistMult', 'DE_TransE', 'DE_SimplE'])
# Parameter 3;
parser.add_argument('-ne', help='Number of epochs', type=int, default=500, choices = [10, 20, 60, 500])
# Parameter 4;
parser.add_argument('-bsize', help='Batch size', type=int, default=256, choices = [256])
# Parameter 5;
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001, choices = [0.001])
# Parameter 6;
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
# Parameter 7;
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=100, choices = [100])
# Parameter 8;
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
# Parameter 9;
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4, choices = [0.0, 0.2, 0.4])
# Parameter 10;
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=20, choices = [5, 20])
# Parameter 11;
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=0.36)

args = parser.parse_args()

# --------------------------------------------- 2. this part use Dataset --------------------------------------
dataset = Dataset(args.dataset)

# --------------------------------------------- 3. this part use Params --------------------------------------
params = Params(
    kg_name = args.dataset,  # Parameter 1;
    model_type = args.model,  # Parameter 2;
    ne=args.ne,  # Parameter 3;
    bsize=args.bsize,  # Parameter 4;
    lr=args.lr,  # Parameter 5;
    reg_lambda=args.reg_lambda,  # Parameter 6;
    emb_dim=args.emb_dim,  # Parameter 7;
    neg_ratio=args.neg_ratio,  # Parameter 8;
    dropout=args.dropout,  # Parameter 9;
    save_each=args.save_each,  # Parameter 10;
    se_prop=args.se_prop  # Parameter 11;
)

# --------------------------------------------- 4. this part use Trainer --------------------------------------
trainer = Trainer(dataset, params, args.model)
trainer.train()

# --------------------------------------------- 5. this part use Tester --------------------------------------

# validating the trained models. we select the model that has the best validation performance as the final model
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset + "/" + params.str_() + "_"

for idx in validation_idx:
    model_path = model_prefix + idx + ".chkpnt"
    tester = Tester(dataset, params, model_path, "valid")
    mrr = tester.test()
    if mrr > best_mrr:
        best_mrr = mrr
        best_index = idx

# testing the best chosen model (of validation set) on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"

tester = Tester(dataset, params, model_path, "test")
tester.test()


