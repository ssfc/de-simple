# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from scripts import shredFacts


class Dataset:
    """Implements the specified dataloader"""

    def __init__(self, ds_name):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.data = {"train": self.read_file(self.ds_path + "train.txt"),
                     "valid": self.read_file(self.ds_path + "valid.txt"),
                     "test": self.read_file(self.ds_path + "test.txt")}

        self.start_batch = 0
        self.all_facts_as_tuples = None

        self.convert_times()

        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])

        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])

    def read_file(self, filename):  # This function reads data in txt and convert them to array consists of numbers;

        with open(filename, "r", encoding='UTF-8') as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")

            head_id = self.get_entity_id(elements[0])
            rel_id = self.get_relation_id(elements[1])
            tail_id = self.get_entity_id(elements[2])
            timestamp = elements[3]

            facts.append([head_id, rel_id, tail_id, timestamp])

        return facts

    def convert_times(self):
        """
        This function spits the timestamp in the day,date and time.
        """
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date

    def numEnt(self):  # number of entity;
        return len(self.ent2id)

    def numRel(self):  # number of relation;
        return len(self.rel2id)

    def get_entity_id(self, ent_name):
        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def get_relation_id(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch:]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch: self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts

    def add_negative_facts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio

            facts[s1:e1, 0] = (facts[s1:e1, 0] + np.random.randint(low=1, high=self.numEnt(),
                                                                   size=neg_ratio)) % self.numEnt()
            facts[s2:e2, 2] = (facts[s2:e2, 2] + np.random.randint(low=1, high=self.numEnt(),
                                                                   size=neg_ratio)) % self.numEnt()

        return facts

    def add_negative_facts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])

        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:, 0] = (facts1[:, 0] + rand_nums1) % self.numEnt()
        facts2[:, 2] = (facts2[:, 2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)

    def get_next_batch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.add_negative_facts2(bp_facts, neg_ratio))
        return batch

    def was_last_batch(self):
        return self.start_batch == 0


