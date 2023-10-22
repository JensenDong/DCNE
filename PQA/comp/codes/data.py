import os
import torch
import numpy as np
import json
from enum import Enum
from typing import Tuple, List, Dict
from collections import defaultdict
from torch.utils.data import Dataset
import logging


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2


class ModeType(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class EvalDataset(object):
    def __init__(self, train_file, test_file, relations_dict, logger):
        self.train_file = train_file
        self.test_file = test_file
        train_triples = self.read_data(self.train_file)
        test_triples = self.read_data(self.test_file)
        _train_cnt = len(train_triples)
        all_triples = train_triples
        all_triples.update(test_triples)
        logger.info("Begin building full graph ...")
        self.full_graph = Graph(all_triples, relations_dict, logger)
        logger.info(self.full_graph)

    def read_data(self, file):
        """
        :param file:
        :return:  set of (s,r,t) original triples
        """
        triples = set()
        for line in open(file):
            tokens = line.strip().split("\t")
            assert len(tokens) == 3
            s, p, o = [int(i) for i in tokens]
            triples.add((s, p, o))
        return triples


class Graph(object):
    def __init__(self, triples, relations_dict, logger):
        self.triples = triples
        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        num = len(relations_dict) // 2
        logger.info(f"The number of relations is (not including inverse relations): {num}")
        self.inverted = lambda id: id > num-1
        self.invert = lambda id: id-num if self.inverted(id) else id+num

        self._node_set = set()
        for s, r, t in triples:
            relation_args[r]['s'].add(s)
            relation_args[r]['t'].add(t)
            neighbors[s][r].add(t)
            neighbors[t][self.invert(r)].add(s)
            self._node_set.add(t)
            self._node_set.add(s)

        def freeze(d):
            frozen = {}
            for key, subdict in d.items():
                frozen[key] = {}
                for subkey, set_val in subdict.items():
                    frozen[key][subkey] = tuple(set_val)
            return frozen

        self.neighbors = freeze(neighbors)
        self.relation_args = freeze(relation_args)
        logger.info("Finish building graph!")

    def __repr__(self):
        s = ""
        s += "graph.relations_args cnt: %d\t" % len(self.relation_args)
        s += "graph.neighbors cnt: %d\t" % len(self.neighbors)
        s += "graph.neighbors node set cnt: %d" % len(self._node_set)
        return s

    def walk_all(self, start, path):
        """
        walk from start and get all the paths
        :param start:  start entity
        :param path: (r1, r2, ...,rk)
        :return: entities set for candidates path
        """
        set_s = set()
        set_t = set()
        set_s.add(start)
        for _, r in enumerate(path):
            if len(set_s) == 0:
                return set()
            for _s in set_s:
                if _s in self.neighbors and r in self.neighbors[_s]:
                    _tset = set(self.neighbors[_s][r])
                    set_t.update(_tset)
            set_s = set_t.copy()
            set_t.clear()
        return set_s

    def repr_walk_all_ret(self, start, path, MAX_T=20):
        cand_set = self.walk_all(start, path)
        if len(cand_set) == 0:
            return "start{} path:{} end: EMPTY!".format(start, "->".join(list(path)))
        _len = len(cand_set) if len(cand_set) < MAX_T else MAX_T
        cand_node_str = ", ".join(cand_set[:_len])
        return "start{} path:{} end: {}".format(start, "->".join(list(path)), cand_node_str)

    def type_matching_entities(self, path, position="t"):
        assert (position == "t")

        if position == "t":
            r = path[-1]
        elif position == "s":
            r = path[0]
        else:
            print("UNKNOWN position at type_matching_entities")
            raise ValueError(position)
        try:
            if not self.inverted(r):
                return r, self.relation_args[r][position]
            else:
                inv_pos = 's' if position == "t" else "t"
                return r, self.relation_args[self.invert(r)][inv_pos]
        except KeyError:
            print("UNKNOWN path value at type_matching_entities :%s from path:%s" % (r, path))
            return None, tuple()

    def is_trival_query(self, start, path):
        """
        :param path:
        :return: Boolean if True/False, is all candidates are right answers, return True
        """
        cand_set = self.type_matching_entities(path, "t")
        ans_set = self.walk_all(start, path)
        _set = cand_set - ans_set
        if len(_set) == 0:
            return True
        else:
            return False


class DataReader(object):
    def __init__(self, data_path: str, logger, length: int=1):
        logger.info(f"Loading length={length} data...")
        self.entity_dict = self.read_dict(data_path, mode="entities")
        self.relation_dict = self.read_dict(data_path, mode="relations")

        self.train_data = self.read_data(data_path, mode="train", length=length)
        self.valid_data = self.read_data(data_path, mode="dev")
        self.test_data = self.read_data(data_path, mode="test")
        logger.info("Done!")
        self.graph = self.creat_graph(data_path, logger)

    def read_dict(self, data_path: str, mode="entities"):
        """
        Read entities / relations dict.
        Format: dict({name: id})
        """

        with open(os.path.join(data_path, f'{mode}.dict'), 'r') as f:
            element_dict = json.load(f)

        return element_dict

    def read_data(self, data_path: str, mode: str="train", length: int=1):
        """
        Read train / valid / test data.
        """
        if mode == "train":
            file = f"train_{length}hop"
        else:
            file = f"{mode}_nhop"
        data = []
        with open(os.path.join(data_path, file), "r") as f:
            for line in f:
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                rels = [int(i) for i in tokens[1].split(",")]  # str2id
                data.append((int(tokens[0]), tuple(rels), int(tokens[2])))

        return data

    def creat_graph(self, data_path, logger):
        train_base_file = os.path.join(data_path, 'train_1hop')
        test_base_file = os.path.join(data_path, 'test_1hop')
        eval_data = EvalDataset(train_base_file, test_base_file, self.relation_dict, logger)

        return eval_data.full_graph


class TrainDataset(Dataset):
    def __init__(self, data_reader: DataReader, neg_size: int, batch_type: BatchType):
        """
        Dataset for training, inherits `torch.utils.data.Dataset`.
        Args:
            data_reader: DataReader,
            neg_size: int, negative sample size.
        """

        self.triples = data_reader.train_data
        self.len = len(self.triples)
        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)
        self.neg_size = neg_size
        self.batch_type = batch_type

        num = self.num_relation // 2
        self.inverted = lambda id: id > num-1
        self.invert = lambda id: id-num if self.inverted(id) else id+num

        self.hr_map, self.tr_map, self.hr_freq, self.tr_freq = self.two_tuple_count()

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """
        Returns a positive sample and `self.neg_size` negative samples.
        """
        pos_triple = self.triples[idx]
        head, path, tail = pos_triple

        subsampling_weight = self.hr_freq[(head, path)] + self.tr_freq[(tail, path)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        neg_ents = []
        neg_size = 0

        while neg_size < self.neg_size:
            neg_ents_tmp = np.random.randint(self.num_entity, size=self.neg_size * 2)
            if self.batch_type == BatchType.HEAD_BATCH:
                mask = np.in1d(
                    neg_ents_tmp,
                    self.tr_map[(tail, path)],
                    assume_unique=True,
                    invert=True
                )
            elif self.batch_type == BatchType.TAIL_BATCH:
                mask = np.in1d(
                    neg_ents_tmp,
                    self.hr_map[(head, path)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Invalid BatchType: {}'.format(self.batch_type))

            neg_ents_tmp = neg_ents_tmp[mask]
            neg_ents.append(neg_ents_tmp)
            neg_size += neg_ents_tmp.size

        neg_ents = np.concatenate(neg_ents)[:self.neg_size]

        pos_ents= torch.LongTensor([head, tail])
        neg_ents = torch.from_numpy(neg_ents)
        path = torch.LongTensor(path)

        return pos_ents, neg_ents, path, subsampling_weight, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_ents = torch.stack([_[0] for _ in data], dim=0)
        negative_ents = torch.stack([_[1] for _ in data], dim=0)
        paths = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        batch_type = data[0][4]
        return positive_ents, negative_ents, paths, subsample_weight, batch_type

    def two_tuple_count(self):
        """
        Return two dict:
        dict({(h, r): [t1, t2, ...]}),
        dict({(t, r): [h1, h2, ...]}),
        """
        hr_map = {}
        hr_freq = {}
        tr_map = {}
        tr_freq = {}

        init_cnt = 3
        for head, rel, tail in self.triples:
            if (head, rel) not in hr_map.keys():
                hr_map[(head, rel)] = set()

            if (tail, rel) not in tr_map.keys():
                tr_map[(tail, rel)] = set()

            if (head, rel) not in hr_freq.keys():
                hr_freq[(head, rel)] = init_cnt

            if (tail, rel) not in tr_freq.keys():
                tr_freq[(tail, rel)] = init_cnt

            hr_map[(head, rel)].add(tail)
            tr_map[(tail, rel)].add(head)
            hr_freq[(head, rel)] += 1
            tr_freq[(tail, rel)] += 1

        for key in tr_map.keys():
            tr_map[key] = np.array(list(tr_map[key]))

        for key in hr_map.keys():
            hr_map[key] = np.array(list(hr_map[key]))

        return hr_map, tr_map, hr_freq, tr_freq


class TestDataset(Dataset):
    def __init__(self, data_reader: DataReader, mode: ModeType, batch_type: BatchType):
        if mode == ModeType.VALID:
            self.data = data_reader.valid_data
        elif mode == ModeType.TEST:
            self.data = data_reader.test_data

        self.len = len(self.data)
        self.graph = data_reader.graph

        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)

        self.mode = mode
        self.batch_type = batch_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tokens = self.data[idx]
        head, path, tail = tokens

        if self.batch_type == BatchType.TAIL_BATCH:
            q_set = self.graph.walk_all(head, path)
            r, cand_set = self.graph.type_matching_entities(path, "t")
            cand_set = set(cand_set)
            neg_set = cand_set - q_set
        elif self.batch_type == BatchType.HEAD_BATCH:
            path_inv = [self.graph.invert(i) for i in path[::-1]]
            q_set = self.graph.walk_all(tail, path_inv)
            r, cand_set = self.graph.type_matching_entities(path, "t")
            cand_set = set(cand_set)
            neg_set = cand_set - q_set
        else:
            raise ValueError('Invalid BatchType: {}'.format(self.batch_type))

        if len(cand_set) < len(q_set):
            raise RuntimeError("The size of correct answer set shouldn't be larger than the size of candidate set!")

        if len(neg_set) == 0:
            raise RuntimeError("The size of neg_set shouldn't be 0!")

        pos_ents = torch.LongTensor([head, tail])
        neg_ents = torch.LongTensor(list(neg_set))
        path = torch.LongTensor(path)
        neg_size = torch.LongTensor([len(neg_ents)])

        return pos_ents, neg_ents, path, neg_size, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_ents = torch.stack([_[0] for _ in data], dim=0)
        negative_ents = torch.stack([_[1] for _ in data], dim=0)
        paths = torch.stack([_[2] for _ in data], dim=0)
        neg_size = torch.stack([_[3] for _ in data], dim=0)
        batch_type = data[0][4]
        return positive_ents, negative_ents, paths, neg_size, batch_type


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
