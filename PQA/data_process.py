"""
data preprocess for pathquery datasets
"""
import os
import sys
import time
import logging
import argparse
import json
from collections import defaultdict, Counter

log_file = "data_process.log"
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class EvalDataset(object):
    def __init__(self, train_file, test_file, relations_dict):
        self.train_file = train_file
        self.test_file = test_file
        train_triples = self.read_data(self.train_file)
        test_triples = self.read_data(self.test_file)
        _train_cnt = len(train_triples)
        all_triples = train_triples
        all_triples.update(test_triples)
        logging.info("Begin building full graph ...")
        self.full_graph = Graph(all_triples, relations_dict)
        logging.info(self.full_graph)

    def read_data(self, file):
        """
        :param file:
        :return:  set of (s,r,t) original triples
        """
        triples = set()
        for line in open(file):
            s, p, o = line.strip().split("\t")
            triples.add((s, p, o))
        return triples


class Graph(object):
    def __init__(self, triples, relations_dict):
        self.triples = triples
        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        num = len(relations_dict) // 2
        logging.info(f"The number of relations is (not including inverse relations): {num}")
        self.inverted = lambda id: int(id) > num-1
        self.invert = lambda id: str(int(id)-num) if self.inverted(id) else str(int(id)+num)

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
        logging.info("Finish building graph!")

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
            logging.error("UNKNOWN position at type_matching_entities")
            raise ValueError(position)
        try:
            if not self.inverted(r):
                return r, self.relation_args[r][position]
            else:
                inv_pos = 's' if position == "t" else "t"
                return r, self.relation_args[self.invert(r)][inv_pos]
        except KeyError:
            logging.error(
                "UNKNOWN path value at type_matching_entities :%s from path:%s" % (r, path))
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


def generate_eval_files(args, dev_file, test_file):
    with open(os.path.join(args.dir, 'entities.dict'), 'r') as f:
        entities_dict = json.load(f)
    with open(os.path.join(args.dir, 'relations.dict'), 'r') as f:
        relations_dict = json.load(f)

    train_base_file = os.path.join(args.dir, 'train_1hop')
    test_base_file = os.path.join(args.dir, 'test_1hop')
    eval_data = EvalDataset(train_base_file, test_base_file, relations_dict)

    dev_trivial_file = os.path.join(args.dir, "dev_trivial")
    dev_trivial = open(dev_trivial_file, "w")
    dev_nhop_file = os.path.join(args.dir, "dev_nhop")
    dev_nhop = open(dev_nhop_file, "w")

    test_trivial_file = os.path.join(args.dir, "test_trivial")
    test_trivial = open(test_trivial_file, "w")
    test_nhop_file = os.path.join(args.dir, "test_nhop")
    test_nhop = open(test_nhop_file, "w")

    def gef(eval_data, entities_dict, relations_dict, test_file, test_trivial, test_nhop):
        cand_cnt = trivial_cnt = 0
        j = 0
        len_can = 0
        len_q = 0
        len_neg = 0
        max_len_neg = 0
        min_len_neg = 100000
        for line in open(test_file):
            j += 1
            tokens = line.strip().split("\t")
            s = str(entities_dict[tokens[0]])
            t = str(entities_dict[tokens[2]])
            path = tuple([str(relations_dict[i]) for i in tokens[1].split(",")])

            q_set = eval_data.full_graph.walk_all(s, path)
            r, cand_set = eval_data.full_graph.type_matching_entities(path, "t")
            cand_set = set(cand_set)
            neg_set = cand_set - q_set

            sen_id = []
            sen_id.append(s)
            sen_id.append(",".join(path))
            sen_id.append(t)
            if len(neg_set) == 0:   # trivial query
                trivial_cnt += 1
                test_trivial.write("\t".join(sen_id) + "\n")
            else:
                len_can += len(cand_set)
                len_q += len(q_set)
                if len(neg_set) > max_len_neg:
                    max_len_neg = len(neg_set)
                if len(neg_set) < min_len_neg:
                    min_len_neg = len(neg_set)
                len_neg += len(neg_set)
                cand_cnt += 1
                test_nhop.write("\t".join(sen_id) + "\n")

            if len(cand_set) < len(q_set):
                logging.error("ERROR! cand_set %d < q_set %d  at line[%d]:%s" %
                         (len(cand_set), len(q_set), j, line))
            if j % 500 == 0:
                logging.info("processing %d" % j)
        logging.info("candidate count: %d " % cand_cnt)
        logging.info("trivial count: %d " % trivial_cnt)
        logging.info("Finish generating evaluation candidates for %s file" % test_file)
        logging.info("The average size of candi_sets: %s" % (len_can/cand_cnt))
        logging.info("The average size of pos_sets: %s" % (len_q/cand_cnt))
        logging.info("The average size of neg_sets: %s" % (len_neg/cand_cnt))
        logging.info("The maximum size of neg_sets: %s" % max_len_neg)
        logging.info("The minimum size of neg_sets: %s" % min_len_neg)
        test_trivial.close()
        test_nhop.close()

    gef(eval_data, entities_dict, relations_dict, test_file, test_trivial, test_nhop)
    gef(eval_data, entities_dict, relations_dict, dev_file, dev_trivial, dev_nhop)


def get_id(train_file, dev_file, test_file):
    entity_set = set()
    relation_set = set()
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        with open(input_file, "r") as f:
            for line in f:
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                entity_set.add(tokens[0])
                entity_set.add(tokens[2])
                relations = tokens[1].split(",")
                for relation in relations:
                    relation_set.add(relation)

    entity_dict = {entity: i for i, entity in enumerate(sorted(list(entity_set)))}
    relation_dict = {relation: i for i, relation in enumerate(sorted(list(relation_set), reverse=True))}
    logging.info("Number of unique entities: %s" % len(entity_dict))
    logging.info("Number of unique relations: %s" % len(relation_dict))
    return entity_dict, relation_dict


def get_nhop(args, file, entities_dict, relations_dict, mode="train"):
    logging.info(f"get {mode}_nhop data...")
    nhop = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            assert len(tokens) == 3
            li = []
            li.append(str(entities_dict[tokens[0]]))
            relations = [str(relations_dict[i]) for i in tokens[1].split(",")]
            li.append(",".join(relations))
            li.append(str(entities_dict[tokens[2]]))
            nhop[len(relations)].append("\t".join(li)+"\n")

    for n in nhop.keys():
        data = nhop[n]
        logging.info(f"Size of {mode}_{n}hop data: {len(data)}")
        if mode == "train" or (mode == "test" and n == 1):
            with open(os.path.join(args.dir, f'{mode}_{n}hop'), 'w') as f:
                for line in data:
                    f.write(line)


def pathquery_data_preprocess(args, train_file, dev_file, test_file):
    entities_dict, relations_dict = get_id(train_file, dev_file, test_file)
    with open(os.path.join(args.dir, 'entities.dict'), 'w') as f:
        json.dump(entities_dict, f, indent=4)
    with open(os.path.join(args.dir, 'relations.dict'), 'w') as f:
        json.dump(relations_dict, f, indent=4)

    get_nhop(args, train_file, entities_dict, relations_dict, mode="train")
    get_nhop(args, test_file, entities_dict, relations_dict, mode="test")

    generate_eval_files(args, dev_file, test_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default=None,
        help="task name: pathqueryFB, pathqueryWN"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        default=None,
        help="task data directory")
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default="train",
        help="train file name, default train")
    parser.add_argument(
        "--valid",
        type=str,
        required=False,
        default="dev",
        help="valid file name, default valid")
    parser.add_argument(
        "--test",
        type=str,
        required=False,
        default="test",
        help="test file name, default test")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    task = args.task.lower()
    assert task in ["pathqueryfb", "pathquerywn"]
    train_file = os.path.join(args.dir, args.train)
    dev_file = os.path.join(args.dir, args.valid)
    test_file = os.path.join(args.dir, args.test)

    pathquery_data_preprocess(args, train_file, dev_file, test_file)
