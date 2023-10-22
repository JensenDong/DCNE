import os
import logging
import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, ModeType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    @abstractmethod
    def comp(self, path):
        """
        path: [batch_size, path_length]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            positive_ents, paths = sample

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_ents[:, 0]
            ).unsqueeze(1)

            relation = self.comp(paths)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_ents[:, 1]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            positive_ents, negative_ents, paths = sample
            batch_size, negative_sample_size = negative_ents.size(0), negative_ents.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_ents[:, 1]
            ).unsqueeze(1)

            relation = self.comp(paths)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=negative_ents.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif batch_type == BatchType.TAIL_BATCH:
            positive_ents, negative_ents, paths = sample
            batch_size, negative_sample_size = negative_ents.size(0), negative_ents.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_ents[:, 0]
            ).unsqueeze(1)

            relation = self.comp(paths)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=negative_ents.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type), (head, tail)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_ents, negative_ents, paths, subsampling_weight, batch_type = next(train_iterator)

        positive_ents = positive_ents.cuda()
        negative_ents = negative_ents.cuda()
        paths = paths.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score, _ = model((positive_ents, negative_ents, paths), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score, ent = model((positive_ents, paths))

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                    ent[0].norm(p=args.reg_level) ** args.reg_level +
                    ent[1].norm(p=args.reg_level) ** args.reg_level
            ) / ent[0].shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataset = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=1,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        logs = []
        logs_hop = defaultdict(list)

        step = 0
        total_steps = len(test_dataset)

        with torch.no_grad():
            for positive_ents, negative_ents, paths, neg_size, batch_type in test_dataset:
                # test_batch_size = 1
                positive_ents = positive_ents.cuda()
                negative_ents = negative_ents.cuda()
                paths = paths.cuda()
                neg_size = neg_size.cuda()

                negative_score, _ = model((positive_ents, negative_ents, paths), batch_type=batch_type)
                positive_score, _ = model((positive_ents, paths))

                rank = (positive_score < negative_score).sum(dim=1)
                MQ = 1 - rank.float() / neg_size.squeeze(1).float()

                rank = rank.item()
                d = {
                    'MQ': MQ.item(),
                    'HITS@1': 1.0 if rank < 1 else 0.0,
                    'HITS@3': 1.0 if rank < 3 else 0.0,
                    'HITS@10': 1.0 if rank < 10 else 0.0,
                }

                logs.append(d)
                length = len(paths[0])
                logs_hop[length].append(d)

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        metrics_hop = defaultdict(dict)
        for hop in sorted(logs_hop.keys()):
            for metric in logs_hop[hop][0].keys():
                metrics_hop[hop][metric] = sum([log[metric] for log in logs_hop[hop]]) / len(logs_hop[hop])

        return metrics, metrics_hop

class DCNE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_e = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        self.relation_ie = nn.Parameter(torch.zeros(num_relation, hidden_dim))

        self.pi = 3.14159262358979323846

    def comp(self, path):
        rels_emb = self.relation_embedding[path]
        rels_e = self.relation_e[path]
        rels_ie = self.relation_ie[path]
        assert rels_emb.shape == torch.Size([path.shape[0], path.shape[1], self.relation_embedding.shape[1]])
        return rels_emb, rels_e, rels_ie

    def calc(self, head, rel, rel_e, rel_ie):
        head_e, head_ie = torch.chunk(head, 2, dim=2)

        phase_relation = rel / (self.embedding_range.item() / self.pi)

        relation_one = torch.cos(phase_relation)
        relation_two = torch.sin(phase_relation)
        relation_three = rel_e
        relation_four = rel_ie

        result_e = 2 * relation_one * relation_four + 2 * relation_two * relation_three + \
                   (relation_one ** 2 - relation_two ** 2) * head_e - 2 * relation_one * relation_two * head_ie
        result_ie = -2 * relation_one * relation_three + 2 * relation_two * relation_four + \
                    (relation_one ** 2 - relation_two ** 2) * head_ie + 2 * relation_one * relation_two * head_e

        return torch.cat([result_e, result_ie], dim=2)

    def func(self, head, rels, tail, batch_type):
        rel_emb_path, rels_e_path, rels_ie_path = rels
        rel_embs = torch.chunk(rel_emb_path, rel_emb_path.shape[1], dim=1)
        rels_e_s = torch.chunk(rels_e_path, rels_e_path.shape[1], dim=1)
        rels_ie_s = torch.chunk(rels_ie_path, rels_ie_path.shape[1], dim=1)

        for rel, rel_e, rel_ie in zip(rel_embs, rels_e_s, rels_ie_s):
            new_head = self.calc(head, rel, rel_e, rel_ie)
            head = new_head

        head_e, head_ie = torch.chunk(head, 2, dim=2)
        tail_e, tail_ie = torch.chunk(tail, 2, dim=2)

        score_e = head_e - tail_e
        score_ie = head_ie - tail_ie

        score = torch.stack([score_e, score_ie], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)

        return score
