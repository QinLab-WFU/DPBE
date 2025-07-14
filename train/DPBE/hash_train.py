# DHaPH
# paper [Deep Hierarchy-aware Proxy Hashing with Self-paced Learning for Cross-modal Retrieval, TKDE 2024]
# (https://ieeexplore.ieee.org/document/10530441)

import os

import numpy as np
import torch
import time

from .stochman.stochman.laplace import DiagLaplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from .stochman.stochman import ContrastiveHessianCalculator, MSEHessianCalculator
from .stochman.stochman.utils import convert_to_stochman
from torch.nn import functional as F

from model.DPBE import MDPBE
from train.base import TrainBase
from model.base.optimization import BertAdam
from .PVSEloss import PVSELoss

from .get_args import get_args
from .triplet_miner import TripletMinner
from ..DDWSH.MDW import Contrastive
from ..DDWSH.MDWit import Contrastit


class DPBETrainer(TrainBase):

    def __init__(self, rank):
        args = get_args()
        args.rank = rank
        super(DPBETrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = MDPBE(use_lam=self.args.use_lam, outputDim=self.args.output_dim,
                           num_classes=self.args.nclass, clipPath=self.args.clip_path, writer=self.writer,
                           logger=self.logger, is_train=True).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()


        to_optim = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_pre.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_pre.parameters(), 'lr': self.args.lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
        ]

        if self.args.loss != "acm":
            self.mdwi = Contrastive(miner=self.args.miner, n_classes=self.nclass).to(self.rank)
            self.mdwt = Contrastive(miner=self.args.miner, n_classes=self.nclass).to(self.rank)
            self.mdwit = Contrastit(miner=self.args.miner, n_classes=self.nclass).to(self.rank)
            mdw_optim = [
                {'params': self.mdwi.parameters(), 'lr': self.args.lr},
                {'params': self.mdwt.parameters(), 'lr': self.args.lr},
                {'params': self.mdwit.parameters(), 'lr': self.args.lr}
            ]
            to_optim.extend(mdw_optim)

        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine',b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.criterion = PVSELoss(self.args, self.rank)

        if self.args.use_lam:
            self.model.image_hash = convert_to_stochman(self.model.image_hash)
            self.model.text_hash = convert_to_stochman(self.model.text_hash)

            if self.args.loss != "acm":
                self.hessian_calculator = ContrastiveHessianCalculator(
                    wrt="weight",
                    shape="diagonal",
                    speed="half",
                    method="fix",
                )
            else:
                self.hessian_calculator = MSEHessianCalculator(  #
                    wrt="weight",
                    shape="diagonal",
                    speed="half",
                    method="",
                )

            self.laplace = DiagLaplace()
            hessian_i = self.laplace.init_hessian(self.args.train_num, self.model.image_hash, self.rank)
            hessian_t = self.laplace.init_hessian(self.args.train_num, self.model.text_hash, self.rank)
            self.scale_hs = self.args.train_num ** 2
            self.model.register_buffer("hessian_i", hessian_i)
            self.model.register_buffer("hessian_t", hessian_t)
            # TODO: notice the special hessian miner
            self.hessian_miner = TripletMinner().to(self.rank)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            embed_i, embed_t, pre_i, pre_t = self.model(image, text)

            # get mean and std of posterior
            sigma_q_i = self.laplace.posterior_scale(torch.relu(self.model.hessian_i))
            mu_q_i = parameters_to_vector(self.model.image_hash.parameters()).unsqueeze(1)
            sigma_q_t = self.laplace.posterior_scale(torch.relu(self.model.hessian_t))
            mu_q_t = parameters_to_vector(self.model.text_hash.parameters()).unsqueeze(1)

            samples_i = self.laplace.sample(mu_q_i, sigma_q_i, self.args.train_n_samples)
            samples_t = self.laplace.sample(mu_q_t, sigma_q_t, self.args.train_n_samples)

            loss = 0
            hessian_i = 0
            hessian_t = 0
            for sample_i, sample_t in zip(samples_i, samples_t):

                # replace the network parameters with the sampled parameters
                vector_to_parameters(sample_i, self.model.image_hash.parameters())
                vector_to_parameters(sample_t, self.model.text_hash.parameters())

                z_i = torch.sign(self.model.image_hash(embed_i))
                z_t = torch.sign(self.model.text_hash(embed_t))

                if self.args.loss == "acm":

                    # ensure that we are on unit sphere
                    z_i = z_i / z_i.norm(dim=-1, keepdim=True)
                    z_t = z_t / z_t.norm(dim=-1, keepdim=True)\

                    criterion = torch.nn.MSELoss().to(self.rank)
                    _, aff_norm, aff_label = self.affinity_tag_multi(label.cpu().numpy(), label.cpu().numpy())
                    aff_label = torch.Tensor(aff_label).to(self.rank)
                    H_i, H_t = F.normalize(z_i), F.normalize(z_t)

                    clf_loss = criterion(torch.sigmoid(pre_i), label) + criterion(torch.sigmoid(pre_t), label)
                    clf_loss += criterion(torch.sigmoid(torch.stack((pre_i, pre_t), dim=0).mean(dim=0)), label)
                    similarity_loss = criterion(H_i.mm(H_i.t()), aff_label) + criterion(H_t.mm(H_t.t()), aff_label)
                    similarity_loss += criterion(H_i.mm(H_t.t()), aff_label)

                    loss += clf_loss + similarity_loss
                else:
                    i, n = self.mdwi(z_i, label)
                    t, m = self.mdwt(z_t, label)
                    it, mm = self.mdwit(z_i, z_t, label)
                    similarity_loss = i + t + it
                    i, n = self.mdwi(pre_i, label)
                    t, m = self.mdwt(pre_t, label)
                    it, mm = self.mdwit(pre_i, pre_t, label)
                    clf_loss = i + t + it
                    loss += clf_loss + similarity_loss
                    # loss += similarity_loss

                with torch.inference_mode():

                    # hessian_indices_tuple = self.hessian_miner(z, y)
                    hessian_indices_tuple = self.hessian_miner(label)

                    # randomly choose 5000 pairs if more than 5000 pairs available.
                    # TODO: decide what to do. What pairs should we use to compute the hessian over?
                    # does it matter? What experiments should we run to get a better idea?
                    n_triplets = len(hessian_indices_tuple[0])
                    if n_triplets > self.args.max_pairs:
                        idx = torch.randperm(hessian_indices_tuple[0].size(0))[: self.args.max_pairs]
                        hessian_indices_tuple = (
                            hessian_indices_tuple[0][idx],
                            hessian_indices_tuple[1][idx],
                            hessian_indices_tuple[2][idx],
                        )

                    h_s_i = self.hessian_calculator.compute_hessian(
                        embed_i.detach(), self.model.image_hash, hessian_indices_tuple
                    )
                    h_s_i = self.laplace.scale(h_s_i, min(n_triplets, self.args.max_pairs), self.scale_hs)
                    hessian_i += h_s_i

                    h_s_t = self.hessian_calculator.compute_hessian(
                        embed_t.detach(), self.model.text_hash, hessian_indices_tuple
                    )
                    h_s_t = self.laplace.scale(h_s_t, min(n_triplets, self.args.max_pairs), self.scale_hs)
                    hessian_t += h_s_t

            # reset the network parameters with the mean parameter (MAP estimate parameters)
            vector_to_parameters(mu_q_i, self.model.image_hash.parameters())
            vector_to_parameters(mu_q_t, self.model.text_hash.parameters())
            loss = loss / self.args.train_n_samples
            hessian_i = hessian_i / self.args.train_n_samples
            hessian_t = hessian_t / self.args.train_n_samples

            self.model.hessian_i = self.args.hessian_memory_factor * self.model.hessian_i + torch.relu(hessian_i)
            self.model.hessian_t = self.args.hessian_memory_factor * self.model.hessian_t + torch.relu(hessian_t)

            # loss_ce = torch.nn.CrossEntropyLoss()
            # ii = embed_i @ embed_i.T
            # tt = embed_t @ embed_t.T
            # it = embed_i @ embed_t.T
            # ll = label @ label.T
            # loss = loss_ce(ii, ll) + loss_ce(tt, ll) + loss_ce(it, ll) + 0.2 * loss_ce(tt, ii)

            all_loss += loss.data
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")

    def sample(self, n_samples):

        # get mean and std of posterior
        mu_q_i = parameters_to_vector(self.model.image_hash.parameters()).unsqueeze(1)
        sigma_q_i = self.laplace.posterior_scale(torch.relu(self.model.hessian_i))
        mu_q_t = parameters_to_vector(self.model.text_hash.parameters()).unsqueeze(1)
        sigma_q_t = self.laplace.posterior_scale(torch.relu(self.model.hessian_t))

        # draw samples
        self.model.nn_weight_samples_i = self.laplace.sample(mu_q_i, sigma_q_i, n_samples)
        self.model.nn_weight_samples_t = self.laplace.sample(mu_q_t, sigma_q_t, n_samples)

    def for_ward(self, image, text):

        image_embed, text_embed, _, _ = self.model(image, text)
        mu_q_i = parameters_to_vector(self.model.image_hash.parameters()).unsqueeze(1)
        mu_q_t = parameters_to_vector(self.model.text_hash.parameters()).unsqueeze(1)

        zs_i, zs_t = [], []
        for i in range(self.args.valid_n_samples):
            # use sample i that was generated in beginning of evaluation
            net_sample_i = self.model.nn_weight_samples_i[i]
            net_sample_t = self.model.nn_weight_samples_t[i]

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample_i, self.model.image_hash.parameters())
            vector_to_parameters(net_sample_t, self.model.text_hash.parameters())

            z_i = self.model.image_hash(image_embed)
            z_t = self.model.text_hash(text_embed)

            # ensure that we are on unit sphere
            z_i = z_i / z_i.norm(dim=-1, keepdim=True)
            z_t = z_t / z_t.norm(dim=-1, keepdim=True)

            zs_i.append(z_i)
            zs_t.append(z_t)

        zs_i = torch.stack(zs_i)
        zs_t = torch.stack(zs_t)

        # compute statistics
        z_mu_i = zs_i.mean(dim=0)
        z_sigma_i = zs_i.std(dim=0)
        z_mu_t = zs_t.mean(dim=0)
        z_sigma_t = zs_t.std(dim=0)

        # put mean parameters back
        vector_to_parameters(mu_q_i, self.model.image_hash.parameters())
        vector_to_parameters(mu_q_t, self.model.text_hash.parameters())

        code = {
            "z_mu_i": z_mu_i,
            "z_sigma_i": z_sigma_i,
            "z_samples_i": zs_i.permute(1, 0, 2),
            "z_mu_t": z_mu_t,
            "z_sigma_t": z_sigma_t,
            "z_samples_t": zs_t.permute(1, 0, 2)
        }
        return code

    def get_code(self, data_loader, length: int):

        self.sample(self.args.valid_n_samples)

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        uncer_codes = []
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()

            code = self.for_ward(image, text)
            image_hash = torch.sign(code['z_mu_i'])
            text_hash = torch.sign(code['z_mu_t'])
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

            if self.args.calculate_uncertainty:
                uncer_code = self.get_uncer_code(code, label)
                uncer_codes.append(uncer_code)

        if self.args.calculate_uncertainty:
            metrics = self.compute_metrics(uncer_code)

        return img_buffer, text_buffer, encoder_time

    def get_uncer_code(self, code, label):
        z_mu_i = code["z_mu_i"]
        z_mu_i = z_mu_i / torch.norm(z_mu_i, dim=-1, keepdim=True)
        z_mu_t = code["z_mu_t"]
        z_mu_t = z_mu_t / torch.norm(z_mu_t, dim=-1, keepdim=True)
        o = {
            "z_mu_i": z_mu_i.cpu(),
            "z_mu_t": z_mu_t.cpu(),
            "label": label.cpu(),
            "z_sigma_i": code["z_sigma_i"].cpu(),
            "z_sigma_t": code["z_sigma_t"].cpu()
        }
        if "z_samples_i" in code:
            z_samples_i = code["z_samples_i"]
            z_samples_i = z_samples_i / torch.norm(z_samples_i, dim=-1, keepdim=True)
            o["z_sigma_i"] = z_samples_i.cpu()
            z_samples_t = code["z_samples_t"]
            z_samples_t = z_samples_t / torch.norm(z_samples_t, dim=-1, keepdim=True)
            o["z_sigma_t"] = z_samples_t.cpu()
        return o

    def compute_metrics(self, outputs):

        z_mu_i = torch.cat([o['z_mu_i'] for o in outputs])
        z_mu_t = torch.cat([o['z_mu_t'] for o in outputs])
        target = torch.cat([o['label'] for o in outputs])
        pidxs = self.get_pos_idx(target)
        z_muQ = z_mu_i + z_mu_t
        z_sigma = torch.cat([o['z_sigma'] for o in outputs])
        o = {
            "z_muQ": z_muQ,
            "z_muDb": None,
            "pidxs": pidxs,
            'target': target,
            "z_sigma": z_sigma
        }


        # id = self.format_outputs(outputs)
        ood = None






    def get_pos_idx(self, target):

        classes = np.unique(target)
        idx = {}
        for c in classes:
            idx[f"{c}"] = {"pos": np.where(target == c)[0]}

        pos_idx = []
        for i in range(len(target)):
            key = f"{target[i].data}"

            pidx = idx[key]["pos"]
            pidx = pidx[pidx != i]  # remove self

            pos_idx.append(pidx)

        return pos_idx

    def zero2eps(self, x):
        x[x == 0] = 1
        return x

    def normalize(self, affinity):
        col_sum = self.zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
        row_sum = self.zero2eps(np.sum(affinity, axis=0))
        out_affnty = affinity / col_sum  # row data sum = 1
        in_affnty = np.transpose(affinity / row_sum)  # col data sum = 1 then transpose
        return in_affnty, out_affnty

    # Check in 2022-1-3
    def affinity_tag_multi(self, tag1: np.ndarray, tag2: np.ndarray):
        '''
        Use label or plabel to create the graph.
        :param tag1:
        :param tag2:
        :return:
        '''
        aff = np.matmul(tag1, tag2.T)
        affinity_matrix = np.float32(aff)
        # affinity_matrix[affinity_matrix > 1] = 1
        affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
        affinity_matrix = 2 * affinity_matrix - 1
        in_aff, out_aff = self.normalize(affinity_matrix)

        return in_aff, out_aff, affinity_matrix
