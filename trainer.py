import math
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from eval import eval_qa
from iterator import read_squad_examples, read_level_file, set_level_in_examples, sort_features_by_level, \
    convert_examples_to_features
from model import DomainQA, DomainDiscriminator
from utils import eta, progress_bar


def get_opt(param_optimizer, num_train_optimization_steps, args):
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return BertAdam(optimizer_grouped_parameters,
                    lr=args.lr,
                    warmup=args.warmup_proportion,
                    t_total=num_train_optimization_steps)


def make_weights_for_balanced_classes(classes, nclasses):
    count = [0] * nclasses
    for c in classes:
        count[c] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(classes)
    for idx, val in enumerate(classes):
        weight[idx] = weight_per_class[val]
    return weight


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.set_random_seed(random_seed=args.random_seed)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                                       do_lower_case=args.do_lower_case)
        if args.debug:
            print("debugging mode on.")
        self.features_lst = self.get_features(self.args.train_folder, self.args.debug)

    def make_model_env(self, gpu, ngpus_per_node):
        if gpu is not None:
            self.args.gpu = self.args.devices[gpu]

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = BertForQuestionAnswering.from_pretrained(self.args.bert_model)
        if self.args.load_model is not None:
            print("loading model from ", self.args.load_model)
            # self.model.load_state_dict(torch.load(self.args.load_model))
            self.model.load_state_dict(torch.load(self.args.load_model, map_location=lambda storage, loc: storage))

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) \
            * self.args.epochs * len(self.features_lst)

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        self.optimizer = get_opt(list(self.model.named_parameters()), num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model.cuda()
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def make_run_env(self):
        if self.args.distributed:
            # distributing dev file evaluation task
            self.dev_files = []
            gpu_num = len(self.args.devices)
            files = os.listdir(self.args.dev_folder)
            for i in range(len(files)):
                if i % gpu_num == self.args.rank:
                    self.dev_files.append(files[i])

            print("GPU {}".format(self.args.gpu), self.dev_files)
        else:
            self.dev_files = os.listdir(self.args.dev_folder)
            print(self.dev_files)

    def get_features(self, train_folder, debug=False):
        level_folder = self.args.level_folder
        pickled_folder = self.args.pickled_folder \
            + "_{}_{}".format(self.args.bert_model, str(self.args.skip_no_ans))

        features_lst = []

        files = [f for f in os.listdir(train_folder) if f.endswith(".gz")]
        print("the number of data-set:{}".format(len(files)))
        for file in files:
            data_name = file.split(".")[0]
            # Check whether pkl file already exists
            pickle_file_name = data_name + '.pkl'
            pickle_file_path = os.path.join(pickled_folder, pickle_file_name)
            if os.path.exists(pickle_file_path):
                with open(pickle_file_path, 'rb') as pkl_f:
                    print("Loading {} file as pkl...".format(data_name))
                    features_lst.append(pickle.load(pkl_f))
            else:
                level_name = data_name + ".tsv"
                print("processing {} file".format(data_name))
                level_path = os.path.join(level_folder, level_name)
                file_path = os.path.join(train_folder, file)

                train_examples = read_squad_examples(file_path, debug=debug)
                # read level file and set level
                levels = read_level_file(level_path, sep='\t')
                train_examples = set_level_in_examples(train_examples, levels)

                train_features = convert_examples_to_features(
                    examples=train_examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.args.max_seq_length,
                    max_query_length=self.args.max_query_length,
                    doc_stride=self.args.doc_stride,
                    is_training=True,
                    skip_no_ans=self.args.skip_no_ans
                )
                train_features = sort_features_by_level(train_features, desc=False)

                features_lst.append(train_features)

                # Save feature lst as pickle (For reuse & fast loading)
                if not debug and self.args.rank == 0:
                    with open(pickle_file_path, 'wb') as pkl_f:
                        print("Saving {} file as pkl...".format(data_name))
                        pickle.dump(train_features, pkl_f)

        return features_lst

    def get_iter(self, features_lst, level, args):
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_start_positions = []
        all_end_positions = []
        all_labels = []

        for i, train_features in enumerate(features_lst):
            all_input_ids.append(torch.tensor([f.input_ids for f in train_features], dtype=torch.long))
            all_input_mask.append(torch.tensor([f.input_mask for f in train_features], dtype=torch.long))
            all_segment_ids.append(torch.tensor([f.segment_ids for f in train_features], dtype=torch.long))

            start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

            all_start_positions.append(start_positions)
            all_end_positions.append(end_positions)
            all_labels.append(i * torch.ones_like(start_positions))

        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_input_mask = torch.cat(all_input_mask, dim=0)
        all_segment_ids = torch.cat(all_segment_ids, dim=0)
        all_start_positions = torch.cat(all_start_positions, dim=0)
        all_end_positions = torch.cat(all_end_positions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_labels)
        if args.distributed:
            train_sampler = DistributedSampler(train_data)
            dataloader = DataLoader(train_data, num_workers=args.workers, pin_memory=True,
                                    sampler=train_sampler, batch_size=args.batch_size)
        else:
            # train_sampler = RandomSampler(train_data)
            # dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

            weights = make_weights_for_balanced_classes(all_labels.detach().cpu().numpy().tolist(), 6)
            weights = torch.DoubleTensor(weights)
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

            dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=None,
                                                     sampler=train_sampler, num_workers=args.workers, worker_init_fn=self.set_random_seed(self.args.random_seed), pin_memory=True, drop_last=True)

        return dataloader, train_sampler

    def save_model(self, epoch, loss):
        loss = round(loss, 3)

        save_file = os.path.join(self.args.save_dir, "base_model_{}_{:.3f}".format(epoch, loss))
        save_file_config = os.path.join(self.args.save_dir, "base_config_{}_{:.3f}".format(epoch, loss))

        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_file)
        model_to_save.config.to_json_file(save_file_config)

    def train(self):
        step = 1
        avg_loss = 0
        global_step = 1
        level = 1.0
        iter_lst = [self.get_iter(self.features_lst, level, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            self.model.train()
            start = time.time()
            batch_step = 1
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)

                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions, _ = batch

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)

                    loss = self.model(input_ids, seg_ids, input_mask, start_positions, end_positions)
                    loss = loss.mean()
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()

                    avg_loss = self.cal_running_avg_loss(loss.item() * self.args.gradient_accumulation_steps, avg_loss)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if epoch != 0 and i % 2000 == 0:
                        result_dict = self.evaluate_model(i)
                        for dev_file, f1 in result_dict.items():
                            print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

                    global_step += 1
                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches),
                                avg_loss)
                    print(msg, end="\r")

            print("{} epoch: {}, final loss: {:.4f}".format(self.args.gpu, epoch, avg_loss))

            # save model
            if self.args.rank == 0:
                self.save_model(epoch, avg_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def evaluate_model(self, epoch):
        # model = self.model.module if hasattr(self.model, "module") else self.model
        # result directory
        result_file = os.path.join(self.args.result_dir, "dev_eval_{}.txt".format(epoch))
        fw = open(result_file, "a")
        result_dict = dict()
        for dev_file in self.dev_files:
            file_name = dev_file.split(".")[0]
            prediction_file = os.path.join(self.args.result_dir, "epoch_{}_{}_.json".format(epoch, file_name))
            file_path = os.path.join(self.args.dev_folder, dev_file)
            metrics = eval_qa(self.model, file_path, prediction_file, args=self.args, tokenizer=self.tokenizer, batch_size=self.args.batch_size
                              )
            f1 = metrics["f1"]
            fw.write("{} : {}\n".format(file_name, f1))
            result_dict[dev_file] = f1
        fw.close()

        return result_dict

    @staticmethod
    def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss

    def set_random_seed(self, random_seed):
        if random_seed is not None:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.set_num_threads(1)
            cudnn.benchmark = False
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')


class AdvTrainer(BaseTrainer):
    def __init__(self, args):
        super(AdvTrainer, self).__init__(args)

    def make_model_env(self, gpu, ngpus_per_node):
        if self.args.distributed:
            self.args.gpu = self.args.devices[gpu]
        else:
            self.args.gpu = 0

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = DomainQA(self.args.bert_model, self.args.num_classes,
                              self.args.hidden_size, self.args.num_layers,
                              self.args.dropout, self.args.dis_lambda,
                              self.args.concat, self.args.anneal,
                              self.args.qa_path, self.args.dis_path)
        if self.args.load_model is not None:
            print("loading model from ", self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model, map_location="cpu"))

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) \
            * self.args.epochs * len(self.features_lst)

        qa_params = list(self.model.bert.named_parameters()) \
            + list(self.model.qa_outputs.named_parameters())
        dis_params = list(self.model.discriminator.named_parameters())
        self.qa_optimizer = get_opt(qa_params, num_train_optimization_steps, self.args)
        self.dis_optimizer = get_opt(dis_params, num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model.cuda()
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def train(self):
        step = 1
        avg_qa_loss = 0
        avg_dis_loss = 0
        iter_lst = [self.get_iter(self.features_lst, 1.0, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            start = time.time()
            self.model.train()
            batch_step = 1
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)

                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions, labels = batch

                    if input_ids.size(0) % 4 != 0:
                        continue

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)

                    qa_loss = self.model(input_ids, seg_ids, input_mask,
                                         start_positions, end_positions, labels,
                                         dtype="qa",
                                         global_step=step)
                    qa_loss = qa_loss.mean()
                    qa_loss.backward()

                    # update qa model
                    avg_qa_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
                    self.qa_optimizer.step()
                    self.qa_optimizer.zero_grad()

                    # update discriminator
                    dis_loss = self.model(input_ids, seg_ids, input_mask,
                                          start_positions, end_positions, labels, dtype="dis",
                                          global_step=step)
                    dis_loss = dis_loss.mean()
                    dis_loss.backward()
                    avg_dis_loss = self.cal_running_avg_loss(dis_loss.item(), avg_dis_loss)
                    self.dis_optimizer.step()
                    self.dis_optimizer.zero_grad()
                    step += 1
                    if epoch != 0 and i % 2000 == 0:
                        result_dict = self.evaluate_model(i)
                        for dev_file, f1 in result_dict.items():
                            print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - QA loss: {:.4f}, DIS loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches),
                                avg_qa_loss, avg_dis_loss)
                    print(msg, end="\r")

            print("{} epoch: {}, final loss: {:.4f}, final dis loss: {:.4f}"
                  .format(self.args.gpu, epoch, avg_qa_loss, avg_dis_loss))

            # save model
            if not self.args.distributed or self.args.rank == 0:
                self.save_model(epoch, avg_qa_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def save_model(self, epoch, loss):
        loss = round(loss, 3)
        save_file = os.path.join(self.args.save_dir, "adv_{}_{}".format(epoch, loss))
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        else:
            model_to_save = self.model.feat_ext
        state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_file)


class PreTrainer(BaseTrainer):
    def __init__(self, args):
        super(PreTrainer, self).__init__(args)

    def make_model_env(self, gpu, ngpus_per_node):
        if self.args.distributed:
            self.args.gpu = self.args.devices[gpu]
        else:
            self.args.gpu = 0

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = DomainDiscriminator(self.args.num_classes,
                                         self.args.hidden_size,
                                         self.args.num_layers,
                                         self.args.dropout)
        self.qa_model = BertForQuestionAnswering.from_pretrained(self.args.bert_model)
        state_dict = torch.load(self.args.qa_path, map_location="cpu")
        self.qa_model.load_state_dict(state_dict)
        self.qa_model.requires_grad = False

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) \
            * self.args.epochs * len(self.features_lst)

        params = list(self.model.named_parameters())
        self.optimizer = get_opt(params, num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
                self.qa_model = DistributedDataParallel(self.qa_model, device_ids=[self.args.gpu],
                                                        find_unused_parameters=True)
            else:
                self.model.cuda()
                self.qa_model.cuda()
                self.model = DataParallel(self.model, device_ids=self.args.devices)
                self.qa_model = DataParallel(self.qa_model, device_ids=self.args.devices)
        cudnn.benchmark = True

    def train(self):
        step = 1
        avg_dis_loss = 0
        iter_lst = [self.get_iter(self.features_lst, 1.0, self.args)]
        num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        criterion = nn.NLLLoss()
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            start = time.time()
            self.model.train()
            batch_step = 1
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)

                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions, labels = batch

                    if input_ids.size(0) % 4 != 0:
                        continue

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    labels = labels.cuda()
                    with torch.no_grad():
                        sequence_output, _ = self.qa_model.module.bert(input_ids, seg_ids, input_mask,
                                                                       output_all_encoded_layers=False)

                        hidden = sequence_output[:, 0]
                    log_prob = self.model(hidden.detach())
                    loss = criterion(log_prob, labels)
                    loss = loss.mean()
                    loss.backward()

                    avg_dis_loss = self.cal_running_avg_loss(loss.item(), avg_dis_loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1

                    if step % 5000 == 0:
                        if not self.args.distributed or self.args.rank == 0:
                            self.save_model(i, avg_dis_loss)
                    msg = "{}/{} {} - ETA : {} -  DIS loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches), avg_dis_loss)
                    batch_step += 1
                    print(msg, end="\r")

            print("{} epoch: {}, final dis loss: {:.4f}"
                  .format(self.args.gpu, epoch, avg_dis_loss))

    def save_model(self, epoch, loss):
        loss = round(loss, 3)
        save_file = os.path.join(self.args.save_dir, "dis_{}_{}".format(epoch, loss))
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        else:
            model_to_save = self.model.feat_ext
        state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_file)
