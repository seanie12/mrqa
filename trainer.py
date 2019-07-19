import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from eval import eval_qa
from generator.iterator import read_squad_examples, \
    read_level_file, set_level_in_examples, sort_features_by_level, convert_examples_to_features

from model import DGLearner, MetaLearner
from utils import eta, progress_bar
import random


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


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.set_random_seed(random_seed=args.random_seed)

        self.save_dir = os.path.join("./save",
                                     "{}_{}".format("meta" if self.args.meta else "base", time.strftime("%m%d%H%M")))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.result_dir = os.path.join("./result",
                                       "{}_{}".format("meta" if self.args.meta else "base", time.strftime("%m%d%H%M")))

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
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
        if not os.path.exists(self.save_dir) and self.args.rank == 0:
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir) and self.args.rank == 0:
            os.makedirs(self.result_dir)

        # distributing dev file evaluation task
        self.dev_files = []
        gpu_num = len(self.args.devices)
        files = os.listdir(self.args.dev_folder)
        for i in range(len(files)):
            if i % gpu_num == self.args.rank:
                self.dev_files.append(files[i])

        print("GPU {}".format(self.args.gpu), self.dev_files)

    def get_features(self, train_folder, debug=False):
        level_folder = self.args.level_folder
        pickled_folder = self.args.pickled_folder

        if not os.path.exists(pickled_folder):
            os.mkdir(pickled_folder)

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
                    is_training=True
                )
                train_features = sort_features_by_level(train_features, desc=False)

                features_lst.append(train_features)

                # Save feature lst as pickle (For reuse & fast loading)
                if not debug and self.args.rank == 0:
                    with open(pickle_file_path, 'wb') as pkl_f:
                        print("Saving {} file as pkl...".format(data_name))
                        pickle.dump(train_features, pkl_f)

        return features_lst

    @staticmethod
    def get_iter(features_lst, level, args):
        iter_lst = []
        for train_features in features_lst:
            if level is not None:
                num_data = int(len(train_features) * level) - 1
            else:
                num_data = len(train_features)
            train_features = train_features[:num_data]

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask,
                                       all_segment_ids, all_start_positions, all_end_positions)

            if args.distributed:
                train_sampler = DistributedSampler(train_data)
                dataloader = DataLoader(train_data, num_workers=args.workers, pin_memory=True,
                                        sampler=train_sampler, batch_size=args.batch_size)
            else:
                train_sampler = RandomSampler(train_data)
                dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

            iter_lst.append((dataloader, train_sampler))
        random.shuffle(iter_lst)

        return iter_lst

    def save_model(self, epoch, loss):
        loss = round(loss, 3)

        save_file = os.path.join(self.save_dir, "base_model_{}_{:.3f}".format(epoch, loss))
        save_file_config = os.path.join(self.save_dir, "base_config_{}_{:.3f}".format(epoch, loss))

        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_file)
        model_to_save.config.to_json_file(save_file_config)

    def train(self):
        step = 1
        avg_loss = 0
        global_step = 1
        update_iter = False
        
        level = 1.0
        iter_lst = self.get_iter(self.features_lst, level, self.args)
            num_batches = sum([len(iterator[0]) for iterator in iter_lst])
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            
            start = time.time()
            batch_step = 1
            for data_loader, sampler in iter_lst:
                if self.args.distributed:
                    sampler.set_epoch(epoch)
                
                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions = batch

                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)

                    loss = self.model(input_ids, seg_ids, input_mask, start_positions, end_positions)
                    # loss = loss.mean()
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()

                    avg_loss = self.cal_running_avg_loss(loss.item() * self.args.gradient_accumulation_steps, avg_loss)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    global_step += 1
                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches),
                                avg_loss)
                    print(msg, end="\r")
                   

            print("{} epoch: {}, final loss: {:.4f}".format(self.args.gpu, epoch, avg_loss))
            del iter_lst

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
        result_dir = os.path.join(self.result_dir, "base_{}".format(epoch))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_file = os.path.join(result_dir, "dev_eval.txt")
        # fw = open(result_file, "w")
        fw = open(result_file, "a")
        result_dict = dict()
        for dev_file in self.dev_files:
            file_name = dev_file.split(".")[0]
            prediction_file = os.path.join(result_dir, "epoch_{}_{}_.json".format(epoch, file_name))
            file_path = os.path.join(self.args.dev_folder, dev_file)
            metrics = eval_qa(self.model, file_path
                              , prediction_file, args=self.args
                              , tokenizer=self.tokenizer
                              , batch_size=self.args.batch_size
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

    def set_random_seed(self, random_seed=2019):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)


class DomainTrainer(BaseTrainer):
    def __init__(self, args):
        super(DomainTrainer, self).__init__(args)

    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = DGLearner(self.args.bert_config_file)
        if self.args.load_model is not None:
            print("loading model from ", self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) \
                                       * self.args.epochs * len(self.features_lst)

        self.theta_optimizer = get_opt(list(self.model.feat_ext.named_parameters()), num_train_optimization_steps,
                                       self.args)
        self.phi_optimizer = get_opt(list(self.model.classifier.named_parameters()), num_train_optimization_steps,
                                     self.args)
        self.omega_optimizer = get_opt(list(self.model.critic.named_parameters()), num_train_optimization_steps,
                                       self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def train(self):
        step = 1
        avg_meta_loss = 0
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            # get data below certain level
            levels = [0.8, 0.85, 0.9, 0.95, 1.0]
            if self.args.curriculum:
                idx = min(epoch, len(levels) - 1)
                level = levels[idx]
            else:
                level = levels[len(levels) - 1]
            # shuffle iterators
            iter_lst = self.get_iter(self.features_lst, level, self.args)
            random.shuffle(iter_lst)
            # half of iterators are for meta train and the others for meta test
            num_train = len(iter_lst) // 2
            meta_train_iters = iter_lst[:num_train]
            meta_test_iters = iter_lst[num_train:]

            assert len(meta_train_iters) == len(meta_test_iters)

            num_batches = sum([min(len(train_iter), len(test_iter))
                               for train_iter, test_iter in
                               zip([t[0] for t in meta_train_iters], [t[0] for t in meta_test_iters])])
            batch_step = 1
            start = time.time()
            for idx in range(len(meta_test_iters)):
                # select domain for meta train and meta test
                meta_train_iter = meta_train_iters[idx][0]
                meta_test_iter = meta_test_iters[idx][0]
                meta_train_sampler = meta_train_iters[idx][1]
                meta_test_sampler = meta_test_iters[idx][1]

                if self.args.distributed:
                    meta_train_sampler.set_epoch(epoch)
                    meta_test_sampler.set_epoch(epoch)

                for j, (train_batch, test_batch) in enumerate(zip(meta_train_iter, meta_test_iter), start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions = train_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    train_batch = (input_ids, input_mask, seg_ids, start_positions, end_positions)
                    # get features

                    input_ids, input_mask, seg_ids, start_positions, end_positions = test_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)

                    test_batch = (input_ids, input_mask, seg_ids, start_positions, end_positions)
                    loss_held_out = self.model(train_batch, test_batch, self.args.lr,
                                               self.theta_optimizer, self.phi_optimizer)

                    loss_held_out = loss_held_out.mean()
                    # update feature extractor and classifier
                    self.theta_optimizer.step()
                    self.phi_optimizer.step()
                    # update critic
                    self.omega_optimizer.zero_grad()

                    loss_held_out.backward()
                    self.omega_optimizer.step()
                    # torch.cuda.empty_cache()

                    avg_meta_loss = self.cal_running_avg_loss(loss_held_out.item(), avg_meta_loss)

                    step += 1
                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - meta_loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches), avg_meta_loss)
                    print(msg, end="\r")

            print("{} epoch: {}, final loss: {:.4f}".format(self.args.gpu, epoch, avg_meta_loss))
            del iter_lst

            # save model
            if self.args.rank == 0:
                self.save_model(epoch, avg_meta_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def save_model(self, epoch, loss):
        loss = round(loss, 3)
        save_file = os.path.join(self.save_dir, "meta_{}_{}".format(epoch, loss))
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        else:
            model_to_save = self.model.feat_ext
        state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_file)


class MetaTrainer(BaseTrainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)

    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        self.model = MetaLearner(self.args.bert_config_file, self.args.meta_lambda)
        if self.args.load_model is not None:
            print("loading model from ", self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))

        if self.args.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / self.args.batch_size) \
                                       * self.args.epochs * len(self.features_lst)

        self.optimizer = get_opt(list(self.model.named_parameters()),
                                 num_train_optimization_steps, self.args)

        if self.args.use_cuda:
            if self.args.distributed:
                torch.cuda.set_device(self.args.gpu)
                self.model.cuda(self.args.gpu)
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                     find_unused_parameters=True)
            else:
                self.model = DataParallel(self.model, device_ids=self.args.devices)

        cudnn.benchmark = True

    def train(self):
        step = 1
        avg_loss = 0
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            # get data below certain level
            levels = [0.8, 0.85, 0.9, 0.95, 1.0]
            if self.args.curriculum:
                idx = min(epoch, len(levels) - 1)
                level = levels[idx]
            else:
                level = levels[len(levels) - 1]
            # shuffle iterators
            iter_lst = self.get_iter(self.features_lst, level, self.args)
            random.shuffle(iter_lst)
            # half of iterators are for meta train and the others for meta test
            num_train = len(iter_lst) // 2
            meta_train_iters = iter_lst[:num_train]
            meta_test_iters = iter_lst[num_train:]

            assert len(meta_train_iters) == len(meta_test_iters)

            num_batches = sum([min(len(train_iter), len(test_iter))
                               for train_iter, test_iter in
                               zip([t[0] for t in meta_train_iters], [t[0] for t in meta_test_iters])])
            batch_step = 1
            start = time.time()
            for idx in range(len(meta_test_iters)):
                # select domain for meta train and meta test
                meta_train_iter = meta_train_iters[idx][0]
                meta_test_iter = meta_test_iters[idx][0]
                meta_train_sampler = meta_train_iters[idx][1]
                meta_test_sampler = meta_test_iters[idx][1]

                if self.args.distributed:
                    meta_train_sampler.set_epoch(epoch)
                    meta_test_sampler.set_epoch(epoch)

                for j, (train_batch, test_batch) in enumerate(zip(meta_train_iter, meta_test_iter), start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions = train_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    train_batch = (input_ids, input_mask, seg_ids, start_positions, end_positions)
                    # get features

                    input_ids, input_mask, seg_ids, start_positions, end_positions = test_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    input_mask = input_mask[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    seg_ids = seg_ids[:, :max_len].clone().cuda(self.args.gpu, non_blocking=True)
                    start_positions = start_positions.clone().cuda(self.args.gpu, non_blocking=True)
                    end_positions = end_positions.clone().cuda(self.args.gpu, non_blocking=True)

                    test_batch = (input_ids, input_mask, seg_ids, start_positions, end_positions)
                    loss = self.model(train_batch, test_batch, self.args.lr)

                    loss = loss.mean()
                    # update feature extractor and classifier
                    self.optimizer.zero_grad()

                    loss.backward()
                    self.optimizer.step()

                    avg_loss = self.cal_running_avg_loss(loss.item(), avg_loss)

                    step += 1
                    batch_step += 1
                    msg = "{}/{} {} - ETA : {} - loss: {:.4f}" \
                        .format(batch_step, num_batches, progress_bar(batch_step, num_batches),
                                eta(start, batch_step, num_batches), loss)
                    print(msg, end="\r")

            print("{} epoch: {}, final loss: {:.4f}".format(self.args.gpu, epoch, loss))
            del iter_lst

            # save model
            if self.args.rank == 0:
                self.save_model(epoch, avg_loss)

            if self.args.do_valid:
                result_dict = self.evaluate_model(epoch)
                for dev_file, f1 in result_dict.items():
                    print("GPU {} evaluated {}: {:.2f}".format(self.args.gpu, dev_file, f1), end="\n")

    def save_model(self, epoch, loss):
        loss = round(loss, 3)
        save_file = os.path.join(self.save_dir, "meta_{}_{}".format(epoch, loss))
        if hasattr(self.model, "module"):
            model_to_save = self.model.module
        else:
            model_to_save = self.model.feat_ext
        state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_file)
