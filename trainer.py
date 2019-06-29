import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from model import FeatureExtractor, Classifier, Critic
from generator.iterator import read_squad_examples, \
    read_level_file, set_level_in_examples, sort_features_by_level, convert_examples_to_features
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
import math
import pickle
import os
import time
import random


class BaseTrainer(object):
    def __init__(self, config):
        self.config = config
        self.save_dir = os.path.join("./save", "base_{}".format(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model,
                                                       do_lower_case=config.do_lower_case)
        print("debugging mode:", config.debug)
        self.features_lst = self.get_features(config.debug)
        self.device = torch.device("cuda:0")
        model = BertForQuestionAnswering.from_pretrained(config.bert_model)
        self.model = model.to(self.device)
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / config.batch_size) \
                                       * config.epochs * len(self.features_lst)

        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=config.lr,
                                  warmup=config.warmup_proportion,
                                  t_total=num_train_optimization_steps)

    def get_features(self, debug=False):
        train_folder = "./data/train"
        level_folder = "./generator/difficulty"
        pickled_folder = './pickled_data'
        if not os.path.exists(pickled_folder):
            os.mkdir(pickled_folder)

        features_lst = []
        files = os.listdir(train_folder)
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
                level_path = os.path.join(level_folder, level_name)
                file_path = os.path.join(train_folder, file)

                train_examples = read_squad_examples(file_path, debug=debug)
                # read level file and set level
                levels = read_level_file(level_path, sep='\t')
                train_examples = set_level_in_examples(train_examples, levels)

                train_features = convert_examples_to_features(
                    examples=train_examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.config.max_seq_length,
                    max_query_length=self.config.max_query_length,
                    doc_stride=self.config.doc_stride
                )
                train_features = sort_features_by_level(train_features, desc=False)
                features_lst.append(train_features)

                # Save feature lst as pickle (For reuse & fast loading)
                if not debug:
                    with open(pickle_file_path, 'wb') as pkl_f:
                        print("Saving {} file as pkl...".format(data_name))
                        pickle.dump(train_features, pkl_f)
        return features_lst

    @staticmethod
    def get_iter(features_lst, level, batch_size):
        iter_lst = []
        for train_features in features_lst:
            num_data = int(len(train_features) * level) - 1
            train_features = train_features[:num_data]

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask,
                                       all_segment_ids, all_start_positions, all_end_positions)
            train_sampler = RandomSampler(train_data)
            dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
            iter_lst.append(dataloader)
        return iter_lst

    def save_model(self, epoch, loss):
        loss = round(loss, 3)

        save_file = os.path.join(self.save_dir, "class_{}_{}".format(epoch, loss))
        state_dict = self.model.state_dict()
        torch.save(state_dict, save_file)

    def train(self):
        step = 1
        avg_loss = 0
        for epoch in range(self.config.epochs):
            levels = [0.3, 0.5, 0.7, 0.9, 1.0]
            idx = min(epoch, len(levels) - 1)
            level = levels[idx]
            iter_lst = self.get_iter(self.features_lst, level, self.config.batch_size)
            for data_loader in iter_lst:
                for i, batch in enumerate(data_loader, start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions = batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].to(self.device)
                    input_mask = input_mask[:, :max_len].to(self.device)
                    seg_ids = seg_ids[:, :max_len].to(self.device)
                    start_positions = start_positions.to(self.device)
                    end_positions = end_positions.to(self.device)

                    loss = self.model(input_ids, seg_ids, input_mask, start_positions, end_positions)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    avg_loss = self.cal_running_avg_loss(loss.item(), avg_loss)
                    if step % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    step += 1
                    if step % 100 == 0:
                        print("Epoch: {}, step :{}, loss :{:.4f}".format(epoch + 1, step, avg_loss))
        # save model
        self.save_model(self.config.epochs, avg_loss)

    @staticmethod
    def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss


class MetaTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config
        self.save_dir = os.path.join("./save", "base_{}".format(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model,
                                                       do_lower_case=config.do_lower_case)
        print("debugging mode:", config.debug)
        self.features_lst = self.get_features(config.debug)
        self.device = torch.device("cuda:0")
        self.f_ext = FeatureExtractor(config.bert_model).to(self.device)
        self.classifier = Classifier(768).to(self.device)
        self.critic = Critic(768).to(self.device)

        param_optimizer = list(self.f_ext.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        max_len = max([len(f) for f in self.features_lst])
        num_train_optimization_steps = math.ceil(max_len / config.batch_size) \
                                       * config.epochs * len(self.features_lst)

        self.theta_optimizer = BertAdam(optimizer_grouped_parameters,
                                        lr=config.lr,
                                        warmup=config.warmup_proportion,
                                        t_total=num_train_optimization_steps)

        param_optimizer = list(self.classifier.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.phi_optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=config.lr,
                                      warmup=config.warmup_proportion,
                                      t_total=num_train_optimization_steps)
        param_optimizer = list(self.critic.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.omega_optimizer = BertAdam(optimizer_grouped_parameters,
                                        lr=config.lr,
                                        warmup=config.warmup_proportion,
                                        t_total=num_train_optimization_steps)

    def train(self):
        step = 1
        avg_loss = 0
        for epoch in range(self.config.epochs):
            levels = [0.3, 0.5, 0.7, 0.9, 1.0]
            idx = min(epoch, len(levels) - 1)
            level = levels[idx]
            iter_lst = self.get_iter(self.features_lst, level, self.config.batch_size)
            random.shuffle(iter_lst)
            num_train = len(iter_lst) // 2
            meta_train_iters = iter_lst[:num_train]
            meta_test_iters = iter_lst[num_train:]
            for i in range(len(meta_test_iters)):
                meta_train_iter = meta_train_iters[i]
                meta_test_iter = meta_test_iters[i]
                for j, (train_batch, test_batch) in enumerate(zip(meta_train_iter, meta_test_iter), start=1):
                    input_ids, input_mask, seg_ids, start_positions, end_positions = train_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].to(self.device)
                    input_mask = input_mask[:, :max_len].to(self.device)
                    seg_ids = seg_ids[:, :max_len].to(self.device)
                    start_positions = start_positions.to(self.device)
                    end_positions = end_positions.to(self.device)

                    features = self.f_ext(input_ids, seg_ids, input_mask)
                    loss_main = self.classifier(features, start_positions, end_positions)
                    loss_main = loss_main / self.config.gradient_accumulation_steps
                    self.theta_optimizer.zero_grad()
                    self.phi_optimizer.zero_grad()
                    loss_main.backward(retain_graph=True)

                    # forward critic
                    loss_dg = self.critic(features)
                    grad_theta = [theta_i.grad for theta_i in self.f_ext.parameters()]
                    theta_updated_old = dict()

                    num_grad = 0
                    for i, (k, v) in enumerate(self.f_ext.state_dict().items()):

                        if grad_theta[num_grad] is None:
                            num_grad += 1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.config.lr * grad_theta[num_grad]
                            num_grad += 1

                    loss_dg.backward(create_graph=True)

                    grad_theta = [theta_i.grad for theta_i in self.f_ext.parameters()]
                    theta_updated_new = {}
                    num_grad = 0
                    for i, (k, v) in enumerate(self.f_ext.state_dict().items()):

                        if grad_theta[num_grad] is None:
                            num_grad += 1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.config.lr * grad_theta[num_grad]
                            num_grad += 1

                    temp_new_feature_extractor_network = FeatureExtractor(self.config.bert_model).to(self.device)
                    self.fix_nn(temp_new_feature_extractor_network, theta_updated_new)
                    temp_new_feature_extractor_network.train()

                    temp_old_feature_extractor_network = FeatureExtractor(self.config.bert_model).to(self.device)
                    temp_old_feature_extractor_network.load_state_dict(theta_updated_old)
                    temp_old_feature_extractor_network.train()
                    input_ids, input_mask, seg_ids, start_positions, end_positions = test_batch
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)
                    input_ids = input_ids[:, :max_len].to(self.device)
                    input_mask = input_mask[:, :max_len].to(self.device)
                    seg_ids = seg_ids[:, :max_len].to(self.device)
                    start_positions = start_positions.to(self.device)
                    end_positions = end_positions.to(self.device)
                    with torch.no_grad():
                        old_features = temp_old_feature_extractor_network(input_ids, seg_ids, input_mask)
                    new_features = temp_new_feature_extractor_network(input_ids, seg_ids, input_mask)

                    loss_main_old = self.classifier(old_features, start_positions, end_positions)
                    loss_main_new = self.classifier(new_features, start_positions, end_positions)

                    reward = loss_main_old - loss_main_new
                    utility = torch.tanh(reward)
                    loss_held_out = - utility.sum()

                    # update feature extractor and classifier
                    self.theta_optimizer.step()
                    self.phi_optimizer.step()
                    # update critic
                    self.omega_optimizer.zero_grad()
                    loss_held_out.backward()
                    self.omega_optimizer.step()
                    torch.cuda.empty_cache()
                    step += 1
                    avg_loss = self.cal_running_avg_loss(loss_main.item(), avg_loss)
                    if step % 100 == 0:
                        print("Epoch: {}, step :{}, loss :{:.4f}".format(epoch + 1, step, avg_loss))
        # save model
        self.save_model(self.config.epochs, avg_loss)

    def save_model(self, epoch, loss):
        loss = round(loss, 3)

        save_file = os.path.join(self.save_dir, "class_{}_{}".format(epoch, loss))
        state_dict = self.classifier.state_dict()
        torch.save(state_dict, save_file)

        save_file = os.path.join(self.save_dir, "feature_{}_{}".format(epoch, loss))
        state_dict = self.f_ext.state_dict()
        torch.save(state_dict, save_file)

    @staticmethod
    def fix_nn(model, theta):
        def k_param_fn(tmp_model, name=None):
            if len(tmp_model._modules) != 0:
                for (k, v) in tmp_model._modules.items():
                    if name is None:
                        k_param_fn(v, name=str(k))
                    else:
                        k_param_fn(v, name=str(name + '.' + k))
            else:
                for (k, v) in tmp_model._parameters.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]

        k_param_fn(model)
        return model
