import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from generator.iterator import read_squad_examples, \
    read_level_file, set_level_in_examples, sort_features_by_level, convert_examples_to_features
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
import math
import pickle
import os


class BaseTrainer(object):
    def __init__(self, config):
        self.config = config
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
        num_train_optimization_steps = math.ceil(max_len / config.batch_size) * config.epochs

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
        if self.config.debug:
            files = [files[0]]
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

    def train(self):
        step = 1
        for epoch in range(self.config.epochs):
            levels = [0.8, 0.8, 0.9, 0.9, 1.0]
            idx = min(epoch, len(levels)-1)
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
                    if step % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    step += 1
                    if step % 100 == 0:
                        print("Epoch: {}, step :{}, loss :{:.4f}".format(epoch + 1, step, loss))
