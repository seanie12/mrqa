import argparse
import os

import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from generator.iterator import Config, read_squad_examples, \
    read_level_file, set_level_in_examples, sort_features_by_level, convert_examples_to_features
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
import math


def get_iter(features_lst, level, batch_size):
    iter_lst = []
    for train_features in features_lst:
        num_data = int(len(train_features) * level) - 1
        train_features = train_features[:num_data]

        features_lst.append(train_features)
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


def main(args):
    config = Config(bert_model=args.bert_model,
                    max_seq_length=args.max_seq_length,
                    max_query_length=args.max_query_length,
                    batch_size=args.batch_size,
                    epochs=args.epochs)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    train_folder = "./data/train"
    level_folder = "./generator/difficulty"
    features_lst = []
    files = os.listdir(train_folder)

    for file in files:
        data_name = file.split(".")[0]
        level_name = data_name + ".tsv"
        level_path = os.path.join(level_folder, level_name)
        file_path = os.path.join(train_folder, file)

        train_examples = read_squad_examples(file_path)
        # read level file and set level
        levels = read_level_file(level_path, sep='\t')
        train_examples = set_level_in_examples(train_examples, levels)

        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            max_query_length=config.max_query_length,
            doc_stride=config.doc_stride
        )
        train_features = sort_features_by_level(train_features, desc=False)
        features_lst.append(train_features)

    device = torch.device("cuda:0")
    model = BertForQuestionAnswering.from_pretrained(config.bert_model)
    model = model.to(device)
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    max_len = max([len(f) for f in features_lst])
    num_train_optimization_steps = math.ceil(max_len / config.batch_size) * config.epochs

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.lr,
                         warmup=config.warmup_proportion,
                         t_total=num_train_optimization_steps)

    for epoch in range(config.epochs):
        levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        assert len(levels) == config.epochs
        level = levels[epoch]
        iter_lst = get_iter(features_lst, level, config.batch_size)
        step = 1
        for data_loader in iter_lst:
            for i, batch in enumerate(data_loader, start=1):
                input_ids, input_mask, seg_ids, start_positions, end_positions = batch
                # remove unnecessary pad token
                seq_len = torch.sum(torch.sign(input_ids), 1)
                max_len = torch.max(seq_len)
                input_ids = input_ids[:, :max_len].to(device)
                input_mask = input_mask[:, :max_len].to(device)
                seg_ids = seg_ids[:, :max_len].to(device)
                start_positions = start_positions.to(device)
                end_positions = end_positions.to(device)

                loss = model(input_ids, seg_ids, input_mask, start_positions, end_positions)
                loss.backward()
                if step % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                if step % 100 == 0:
                    print("Epoch: {}, step :{}, loss :{:.4f}".format(epoch + 1, step, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="max query length")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=5, type=int, help="number of epochs")
    args = parser.parse_args()
    main(args)
