import collections
import json
from sparse_model.modeling import BertForQuestionAnswering as SparseBert
import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler

from .squad_utils import read_squad_examples, convert_examples_to_features, write_predictions, evaluate

# hyper-param
device = torch.device("cuda")
drop_rate = 0.6
targ_perc = 0.68
batch_size = 8

warmup_proportion = 0.1
max_seq_len = 384
max_query_len = 64
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_file = "./data/train-v1.1.json"
train_examples = read_squad_examples(train_file, is_training=True, debug=True)
train_features = convert_examples_to_features(train_examples,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_len,
                                              max_query_length=max_query_len,
                                              doc_stride=128,
                                              is_training=True)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                           all_start_positions, all_end_positions)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

model = SparseBert.from_pretrained("bert-base-uncased").to(device)

param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = len(train_dataloader) * 5
warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                     t_total=num_train_optimization_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=5e-5,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)
global_step = 1

is_training = True
for epoch in range(1, 6):
    for batch in train_dataloader:
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        seq_len = torch.sum(torch.sign(input_ids), 1)
        max_len = torch.max(seq_len)
        input_ids = input_ids[:, :max_len].to(device)
        input_mask = input_mask[:, :max_len].to(device)
        segment_ids = segment_ids[:, :max_len].to(device)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        loss = model(input_ids, segment_ids, input_mask,
                     drop_rate, targ_perc, is_training, global_step,
                     start_positions, end_positions)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if global_step % 10 == 0:
            print("epoch: {}, step: {}, loss: {:.4f}".format(epoch, global_step, loss.item()), end="\r")
        global_step += 1

# model = SparseBert.from_pretrained("bert-base-uncased")
# state_dict = torch.load("./save/sparse_bert", map_location="cuda")
# model.load_state_dict(state_dict)
# model = model.to(device)
state_dict = model.state_dict()
save_file = "sparse_bert"
torch.save(state_dict, "./save/sparse_bert2")

test_file = "./data/dev-v1.1.json"
eval_examples = read_squad_examples(test_file, is_training=False, debug=False)
eval_features = convert_examples_to_features(eval_examples,
                                             tokenizer=tokenizer,
                                             max_seq_length=384,
                                             max_query_length=64,
                                             doc_stride=128,
                                             is_training=False)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
model.eval()
all_results = []
example_index = -1
for j, batch in enumerate(eval_dataloader):

    with torch.no_grad():
        input_ids, input_mask, segment_ids = batch
        seq_len = torch.sum(torch.sign(input_ids), 1)
        max_len = torch.max(seq_len)
        input_ids = input_ids[:, :max_len].to(device)
        input_mask = input_mask[:, :max_len].to(device)
        segment_ids = segment_ids[:, :max_len].to(device)

        batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask,
                                                     is_training=False,
                                                     targ_perc=targ_perc)
        batch_size = batch_start_logits.size(0)
    for i in range(batch_size):
        example_index += 1
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = eval_features[example_index]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits))

output_prediction_file = "./result/sparse_predictions.json"
output_nbest_file = "./result/sparse_nbest_predictions.json"
output_null_log_odds_file = "./result/sparse_null_odds.json"
write_predictions(eval_examples, eval_features, all_results,
                  n_best_size=20, max_answer_length=30, do_lower_case=True,
                  output_prediction_file=output_prediction_file,
                  output_nbest_file=output_nbest_file,
                  output_null_log_odds_file=output_null_log_odds_file,
                  verbose_logging=False,
                  version_2_with_negative=False,
                  null_score_diff_threshold=0,
                  noq_position=False)

with open(test_file) as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
with open("./result/sparse_predictions.json") as prediction_file:
    predictions = json.load(prediction_file)
print(json.dumps(evaluate(dataset, predictions)))
