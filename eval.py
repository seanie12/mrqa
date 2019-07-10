from pytorch_pretrained_bert import BertForQuestionAnswering, BertTokenizer
from generator.iterator import read_squad_examples, convert_examples_to_features, write_predictions, \
    set_level_in_examples
from mrqa_official_eval import evaluate, read_answers
import argparse
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import collections
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--file_path", type=str, required=True, help="data file to evaluate")
    parser.add_argument("--prediction_file", type=str, required=True, help="prediction file")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="max query length")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for inference")

    args = parser.parse_args()

    device = "cuda"
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    eval_examples = read_squad_examples(args.file_path, debug=False)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # In test time, there is no level file and it is not necessary for inference
    for example in eval_examples:
        example.level = 0

    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.config.max_seq_length,
        max_query_length=args.config.max_query_length,
        doc_stride=args.config.doc_stride,
        is_training=False
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=sampler, batch_size=args.batch_size)

    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    model.eval()
    all_results = []
    example_index = -1
    for j, batch in enumerate(eval_loader):
        input_ids, input_mask, seg_ids = batch
        seq_len = torch.sum(torch.sign(input_ids), 1)
        max_len = torch.max(seq_len)
        input_ids = input_ids[:, :max_len]
        input_mask = input_mask[:, :max_len]
        seg_ids = seg_ids[:, :max_len]
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, seg_ids, input_mask)
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

    predictions = write_predictions(eval_examples, eval_features, all_results,
                                    n_best_size=20, max_answer_length=30, do_lower_case=True,
                                    output_prediction_file=args.prediction_file)

    answers = read_answers(args.file_path)
    metrics = evaluate(answers, predictions, args.skip_no_answer)
    print(json.dumps(metrics))
