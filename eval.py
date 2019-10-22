import collections
import json

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from iterator import read_squad_examples, convert_examples_to_features, write_predictions
from mrqa_official_eval import evaluate, read_answers


def eval_qa(model, file_path, prediction_file, args, tokenizer, batch_size=50):
    eval_examples = read_squad_examples(file_path, debug=False)

    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_query_length=args.max_query_length,
        doc_stride=args.doc_stride,
        is_training=False
    )
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=sampler, batch_size=batch_size)

    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    model.eval()
    all_results = []
    example_index = -1
    for _, batch in enumerate(eval_loader):
        input_ids, input_mask, seg_ids = batch
        seq_len = torch.sum(torch.sign(input_ids), 1)
        max_len = torch.max(seq_len)

        input_ids = input_ids[:, :max_len].clone()
        input_mask = input_mask[:, :max_len].clone()
        seg_ids = seg_ids[:, :max_len].clone()

        if args.use_cuda:
            input_ids = input_ids.cuda(args.gpu, non_blocking=True)
            input_mask = input_mask.cuda(args.gpu, non_blocking=True)
            seg_ids = seg_ids.cuda(args.gpu, non_blocking=True)

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

    preds = write_predictions(eval_examples, eval_features, all_results,
                              n_best_size=20, max_answer_length=30, do_lower_case=True,
                              output_prediction_file=prediction_file)

    answers = read_answers(file_path)
    preds_dict = json.loads(preds)
    metrics = evaluate(answers, preds_dict, skip_no_answer=args.skip_no_ans)

    return metrics
