import os
import pickle
import argparse
from generator.iterator import read_squad_examples, read_level_file, convert_examples_to_features, \
    set_level_in_examples, sort_features_by_level

from pytorch_pretrained_bert import BertTokenizer


def pre_process(tokenizer, args):
    level_folder = args.level_folder

    pickled_folder = args.pickled_folder + "_{}_{}".format(args.bert_model, str(args.skip_no_ans))

    if not os.path.exists(pickled_folder):
        os.mkdir(pickled_folder)

    features_lst = []
    files = [f for f in os.listdir(args.train_folder) if f.endswith(".gz")]
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
            file_path = os.path.join(args.train_folder, file)

            train_examples = read_squad_examples(file_path, debug=args.debug)
            # read level file and set level
            levels = read_level_file(level_path, sep='\t')
            train_examples = set_level_in_examples(train_examples, levels)

            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                max_query_length=args.max_query_length,
                doc_stride=args.doc_stride,
                is_training=True,
                skip_no_ans=args.skip_no_ans
            )
            train_features = sort_features_by_level(train_features, desc=False)

            features_lst.append(train_features)

            # Save feature lst as pickle (For reuse & fast loading)
            with open(pickle_file_path, 'wb') as pkl_f:
                print("Saving {} file as pkl...".format(data_name))
                pickle.dump(train_features, pkl_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", type=int, default=384, help="max seq length")
    parser.add_argument("--max_query_length", type=int, default=64, help="max query len")
    parser.add_argument("--doc_stride", type=int, default=128, help="doc stride")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="bert-base-uncased or bert-large-uncased")
    parser.add_argument("--level_folder", type=str, default="./generator/difficulty", help="level folder")
    parser.add_argument("--train_folder", type=str, default="./data/train", help="train folder")
    parser.add_argument("--pickled_folder", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--skip_no_ans", action="store_true", help="whether to exclude no answer example")
    parser.add_argument("--do_lower_case", type=bool, default=False, help="whether to do lower case")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    pre_process(tokenizer, args)
