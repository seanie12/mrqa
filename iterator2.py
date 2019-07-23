import os

import pickle
import multiprocessing
import argparse

from generator.iterator import *


def save_features(args):
    file, args = args
    print(file)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    data_name = file.split(".")[0]
    pickled_folder = args.pickled_folder + "_{}_{}".format(args.bert_model, str(args.skip_no_ans))
    # Check whether pkl file already exists
    pickle_file_name = data_name + '.pkl'
    pickle_file_path = os.path.join(pickled_folder, pickle_file_name)
    if os.path.exists(pickle_file_path):
        pass
    else:
        level_name = data_name + ".tsv"
        print("processing {} file".format(data_name))
        level_path = os.path.join(args.level_folder, level_name)
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

        # Save feature lst as pickle (For reuse & fast loading)
        with open(pickle_file_path, 'wb') as pkl_f:
            print("Saving {} file as pkl...".format(data_name))
            pickle.dump(train_features, pkl_f)

        print(file, "done")
        # return train_features


def iter_main(args):
    print("start data pre-load with multiprocessing.")

    files = [(f, args) for f in os.listdir(args.train_folder) if f.endswith(".gz")]
    print("the number of data-set:{}".format(len(files)))

    pool = multiprocessing.Pool(processes=len(files))
    pool.map(save_features, files)
    pool.close()
    pool.join()

    # return features_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="max query length")
    parser.add_argument("--doc_stride", default=128, type=int)

    parser.add_argument("--do_lower_case", default=True, help="do lower case on text")
    parser.add_argument("--curriculum", action="store_true", help="enable curriculum mechanism")

    parser.add_argument("--do_valid", default=True, help="do validation or not")
    parser.add_argument("--freeze_bert", action="store_true", help="freeze bert parameters or not")

    parser.add_argument("--train_folder"
                        , default="/home/adam/data/mrqa2019/download_train"
                        , type=str, help="path of training data file")
    parser.add_argument("--dev_folder"
                        , default="/home/adam/data/mrqa2019/download_out_of_domain_dev"
                        , type=str, help="path of training data file")
    parser.add_argument("--level_folder"
                        , default="./difficulty"
                        , type=str, help="path of difficulty file")
    parser.add_argument("--pickled_folder"
                        , default="./pickled_data"
                        , type=str, help="path of saved pickle file")

    args = parser.parse_args()

    iter_main(args)
