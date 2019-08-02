from trainer import BaseTrainer, AdvTrainer, PreTrainer
from distributed_run import distributed_main
import torch
from iterator2 import *
import os
import time


def main(args):
    # data loading before initializing model
    pickled_folder = args.pickled_folder + "_{}_{}".format(args.bert_model, str(args.skip_no_ans))

    if not os.path.exists(pickled_folder) and not args.debug:
        os.mkdir(pickled_folder)
    iter_main(args)

    # make save and result directory
    save_dir = os.path.join("./save",
                            "{}_{}".format("adv" if args.adv else "base", time.strftime("%m%d%H%M")))
    if not os.path.exists(save_dir) and not args.debug:
        os.makedirs(save_dir)
    args.save_dir = save_dir

    result_dir = os.path.join("./result",
                              "{}_{}".format("adv" if args.adv else "base", time.strftime("%m%d%H%M")))
    if not os.path.exists(result_dir) and not args.debug:
        os.makedirs(result_dir)
    args.result_dir = result_dir
    args.devices = [int(gpu) for gpu in args.devices.split('_')]
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.distributed = (args.use_cuda and args.multiprocessing_distributed)

    if args.distributed:
        distributed_main(args)
    else:
        if args.adv:
            model = AdvTrainer(args)
        elif args.pretraining:
            model = PreTrainer(args)
        else:
            model = BaseTrainer(args)

        model.make_model_env(None, 4)
        model.make_run_env()
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--meta", action="store_true", help="whether to trian meta")
    parser.add_argument("--meta_lambda", default=0.5, help="meta lambda")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="max query length")
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--batch_size", default=68, type=int, help="batch size")
    parser.add_argument("--epochs", default=2, type=int, help="number of epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch point")
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient_accumulation_steps")
    parser.add_argument("--bert_config_file", default="./data/bert_base_config.json", type=str, help="bert config file")

    parser.add_argument("--do_lower_case", default=True, help="do lower case on text")
    parser.add_argument("--use_cuda", default=True, help="use cuda or not")
    parser.add_argument("--curriculum", action="store_true", help="enable curriculum mechanism")

    parser.add_argument("--do_valid", default=True, help="do validation or not")
    parser.add_argument("--freeze_bert", action="store_true", help="freeze bert parameters or not")

    parser.add_argument("--train_folder"
                        , default="./data/train"
                        # , default="/home/adam/data/mrqa2019/download_train"
                        , type=str, help="path of training data file")
    parser.add_argument("--dev_folder"
                        , default="./data/dev"
                        # , default="/home/adam/data/mrqa2019/download_out_of_domain_dev"
                        , type=str, help="path of training data file")
    parser.add_argument("--level_folder"
                        , default="./generator/difficulty"
                        , type=str, help="path of difficulty file")
    parser.add_argument("--pickled_folder"
                        , default="./pickled_data"
                        , type=str, help="path of saved pickle file")
    parser.add_argument("--load_model"
                        , default=None
                        , type=str, help="load model")
    parser.add_argument("--skip_no_ans", action="store_true", help="whether to exclude no answer example")
    parser.add_argument("--devices",
                        type=str,
                        default='0_1_2_3',
                        help="gpu device ids to use")

    parser.add_argument("--workers", default=4
                        , help="Number of processes(workers) per node."
                               "It should be equal to the number of gpu devices to use in one node")
    parser.add_argument("--world_size", default=1,
                        help="Number of total workers. Initial value should be set to the number of nodes."
                             "Final value will be Num.nodes * Num.devices")
    parser.add_argument("--rank", default=0, help="The priority rank of current node.")
    parser.add_argument("--dist_backend", default="nccl",
                        help="Backend communication method. NCCL is used for DistributedDataParallel")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:9999", help="DistributedDataParallel server")
    parser.add_argument("--gpu", default=None,
                        help="Manual setting of gpu device. If it is not None, all parallel processes are disabled")
    parser.add_argument("--multiprocessing_distributed", default=False, help="Use multiprocess distribution or not")
    parser.add_argument("--save_model_by_all_devices", default=False, help="Save best model in all devices or not")
    parser.add_argument("--make_sample_prediction", default=False, help="Make sample prediction during training or not")
    parser.add_argument("--random_seed", default=2019, help="random state (seed)")
    # for adversarial learning
    parser.add_argument("--adv", action="store_true", help="adversarial training")
    parser.add_argument("--pretraining", action="store_true", help="pretraining discriminator")
    parser.add_argument("--dis_lambda", type=float, default=0.01, help="importance of adversarial loss")
    parser.add_argument("--num_classes", type=int, default=6, help="num_classes for discriminator")
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size for discriminator")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers for discriminator")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout for discriminator")
    parser.add_argument("--anneal", action="store_true")
    parser.add_argument("--qa_path", type=str, default="./save/qa/base_model", help="pre-trained model path")
    parser.add_argument("--dis_path", type=str, default="./save/dis/dis_best")
    args = parser.parse_args()

    main(args)
