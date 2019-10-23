import os
import time
import argparse

import torch
import torch.multiprocessing as mp

from trainer import BaseTrainer, AdvTrainer
from iterator import iter_main


# should be located outside of main function!
def worker(gpu, ngpus_per_node, args):
    if args.adv:
        print("running adv training...")
        model = AdvTrainer(args)
    else:
        print("running base training...")
        model = BaseTrainer(args)
    model.make_model_env(gpu, ngpus_per_node)
    model.make_run_env()
    model.train()


def main(args):
    # data loading before initializing model
    pickled_folder = args.pickled_folder + "_{}_{}".format(args.bert_model, str(args.skip_no_ans))
    if not os.path.exists(pickled_folder):
        os.mkdir(pickled_folder)
    file_num = iter_main(args)
    args.num_classes = file_num

    # make save and result directory
    save_dir = os.path.join("./save", "{}_{}".format("adv" if args.adv else "base", time.strftime("%m%d%H%M")))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir

    result_dir = os.path.join("./result", "{}_{}".format("adv" if args.adv else "base", time.strftime("%m%d%H%M")))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    args.result_dir = result_dir
    args.devices = [int(gpu) for gpu in args.devices.split('_')]
    args.use_cuda = (args.use_cuda and torch.cuda.is_available())
    args.distributed = (args.use_cuda and args.distributed)

    ngpus_per_node = 0
    if args.use_cuda:
        ngpus_per_node = len(args.devices)
        assert ngpus_per_node <= torch.cuda.device_count(), "GPU device number exceeds max capacity. select device ids correctly."

    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        worker(None, ngpus_per_node, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debugging mode, taking only first 100 data")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="Bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="Max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="Max query length")
    parser.add_argument("--doc_stride", default=128, type=int, help="doc stride")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch point")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient_accumulation_steps")

    parser.add_argument("--do_lower_case", action='store_true', default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_cuda", action='store_true', help="use cuda or not")

    parser.add_argument("--do_valid", action='store_true', help="do validation or not")
    parser.add_argument("--freeze_bert", action="store_true", help="freeze bert parameters or not")

    parser.add_argument("--train_folder", default="./data/train", type=str, help="path of training data file")
    parser.add_argument("--dev_folder", default="./data/dev", type=str, help="path of training data file")
    parser.add_argument("--pickled_folder", default="./pickled_data", type=str, help="path of saved pickle file")
    parser.add_argument("--load_model", default=None, type=str, help="load model")
    parser.add_argument("--skip_no_ans", action="store_true", help="whether to exclude no answer example")
    parser.add_argument("--devices", default='0', type=str, help="gpu device ids to use, concat with '_', ex) '0_1_2_3'")

    parser.add_argument("--workers", default=4, help="Number of processes(workers) per node."
                                                     "It should be equal to the number of gpu devices to use in one node")
    parser.add_argument("--world_size", default=1,
                        help="Number of total workers. Initial value should be set to the number of nodes."
                             "Final value will be Num.nodes * Num.devices")
    parser.add_argument("--rank", default=0, help="The priority rank of current node.")
    parser.add_argument("--dist_backend", default="nccl",
                        help="Backend communication method. NCCL is used for DistributedDataParallel")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:9999", help="DistributedDataParallel server")
    parser.add_argument("--gpu", default=None, help="Manual setting of gpu device. If it is not None, all parallel processes are disabled")
    parser.add_argument("--distributed", action="store_true", help="Use multiprocess distribution or not")
    parser.add_argument("--random_seed", default=2019, help="Random state(seed)")

    # For adversarial learning
    parser.add_argument("--adv", action="store_true", help="Use adversarial training")
    parser.add_argument("--dis_lambda", default=0.01, type=float, help="Importance of adversarial loss")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden size for discriminator")
    parser.add_argument("--num_layers", default=3, type=int, help="Number of layers for discriminator")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for discriminator")
    parser.add_argument("--anneal", action="store_true")
    parser.add_argument("--concat", action="store_true", help="Whether to use both cls and sep embedding")
    args = parser.parse_args()

    main(args)
