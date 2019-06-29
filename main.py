import argparse

from generator.iterator import Config
from trainer import BaseTrainer


def main(args):
    config = Config(bert_model=args.bert_model,
                    max_seq_length=args.max_seq_length,
                    max_query_length=args.max_query_length,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    debug=args.debug,
                    gradient_accumulation_steps=args.grad_accum)

    trainer = BaseTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--max_seq_length", default=384, type=int, help="max sequence length")
    parser.add_argument("--max_query_length", default=64, type=int, help="max query length")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=5, type=int, help="number of epochs")
    parser.add_argument("--grad_accum", default=1, type=int, help="gradient_accumulation_steps")
    args = parser.parse_args()
    main(args)
