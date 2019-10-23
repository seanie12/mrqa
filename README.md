# Domain-agnostic Question-Answering with Adversarial Training

Code for our paper ["Domain-agnostic Question-Answering with Adversarial Training"](https://arxiv.org/abs/1910.09342) which is accepted by EMNLP-IJCNLP 2019 MRQA Workshop.

## Data Preparation

### Option 1: Download the original data

- Download the data by running shell file.
- Then run the code. Preprocessed train data will be created before training (It will takes quite a long time)

```bash
$ cd data
$ ./download_data.sh
```

### Option 2: Download pickled data (Much faster)

- Download the pickled data from this [link](https://drive.google.com/open?id=1-IHdLL4oLOI_Ur8ej-KUZ4kVGGuSKcJ2).

- Unzip the zipfile on the root directory.

```bash
.
├── ...
├── pickled_data_bert-base-uncased_False
│   ├── HotpotQA.pkl
│   ├── NaturalQuestions.pkl
│   ├── NewsQA.pkl
│   ├── SQuAD.pkl
│   ├── SearchQA.pkl
│   └── TriviaQA.pkl
└── ...

```

- **Arguments should be same as below if you use pickled data. If you want to change one of these two arguments.**

```bash
parser.add_argument("--bert_model", default="bert-base-uncased", type=str, help="Bert model")
parser.add_argument("--skip_no_ans", action='store_true', default=False, help="whether to exclude no answer example")
```

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

```bash
torch==1.1.0
pytorch-pretrained-bert>=0.6.2
json-lines>=0.5.0
```

## Model Training & Validation

```bash
$ python3 main.py \
         --epochs 2 \
         --batch_size 64 \
         --lr 3e-5 \
         --do_lower_case \
         --use_cuda \
         --do_valid \
         --adv \
         --dis_lambda 0.01
```

- If you are using uncased bert model, give the option `--do_lower_case`.
- If you want to do validation, give the option `--do_valid`.

## Reference

```
@misc{lee2019domainagnostic,
    title={Domain-agnostic Question-Answering with Adversarial Training},
    author={Seanie Lee and Donggyu Kim and Jangwon Park},
    year={2019},
    eprint={1910.09342},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contributors

- Lee, Seanie (https://github.com/seanie12)
- Kim, Donggyu (https://github.com/donggyukimc)
- Park, Jangwon (https://github.com/monologg)
