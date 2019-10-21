# Domain-agnostic Question-Answering with Adversarial Training

Implementation of our paper ["Domain-agnostic Question-Answering with Adversarial Training"](null) which is accepted by EMNLP-IJCNLP 2019 MRQA Workshop.

## Data Preparation

### Option 1: Download pickled data (Much faster)

- Download the pickled data from this [link](https://drive.google.com/open?id=150ZzHpjo_ddeCICIOwXso6VRk2FoZ383).

- Unzip the zipfile on the root directory.

- `.pkl` files will be in `pickled_data_bert-base-uncased_False`.

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

**Must give the options --bert_model bert-base-cased & --skip_no_ans False**

### Option 2: Download the original data

```bash
$ cd data
$ ./download_data.sh
```

## Requirements

TBD

## How to Run

```python
$ python3 main.py
```

## Reference

```
TBD
```
