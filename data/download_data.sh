#! /bin/bash
set -e

# 1. Download train data
mkdir -p train

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O ./train/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O ./train/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O ./train/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O ./train/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O ./train/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O ./train/NaturalQuestions.jsonl.gz

# 2. Download dev data
mkdir -p dev

wget http://participants-area.bioasq.org/MRQA2019/ -O ./dev/BioASQ.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz -O ./dev/TextbookQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz -O ./dev/RelationExtraction.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz -O ./dev/DROP.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz -O ./dev/DuoRC.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz -O ./dev/RACE.jsonl.gz
