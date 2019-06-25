import gzip
import json_lines
import copy
import sys
import logging
import collections
import codecs

from pytorch_pretrained_bert.tokenization import BertTokenizer


def get_logger(log_name):
    # Write log
    logging.basicConfig(filename=log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
    logger.info('')
    logger.info("#################################### New Start #####################################")
    return logger


logger = get_logger('qa.log')


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.level = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 level=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        # For level
        self.level = level


def read_squad_examples(input_file):
    # Read data
    unproc_data = []
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:  # opening file in binary(rb) mode
        for item in json_lines.reader(f):
            # print(item) #or use print(item['X']) for printing specific data
            unproc_data.append(item)

    # Delete header
    unproc_data = unproc_data[1:]

    ###################### Make Examples ######################
    examples = []
    for item in unproc_data:
        # 1. Get Context
        doc_tokens = []
        for token in item['context_tokens']:
            doc_tokens.append(token[0])

        # 2. qas
        for qa in item['qas']:
            qas_id = qa['qid']  # NOTE: 모든 데이터셋에 qid는 존재하고, unique하다
            question_text = qa['question']

            answer_lst = []  # question 중복이 있는지 없는지 확인
            for answer in qa['detected_answers']:
                orig_answer_text = answer['text']
                # NOTE: Detected Answer를 보면 중복되는 것이 너무 많다. 해당 부분 전부 제거 필요
                if orig_answer_text in answer_lst:
                    continue
                else:
                    answer_lst.append(orig_answer_text)

                # Only take the first span
                start_position = answer['token_spans'][0][0]
                end_position = answer['token_spans'][0][1]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """
    max_seq_length: "The maximum total input sequence length after WordPiece tokenization. Sequences 
                    longer than this will be truncated, and sequences shorter than this will be padded."
    doc_stride: "When splitting up a long document into chunks, how much stride to take between chunks."
    max_query_length: "The maximum number of tokens for the question. Questions longer than this will be truncated to this length."
    """
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None

            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                answer_text = " ".join(tokens[start_position:(end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info(
                    "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    level=example.level))
            unique_id += 1

    return features


def read_level_file(input_file, sep='\t'):
    '''
    - id, level의 두개의 column으로 존재
    - 딕셔너리 형태로 저장함
    '''
    levels = dict()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item_lst = line.rstrip().split(sep)
            levels[item_lst[0]] = float(item_lst[1])

    return levels


def sort_features_by_level(features, desc=False):
    """
    features: feature 객체가 들어있는 list
    desc: 난이도별로 내림차순으로 정렬할 것인가
    """
    # 난이도 별로 sort하기
    features.sort(key=lambda x: x.level, reverse=desc)
    return features


def set_level_in_examples(examples, levels):
    """
    examples: example 객체가 들어있는 list
    levels: {'id1': 0.48. 'id2': 0.12, ...}의 딕셔너리
    """
    for example in examples:
        try:
            example.level = levels[example.qas_id]  # 해당 객체의 id를 levels의 딕셔너리에 넣으면 난이도가 나옴
        except:
            print("Level doesn't exists...Setting the level 0.5 as default")
            example.level = 0.5

    return examples


class Config(object):
    def __init__(self,
                 bert_model='bert-base-uncased',
                 do_lower_case=True,
                 max_seq_length=384,
                 max_query_length=64,
                 doc_stride=128,
                 batch_size=8,
                 gradient_accumulation_steps=1,
                 warmup_proportion=0.1,
                 lr=5e-5,
                 epochs=5):
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr
        self.epochs = epochs


if __name__ == "__main__":
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    train_examples = read_squad_examples('SQuAD.jsonl.gz')

    # import random

    # with open('squad_level.txt', 'w', encoding='utf-8') as f:
    #     for example in train_examples:
    #         f.write("{}\t{}\n".format(example.qas_id, random.random()))

    """
    ***** Running training *****
        Num orig examples = 34287
        Num split examples = 35111 (train_features)
    examples와 features의 길이가 서로 다르다. 그런데 난이도 함수는 보통 전체 데이터셋(examples) 기준
    """
    # Read level
    levels = read_level_file('SQuAD.tsv', sep='\t')
    # Set level attribute in each example
    train_examples = set_level_in_examples(train_examples, levels)

    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        max_query_length=config.max_query_length,
        doc_stride=config.doc_stride
    )

    # Read Level and sort
    train_features = sort_features_by_level(train_features, desc=False)

    # Check
    for i in range(10):
        print(train_features[i].level)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
