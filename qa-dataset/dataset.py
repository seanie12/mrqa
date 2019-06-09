import json_lines
import sys

# TODO: offset에 대한 Dictionary는 어떻게?, 모든 dataset의 dictionary가 동일한가?
'''
[qas 한 instance의 구성]

{'answers': ['Super Bowl', 'Super Bowl', 'Super Bowl'], 
'question': 'The name of the NFL championship game is?', 
'id': '56d9895ddc89441400fdb50e', 
'qid': '5668cdd5c25b4549856d628a3ec248d9', 
'question_tokens': [['The', 0], ['name', 4], ['of', 9], ['the', 12], ['NFL', 16], ['championship', 20], ['game', 33], ['is', 38], ['?', 40]], 
'detected_answers': [{'text': 'Super Bowl', 'token_spans': [[0, 1], [86, 87], [51, 52], [114, 115], [131, 132]], 'char_spans': [[0, 9], [449, 458], [293, 302], [609, 618], [693, 702]]}, 
                     {'text': 'Super Bowl', 'token_spans': [[0, 1], [86, 87], [51, 52], [114, 115], [131, 132]], 'char_spans': [[0, 9], [449, 458], [293, 302], [609, 618], [693, 702]]}, 
                     {'text': 'Super Bowl', 'token_spans': [[0, 1], [86, 87], [51, 52], [114, 115], [131, 132]], 'char_spans': [[0, 9], [449, 458], [293, 302], [609, 618], [693, 702]]}]}
'''

# answers가 searching heuristic에 의해서 구해져서 중복이 여러개 있을 수 있다고 함.
# NOTE: 중복되는 것은 모두 제거해야할 필요가 있음!


class Dataset(object):
    """
    parameters: c_max_len, (for padding)
                q_max_len, (for padding),
                batch_size
    """

    def __init__(self, parameter, data_path):
        self.parameter = parameter
        self.data_path = data_path
        self.read_jsonl_file()
        self.make_input_data()

    def read_jsonl_file(self):
        self.json_data = []
        with open(self.data_path, 'rb') as f:  # opening file in binary(rb) mode
            for item in json_lines.reader(f):
                # print(item) #or use print(item['X']) for printing specific data
                self.json_data.append(item)

        self.json_data = self.json_data[1:]  # Erase header

    def padding(self, max_len, lst):
        '''
        max_sent_len = 80
        len(lst) = 60
        '''
        padding_token_idx = 0
        num_padding = max(max_len - len(lst), 0)
        for _ in range(num_padding):
            lst.append(padding_token_idx)
        return lst[:max_len]

    def make_input_data(self):
        self.c_lst = []
        self.q_lst = []
        self.a_lst = []
        for item in self.json_data:
            # 1. get context
            c = []
            c_tokens = item['context_tokens']
            # "context_tokens": [(token_1, offset_1), ..., (token_l, offset_l)]
            for token, offset in c_tokens:
                c.append(offset)

            # NOTE: Padding for context
            c = self.padding(self.parameter['c_max_len'], c)

            # 2. For each questions, get question
            for question in item['qas']:
                q = []
                for token, offset in question['question_tokens']:
                    q.append(offset)

                # NOTE: Padding for questions
                q = self.padding(self.parameter['q_max_len'], q)

                # 3. check all the possible answer list
                a_set = []
                for answer in question['detected_answers']:
                    for aa in answer['token_spans']:
                        a_set.append(aa)

                # 중복 제거가 필요함
                # https://inma.tistory.com/132
                a_set = list(set([tuple(set(item)) for item in a_set]))

                # 4. 중복 제거된 answer 기준으로 데이터에 넣기
                for a in a_set:
                    self.c_lst.append(c)
                    self.q_lst.append(q)
                    self.a_lst.append(a)

        # assert len(self.c_lst) == len(self.q_lst) == len(self.a_lst)
        

    def get_iterator(self):
        # TODO: Padding이 필요할텐데...
        batch_size = self.parameter['batch_size']
        for i, step in enumerate(range(0, len(self.c_lst), batch_size)):
            if len(self.c_lst[step:step+batch_size]) == batch_size:
                yield self.c_lst[step:step+batch_size], \
                    self.q_lst[step:step+batch_size], \
                    self.a_lst[step:step+batch_size]

    def __len__(self):
        return len(self.c_lst)


if __name__ == "__main__":
    parameter = {
        'c_max_len' : 100,
        'q_max_len' : 50,
        'batch_size' : 2
    }
    dataset = Dataset(parameter, 'SQuAD.jsonl')
    print(len(dataset))
    for c, q, a in dataset.get_iterator():
        print(c)
        print(q)
        print(a)
        print()
        sys.exit(0)
