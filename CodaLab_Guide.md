# CodaLab 사용법

- Web CLI의 경우는 업로드 등이 느린 관계로, 내 로컬에서 codalab 파이썬 라이브러리를 설치하는 것을 권장
- 근데 이 CodaLab이 Python2에서만 작동함....
    ```
    $ virtualenv -p python2.7 venv
    Running virtualenv with interpreter /usr/bin/python2
    ...
    $ . venv/bin/activate
    (venv) $ pip install codalab -U
    ```

- cl work <worksheet_name>

- cl upload <local_file_path>

- CodaLab 내부에서 인터넷을 사용할 수 없어서 pip 등도 사용이 불가능하다. 고로 Docker에 미리 다 다운을 받아야한다.

- 위와 같은 이유로 pytorch_pretrained_bert의 Tokenizer파일 같은 것도 미리 cached_path에 다운받아서 올려놔야한다. (신의가 이미 다 구현함! [Github 링크](https://github.com/seanie12/mrqa-serving))


## Tutorial
- 기타 튜토리얼: [https://github.com/codalab/worksheets-examples/tree/master/00-quickstart](https://github.com/codalab/worksheets-examples/tree/master/00-quickstart)

- 한국어 튜토리얼: [https://github.com/graykode/KorQuAD-beginner](https://github.com/graykode/KorQuAD-beginner)

- Docker와 Codalab 연동법: [https://github.com/codalab/codalab-worksheets/wiki/Execution](https://github.com/codalab/codalab-worksheets/wiki/Execution)