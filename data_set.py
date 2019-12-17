from torch.utils.data import Dataset
import gensim
import numpy as np
import torch
import json
from torch.nn.utils.rnn import pad_sequence

PATH = r'D:\PyCodes\wos_extraction\preprocess\data.json'
w2v_path = r'D:\PyCodes\wos_extraction\preprocess\word2vec.w2v'
label_dict = {"<START>": 0, "P": 1, "M": 2, "MP": 3, "O": 4, "<END>": 5}
special_token_list = ["<PADDING>", "<UNKNOWN>"]


class ModelEmbedding:

    def __init__(self, w2v_path):
        # 加载词向量矩阵
        self.word2vector = gensim.models.word2vec.Word2Vec.load(w2v_path).wv
        # 输入的总词数
        self.input_word_list = special_token_list + self.word2vector.index2word
        self.embedding = np.concatenate((np.random.rand(2, 100), self.word2vector.vectors))
        self.n_voc = len(self.input_word_list)
        self.n_class = len(label_dict)
        self.get_id2label()
        self.get_word2id()

    def get_word2id(self):
        # word2id词典
        self.word2id = dict(zip(self.input_word_list, list(range(len(self.input_word_list)))))
        # return self.word2id

    def get_id2label(self):
        # id2label词典
        self.id2word = dict(zip(label_dict.values(),label_dict.keys()))
        # return self.id2word


class MyDataSet(Dataset):

    def __init__(self, JSON_PATH,  word2id):
        file = open(JSON_PATH, "r")
        self.raw_data= json.load(file)[:1000]

        # 填充输入序列
        self.x_list = [[word2id[w] if w in word2id else word2id["<UNKNOWN>"]for w in item[0]]for item in self.raw_data ]
        # print(len(self.x_list))
        # 填充标记序列
        self.y_list = [[label_dict[label] for label in item[1]] for item in self.raw_data ]
        # print(len(self.y_list))

    def __getitem__(self, item):
        return torch.Tensor(self.x_list[item]), torch.Tensor(self.y_list[item])

    def __len__(self):
        return len(self.raw_data)


def collate_fn(batch):
    """
    :param batch: (batch_num, ([sentence_len, word_embedding], [sentence_len]))
    :return:
    """
    x_list = [x[0] for x in batch]
    y_list = [x[1] for x in batch]
    lengths = [len(item[0]) for item in batch]
    x_list = pad_sequence(x_list, padding_value=0)
    y_list = pad_sequence(y_list, padding_value=-1)
    return x_list.transpose(0, 1), y_list.transpose(0, 1), lengths




