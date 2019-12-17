from __future__ import unicode_literals
import torch
from model.dependency.data_set import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch.nn import init


def get_mask(length):
    # 求出一个batch中最大的输入长度
    max_len = int(max(length))
    mask = torch.Tensor()
    # length = length.numpy()
    # 与每个序列等长度的全1向量连接长度为最大序列长度-当前序列长度的全0向量。
    for len_ in length:
        mask = torch.cat((mask, torch.Tensor([[1] * len_ + [0] * (max_len-len_)])), dim=0)
    return mask


class LstmCrf(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_class, rnn_type, n_voc, model_embedding):
        super(LstmCrf, self).__init__()
        # argument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.rnn_type = rnn_type
        self.n_voc = n_voc
        # embedding
        self.model_embedding = model_embedding
        self.embedding = nn.Embedding(num_embeddings=self.n_voc,
                                 embedding_dim=self.input_dim, padding_idx=0)
        # self.embedding.weight.data.copy_(torch.from_numpy(model_embedding.embedding))

        # neural network
        if self.rnn_type[0] == "l":
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        elif self.rnn_type[0] == "g":
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        else:
            print("Fucking wrong type of rnn!")
        self.linear = nn.Linear(2*hidden_dim, n_class)
        # transition matrix logP(y_i, y_i+1)
        self.transition_matrix = nn.Parameter(torch.rand(n_class, n_class))
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=1)

    def reset_parameters(self):
        # 转移概率位于log空间
        init.normal_(self.transition_matrix)
        # initialize START_TAG, END_TAG probability in log space
        self.transition_matrix.detach()[label_dict["<START>"], :] = -10000
        self.transition_matrix.detach()[:, label_dict["<END>"]] = -10000

    def forward(self, input_data, input_len):
        """
        :param input: data [batch_size, len]
        :param input_len: data_len_list，
        :return:
        """
        # print(input_data)
        embedded_input = self.embedding(input_data)
        packed_embedded_input = pack_padded_sequence(input=embedded_input, lengths=input_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_embedded_input)
        output, _ = pad_packed_sequence(packed_output)
        output = self.linear(output)
        # 位于log空间不加softmax
        # output = self.softmax(output)
        output = output.transpose(0, 1)
        return output

    def forward_alpha(self, emission, mask):
        # 发射矩阵， emission
        # batch大小，最大序列长度
        batch_size, seq_len = mask.size()
        # logα_0
        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000)
        # i=0时，label为start的概率为1，其余为0,取log
        log_alpha_init[:, 0] = 0
        # alpha, [batch_size, n_class]
        log_alpha = log_alpha_init
        # 该部分一直到计算α_1到α_n
        for w in range(0, seq_len):
            # 取出当前时刻的mask,每个batch中的元素，对应一个n_class*n_class的矩阵，矩阵所有元素相同，与mask[:.w]同
            # [batch_size, 1]
            mask_t = mask[:, w].unsqueeze(-1)
            # [batch_size, n_class]
            current = emission[:, w, :]
            # [batch_size, n_class, n_class]，
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)  # 增列
            # print(alpha_matrix.size())
            # [batch_size, n_class, n_class]，复制n_class行，
            current_matrix = current.unsqueeze(1).expand(-1, self.n_class, -1)  # 增行
            # 当前时刻的M_i(x) = exp(Transition + Emission)
            log_M_matrix = (current_matrix + self.transition_matrix)
            # [batch, n_class, n_class]
            add_matrix = log_alpha_matrix + log_M_matrix # wise add
            log_alpha = torch.logsumexp(add_matrix, dim=1) * mask_t + log_alpha * (1-mask_t)
        # log(α*1)
        total_score = torch.logsumexp(log_alpha, dim=1)
        return total_score

    def get_sentence_score(self, emission, labels, mask):
        batch_size, seq_len, n_class = emission.size()
        # 增加<START>label
        labels = torch.cat([labels.new_full((batch_size, 1), fill_value=label_dict["<START>"]), labels], 1)
                            # ,tag2idx[START_TAG])], 0) # (seq_len + 1, batch_size)
        scores = emission.new_zeros(batch_size)

        # M_1到M_n
        for i in range(seq_len):
            # [batch_size, 1]
            mask_i = mask[:, i]
            current = emission[:, i, :]
            # 取出每个词对应标签的score,i+1表明跳过start
            emit_score = torch.cat([each_score[next_label].unsqueeze(-1) for each_score, next_label in zip(current, labels[:, i+1])], dim=0)
            # print(emit_score.size())
            transition_score = torch.stack([self.transition_matrix[labels[b, i],labels[b, i+1]] for b in range(batch_size)])
            # print(transition_score.size())
            scores += (emit_score + transition_score) * mask_i
        transition_to_end = torch.stack([self.transition_matrix[label[mask[b,:].sum().long()], label_dict["<END>"]] for b, label in enumerate(labels)])
        scores += transition_to_end
        return scores

    def get_loss(self, emission, labels, mask):
        # log_Z
        log_Z = self.forward_alpha(emission, mask)
        # log_y
        log_alpha_n = self.get_sentence_score(emission, labels, mask)
        # -log(y/z)
        loss = (log_Z - log_alpha_n).sum()
        return loss

    def get_best_path(self, emission, mask):
        # 发射矩阵， emission
        # batch大小，最大序列长度
        batch_size, seq_len = mask.size()
        # logα_0
        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000)
        # i=0时，label为start的概率为1，其余为0,取log
        log_alpha_init[:, 0] = 0
        # alpha, [batch_size, n_class]
        log_alpha = log_alpha_init
        # 指针
        pointers = []
        # 该部分一直到计算α_1到α_n
        for w in range(0, seq_len):
            # 取出当前时刻的mask,每个batch中的元素，对应一个n_class*n_class的矩阵，矩阵所有元素相同，与mask[:.w]同
            # [batch_size, 1]
            mask_t = mask[:, w].unsqueeze(-1)
            # [batch_size, n_class]，当前的词对应的emission
            current = emission[:, w, :]
            # 扩展上一时刻的log_alpha [batch_size, n_class, n_class]，
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)  # 增列
            # 上一时刻的log_alpha与transition矩阵相作用，扩展到三阶张量, [batch_size, n_class, n_class]
            trans = log_alpha_matrix + self.transition_matrix
            # 选取上一时刻的y_i-1使得到当前时刻的某个y_i的路径分数最大,max_trans[batch_size, n_class]
            max_trans, pointer = torch.max(trans, dim=2)
            # 添加路径,注意此时的pointer指向的是上一时刻的label
            pointers.append(pointer)
            # 获取当前时刻新的log_alpha [batch_size, n_class]
            cur_log_alpha = current + max_trans
            # 根据mask判断是否更新，得到当前时刻的log_alpha
            log_alpha = log_alpha * (1-mask_t) + cur_log_alpha * mask_t
        # 将pointers转为张量, [batch_size, seq_len, n_class]
        pointers = torch.stack(pointers, 1)
        # n->END
        log_alpha = log_alpha + self.transition_matrix[label_dict["<END>"]]
        # 找到n->END的最优路径, [batch_size], [batch_size]
        best_log_alpha, best_label = torch.max(log_alpha, dim=1)
        best_path = []
        # 从后向前，不断寻路(注意不同数据的路长不同，不同数据单独寻路)
        for i in range(batch_size):
            # 当前数据的路径长度
            seq_len_i = int(mask[i].sum())
            # 当前数据对应的有效pointers[seq_len_i, n_class]
            pointers_i = pointers[i, :seq_len_i]
            # 当前数据的best_label
            best_label_i = best_label[i]
            # 遍历寻路,当前数据的路径
            best_path_i = [best_label_i]
            for j in range(seq_len_i):
                # 从后向前遍历
                index = seq_len_i-j-1
                # 当前时刻的best_label_i
                best_label_i = pointers_i[index][best_label_i]
                best_path_i = [best_label_i] + best_path_i
            # 除去时刻1之前的的路径
            best_path_i = best_path_i[1:]
            # 添加到总路径中
            best_path.append(best_path_i)
        return best_path




