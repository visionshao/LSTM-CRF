from model.dependency.data_set import *
from model.dependency.BiLstmCrf import *
from torch.utils.data import DataLoader

# embedding
model_embedding = ModelEmbedding(w2v_path)  # 构造Embedding model
# model建立，100是batch_size，4是类别数
model = LstmCrf(input_dim=100, hidden_dim=100, n_class=model_embedding.n_class, rnn_type="l",
                n_voc=model_embedding.n_voc, model_embedding=model_embedding)
# 优化器
optim = torch.optim.Adam(model.parameters(), lr=0.1)
# 得到word2id字典，构造dataset
word2id = model_embedding.word2id
# 构造dataset
my_dataset = MyDataSet(PATH, word2id)
# 训练集数据与测试集数据4:1
train_size = int(0.8 * len(my_dataset))
test_size = len(my_dataset) - train_size
# 训练集，测试集
train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])
# 构造dataloader，batch_size=4，注意：数据内存要求高，batch_size不能过大
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, collate_fn=collate_fn)
# 100个epoch
for i in range(100):
    loss = 0
    for x, y, lengths in train_dataloader:
        x = x.long()
        y = y.long()
        mask = get_mask(lengths)
        emission = model.forward(input_data=x, input_len=lengths)
        loss = model.get_loss(emission=emission, labels=y, mask=mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(loss)
