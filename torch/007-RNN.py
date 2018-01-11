import torch
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

EPOCH = 1
TIME_STEP = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
LR = 0.01

train_data = torchvision.datasets.MNIST('./mnist', True, torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST('./mnist', False, torchvision.transforms.ToTensor())
test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:10]/255.0
#这个地方test_x不需要 dim=1 加维成4D, 因为 rnn 接受 [batch, time_step, input_size] 3D的数据
test_y = test_data.test_labels.numpy().squeeze()[:10]


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.LSTM( # LSTM 效果要比 nn.RNN() 好多了
            input_size=INPUT_SIZE, #图片每行的数据像素点
            hidden_size=64, # rnn hidden unit, 输出
            num_layers=1, # 有几层 RNN layers
            batch_first=True,
            # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None) # None 表示初始 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值 [50, 28, 64]
        out = self.out(r_out[:, -1, :]) #[batch, time_step, input_size] 最后一个时刻的输出
        return out

rnn = RNN()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss() #target 是 LongTensor,is not one-hotted

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # x.size() = [50, 1, 28, 28]
        b_x = Variable(x.view(-1, 28, 28)) # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            predict = rnn(test_x)
            p = torch.max(predict, 1)[1].data.numpy()
            print(p)
            print(test_y)