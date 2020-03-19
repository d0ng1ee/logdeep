import torch
import torch.nn as nn
from torch.autograd import Variable

class Model_attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model_attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28


    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        #print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input0, input1):
        h0_0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0), self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        # print(out0[:, -1, :].size(),out1[:, -1, :].size())
        # print(out0.size()) # batch_size,sequence size,hidden_size

        # attn_output = self.attention_net(out1)

        # multi_out = torch.cat((out0[:, -1, :], attn_output),-1)
        # print(multi_out.size())
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]),-1)
        out = self.fc(multi_out)
        return out