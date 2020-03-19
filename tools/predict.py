import torch
import torch.nn as nn
import time
import argparse
from tqdm import tqdm
from collections import Counter
from torch.autograd import Variable

# Device configuration
device = torch.device("cpu")
# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 28
num_candidates = 9
model_path = '../result/model_test/model_test_last.pth'


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = set()
    # hdfs = []
    with open('../data/hdfs/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            # 确保当实例小于10的时候也能预测结果
            line = line + [-1] * (window_size + 1 - len(line))
            hdfs.add(tuple(line))
            # hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs



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

        attn_output = self.attention_net(out1)

        multi_out = torch.cat((out0[:, -1, :], attn_output),-1)
        # print(multi_out.size())

        out = self.fc(multi_out)
        return out

'''
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, num_keys)

    def forward(self, input0, input1):
        h0_0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0), self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        # print(out0[:, -1, :].size(),out1[:, -1, :].size())

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]),-1)
        # print(multi_out.size())

        out = self.fc(multi_out)
        return out
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model_attention(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate('hdfs_test_normal')
    test_abnormal_loader = generate('hdfs_test_abnormal')


    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in tqdm(test_normal_loader):
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                label = line[i + window_size]
                seq1 = [0]*28
                log_conuter =  Counter(seq0)
                for key in log_conuter:
                    seq1[key] = log_conuter[key]

                seq0 = torch.tensor(seq0, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq0,seq1)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break
    with torch.no_grad():
        for line in tqdm(test_abnormal_loader):
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                label = line[i + window_size]
                seq1 = [0]*28
                log_conuter =  Counter(seq0)
                for key in log_conuter:
                    seq1[key] = log_conuter[key]

                seq0 = torch.tensor(seq0, dtype=torch.float).view(-1, window_size, input_size).to(device)
                seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, num_classes, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq0,seq1)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
