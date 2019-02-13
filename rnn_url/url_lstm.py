import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

import glob
import os,sys
import string
import numpy as np

all_letters = string.ascii_letters + " .,;'-:/"
all_letters = string.printable
n_letters = len(all_letters) + 1

with open("output.log", 'w') as f:
    pass

def dprint(*line):
    if len(sys.argv) > 1 and sys.argv[1]=='d':
        print(line)

def fprint(line):
    print(line)
    with open("output.log",'a') as f:
        f.seek(2,0)
        print(line, file=f)

def readlines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return lines

filename = "data/url.txt"
url_lines = readlines(filename)
dprint(url_lines)
dprint(len(url_lines))

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    #dprint("inputTensor", tensor.size())
    return tensor

def targetTensor(line):
    later_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    later_indexes.append(n_letters - 1) # EOS
    #dprint("targetTensor", len(later_indexes))
    return torch.LongTensor(later_indexes)


class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        #inputs = inputs.unsqueeze(0)
        dprint("forward", inputs.size())
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet():
    train_x = []
    train_t = []

    for i in range(len(url_lines)):
        train_x.append(inputTensor(url_lines[i]))
        train_t.append(targetTensor(url_lines[i]))
    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    return torch.cat(batch_x), torch.cat(batch_t)

def sample(start_letter, model):
    with torch.no_grad():
        input = inputTensor(start_letter)
        output_name = start_letter
        for i in range(100):
            output = model(input)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name

def main():
    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 128
    batch_size = 1

    train_x, train_t = mkDataSet()

    model = Predictor(n_letters, hidden_size, n_letters)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)
            dprint("data", data.size())

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

        print('%d loss: %.3f' % (epoch, running_loss))
        for i in range(ord('a'), ord('z')):
            fprint(sample(str(i), model))

if __name__ == '__main__':
    main()
