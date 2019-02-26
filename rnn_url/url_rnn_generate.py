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

#filename = "data/Japanese.txt"
filename = "data/url_dataset.txt"
url_lines = readlines(filename)
dprint(url_lines)

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        dprint("input", input.size())
        dprint("hidden", hidden.size())
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

#class randomChoice:
#    def __init__(self, l):
#        self.l = l
#        self.c = 0
#        self.len = len(self.l)
#
#    def choice(self):
#        ret = self.l[self.c]
#        self.c += 1
#        self.c %= self.len
#        return ret

import random
counter = 0
def randomChoice(l):
    #return l[random.randint(0, len(l) - 1)]
    global counter
    choice = l[counter]
    counter += 1
    counter %= len(l)
    dprint("choice", choice)
    return choice

def randomTrainingPair():
    line = randomChoice(url_lines)
    return line

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    dprint("line", line)
    later_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    later_indexes.append(n_letters - 1) # EOS
    dprint("later_indexes", later_indexes)
    return torch.LongTensor(later_indexes)


def randomTrainingExample():
    line = randomTrainingPair()
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor

# Training the Network
criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    dprint("unsqueeze_", target_line_tensor.size())
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        dprint("input_line_tensor[i]", input_line_tensor[i].size())
        dprint("hidden", hidden.size())
        output, hidden = rnn(input_line_tensor[i], hidden)
        dprint("---------------")
        dprint("output", output.size(), output)
        dprint("target_line_tensor[i]", target_line_tensor[i].size(), target_line_tensor[i])
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 100)
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

def sample(start_letter='A'):
    with torch.no_grad():
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(100):
            output, hidden = rnn(input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


rnn = RNN(n_letters, 128, n_letters)

n_iters = 100000
print_every = 1000
plot_every = 500
all_losses = []
total_loss = 0

for iter in range(1, n_iters+1):
    output, loss = train(*randomTrainingExample())
    if iter % print_every == 0:
        fprint('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, loss))
        assert np.isnan(loss)==False, 'nan'
        for i in range(ord('a'), ord('z')):
            fprint(sample(chr(i)))
        for i in range(ord('0'), ord('9')):
            fprint(sample(chr(i)))
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
        torch.save(rnn.state_dict(), './weights/checkpoint')
