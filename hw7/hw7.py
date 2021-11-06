import torch
import os
from pathlib import Path
# from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
# import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from zipfile import ZipFile

SAVE_WEIGHTS = True
MODEL_FILENAME = "model.pth"
LOAD_WEIGHTS = False
ETA = 0.01
EON = '<EON>'
N = 27
SEQ_LEN = 11
BATCH_SIZE = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_filename():
  return "model_" + str(datetime.datetime.now()) + '.pth'

def load_weights(model):
  print("Loading weights from {}".format(MODEL_FILENAME))
  model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device('cpu')))
  print("Loaded weights from {}".format(MODEL_FILENAME))

def save_model(model):
  print("Saving model...")
  torch.save(model.state_dict(), get_model_filename())
  print("Model saved.")

def str_from_3d_tensor(tensor):
  words = []
  for batch_index in range(BATCH_SIZE):
    word = tensor[batch_index]
    w = ''
    for ch in word:
      c = get_char_from_tensor(ch)
      w = w + c
    words.append(w)
  return words
    

def name_to_2d_tensor(_name, shift=False):
  if shift:
    name = _name[1:]
  else:
    name = _name

  tensor = torch.zeros((SEQ_LEN, N))

  for index in range(SEQ_LEN):
    ch = name[index] if index < len(name) else EON
    tensor[index][get_index_value(ch)] = 1

  return tensor

def get_char_from_tensor(tensor):
  for i in range(N):
    if tensor[i].item() == 1:
      if i == 0:
        return EON
      else:
        return chr(i + 97 - 1)

def get_index_value(ch):
  if ch == EON:
    return 0
  return ord(ch[0]) - 97 + 1

class NamesDataset(Dataset):
  def __init__(self, train=True ,transform=None, target_transform=None):
    self.train = train
    self.test = not train

    f = open('names.txt', 'r')
    lines = f.readlines()
    self.names = [line.strip() for line in lines]
    self.names = list(map(lambda name: name.lower(), self.names))
    f.close()

    self.transform = transform
    self.target_transform = target_transform
    self.char_tensors = []

    for name in self.names:
      for i in range(SEQ_LEN):
        tensor = torch.zeros([N])
        if i < len(name):
          tensor[get_index_value(name[i])] = 1
        else:
          tensor[get_index_value(EON)] = 1
        self.char_tensors.append(tensor)


  def __len__(self):
    return len(self.names)

  def __getitem__old(self, index):
    if index == len(self.char_tensors) - 1:
      return self.char_tensors[index], EON # change this

    return self.char_tensors[index], self.char_tensors[index + 1]
  
  def __getitem__(self, index):
    name = self.names[index]
    return name_to_2d_tensor(name), name_to_2d_tensor(name, shift=True)


class RNN(nn.Module):
  def __init__(self):
    super(RNN, self).__init__()
    self.lstm = nn.LSTM(input_size=N, hidden_size=N, num_layers=1, batch_first=True)
    self.i2o = nn.Linear(N, N)
    # self.i2o = nn.Linear(2*N, N)


  def forward(self, x, prev_state):
    h, c = prev_state
    # print(x.shape)
    # print(prev_state.shape)

    # combined = torch.cat((x, prev_state), 1)
    # print("com")
    # print(combined.shape)
    output, (hn, cn) = self.lstm(x, (h, c))
    output = self.i2o(output)
    # output = F.log_softmax(output, dim=1) # CE loss may require unnormalized inputs
    return output, (hn, cn)

def train_epoch():
  prev_state = torch.zeros([BATCH_SIZE, SEQ_LEN, N])
  h = torch.zeros([1, BATCH_SIZE, N])
  c = torch.zeros([1, BATCH_SIZE, N])

  model.zero_grad()
  loss = 0


  for i, data in enumerate(training_dataloader, 0):
    inputs, labels = data
    
    # words = str_from_3d_tensor(inputs)
    # print("Words")
    # print(words)
    # words = str_from_3d_tensor(labels)
    # print("Labels")
    # print(words)
    # print("i: {}".format(i))
    # print("inputs shape")
    # print(inputs.shape)
    # print("labels shape")
    # print(labels.shape)
    # model.zero_grad() # weird
    # optimizer.zero_grad()

    # print("h.shape")
    # print(h.shape)
    # print("c.shape")
    # print(c.shape)

    output, (h, c) = model(inputs, (h, c))
    # print("output.shape")
    # print(output.shape)
    # print("hidden.shape")
    # print(hidden.shape)
    

    l = criterion(output, labels)
    loss += l
    # optimizer.step() # throwing an error
    # scheduler.step()

    if i % 100 == 0:
      pass
      # print("{} done".format(i))
      # print("loss: {}".format(loss.item()))

  loss.backward()

  for p in model.parameters():
    p.data.add_(p.grad.data, alpha=-ETA)

  return output, loss.item() / 11

def train():
  print("Starting training")
  total_loss = 0
  for epoch in range(100): # 2 epochs
    output, loss = train_epoch()
    total_loss += loss

    if epoch % 2 == 0:
      print('Epoch: {} Loss: {}'.format(epoch, loss))

    if epoch % 20 == 0 and epoch != 0:
      save_model(model)

def test():
  h = torch.zeros([1, BATCH_SIZE, N])
  c = torch.zeros([1, BATCH_SIZE, N])


if __name__ == "__main__":

  training_set = NamesDataset(train=True)
#   test_set = ImageDataset("test", get_image_tensor, get_label_tensor)

  training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
#   test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

  # for i in range(33):
    # print(get_char_from_tensor(training_set.__getitem__(i)))

  model = RNN()

  if LOAD_WEIGHTS:
    load_weights(model)
  else:
    print("Initialzing with random weights.")

  model.to(device)

  criterion = nn.CrossEntropyLoss() # handle for cuda as well
  # optimizer = optim.Adam(model.parameters(), lr=0.1)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

  train()
  # test()

  


  

'''
https://discuss.pytorch.org/t/understanding-lstm-input/31110
https://discuss.pytorch.org/t/why-3d-input-tensors-in-lstm/4455
https://towardsdatascience.com/lstms-in-pytorch-528b0440244
https://discuss.pytorch.org/t/understanding-lstm-input/31110/5
https://discuss.pytorch.org/t/please-help-lstm-input-output-dimensions/89353/3
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
Read about torch autograd and retaining gradients
Add probablistic labels
'''


