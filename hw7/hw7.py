import torch
import os
from pathlib import Path
# from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
# import pandas as pd
from torchvision import transforms
from torchvision.io import ImageReadMode
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from zipfile import ZipFile

EON = '<EON>'
N = 27
SEQ_LEN = 11
BATCH_SIZE = 3

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
    self.i2o = nn.Linear(N + N, N)

  def forward(self, x, prev_state):
    h, c = prev_state
    # print(x.shape)
    # print(prev_state.shape)

    # combined = torch.cat((x, prev_state), 1)
    # print("com")
    # print(combined.shape)
    output, (hn, cn) = self.lstm(x, (h, c))
    output = F.log_softmax(output, dim=1) # CE loss may require unnormalized inputs
    return output, (hn, cn)



if __name__ == "__main__":

  training_set = NamesDataset(train=True)
#   test_set = ImageDataset("test", get_image_tensor, get_label_tensor)

  training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False)
#   test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

  # for i in range(33):
    # print(get_char_from_tensor(training_set.__getitem__(i)))

  model = RNN()
  criterion = nn.MSELoss()

  prev_state = torch.zeros([BATCH_SIZE, SEQ_LEN, N])
  h = torch.zeros([1, BATCH_SIZE, N])
  c = torch.zeros([1, BATCH_SIZE, N])


  for i, data in enumerate(training_dataloader, 0):
    inputs, labels = data
    # print("inputs shape")
    # print(inputs.shape)
    # print("labels shape")
    # print(labels.shape)


    

    model.zero_grad()

    output, (h, c) = model(inputs, (h, c))
    # print("output.shape")
    # print(output.shape)
    # print("hidden.shape")
    # print(hidden.shape)


    loss = criterion(output, labels)
    loss.backward(retain_graph=True)

    print("{} done".format(i))
    print("loss: {}".format(loss.item()))


    




