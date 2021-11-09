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

EPOCHS = 5000
SAVE_WEIGHTS = True
MODEL_FILENAME = "model_v2_2021-11-08 01_37_07.337719.pth" # load the model from this file
LOAD_WEIGHTS = False
HAS_CUDA = torch.cuda.is_available()
ETA = 0.001
EON = '<EON>'
N = 27
SEQ_LEN = 11
BATCH_SIZE = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))


def word2tensor(word, from_index, to_index):
  length = to_index - from_index + 1
  tensor = torch.zeros((1, length, N), device=device)
  # print(word)
  for i in range(from_index, to_index + 1):
    # ch = word[i] if i < len(word) else EON
    ch_index = (ord(word[i]) - 97 + 1) if i < len(word) else 0 # EON has index 0
    tensor[0][i][ch_index] = 1
  
  return tensor


def compress_labels(labels): # Tensor of: [BATCH_SIZE, v, 27]
  v = labels.size(dim=1)
  compressed = torch.zeros((BATCH_SIZE, v), dtype=torch.long, device=device)
  for batch_index in range(BATCH_SIZE):
    for ch_index in range(v):
      compressed[batch_index][ch_index] = torch.argmax(labels[batch_index][ch_index])
  return compressed

def get_model_filename():
  return "model_v2_" + str(datetime.datetime.now()) + '.pth'

def load_weights(model):
  print("Loading weights from {}".format(MODEL_FILENAME))
  model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device('cpu')))
  print("Loaded weights from {}".format(MODEL_FILENAME))

def save_model(model):
  print("Saving model...")
  torch.save(model.state_dict(), get_model_filename())
  print("Model saved.")

def get_letter(tensor, randomize=False): # tensor should be (1,1,27)
  if randomize:
    index = torch.topk(tensor[0][0], 3)[1][1].item()
  else:
    index = torch.argmax(tensor[0][0]).item()

  if index == 0:
    return EON
  return chr(index + 97 - 1)
  

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

  tensor = torch.zeros((SEQ_LEN, N), device=device)

  for index in range(SEQ_LEN):
    ch = name[index] if index < len(name) else EON
    tensor[index][get_index_value(ch)] = 1

  return tensor

def get_char_from_tensor(tensor):
  max_index = torch.argmax(tensor).item()
  if max_index == 0:
    return EON
  else:
    return chr(max_index + 97 - 1)

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
        tensor = torch.zeros([N], device=device)
        if i < len(name):
          tensor[get_index_value(name[i])] = 1
        else:
          tensor[get_index_value(EON)] = 1
        self.char_tensors.append(tensor)


  def __len__(self):
    # return 1
    return len(self.names)

  def __getitem__old(self, index):
    if index == len(self.char_tensors) - 1:
      return self.char_tensors[index], EON # change this

    return self.char_tensors[index], self.char_tensors[index + 1]
  
  def __getitem_2_(self, index):
    name = self.names[index]
    return name_to_2d_tensor(name).to(device), name_to_2d_tensor(name, shift=True).to(device)

  def __getitem__(self, index):
    name = self.names[index]
    return name, name[1:]


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

def train_epoch(model):
  prev_state = torch.zeros([BATCH_SIZE, SEQ_LEN, N], device=device)
  h = torch.zeros([1, BATCH_SIZE, N], device=device)
  c = torch.zeros([1, BATCH_SIZE, N], device=device)

  model.zero_grad()
  loss = 0


  for i, data in enumerate(training_dataloader, 0):
    inputs, labels = data
    # print(type(inputs))
    # print(type(labels))

    # print(len(inputs))
    # print(len(labels))
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

    # h = h.0to(device)
    # c = c.to(device)
    word = inputs[0]

    for word_end_index in range(SEQ_LEN):
      # print("inputs")
      # print(inputs)
      # print("labels")
      # print(labels)
      if word_end_index == len(word) + 1:
        break
    

      vinput = word2tensor(inputs[0], 0, word_end_index)
      vlabel = word2tensor(labels[0], 0, word_end_index)

      # print("vinput.shape")
      # print(vinput.shape)
      # print("vlabel.shape")
      # print(vlabel.shape)

      output, (h, c) = model(vinput, (h, c))

      # print("output.shape")
      # print(output.shape)
      # print("output.shape")
      # print(output.shape)
      # print("hidden.shape")
      # print(hidden.shape)
      # print("labels.shape")
      # print(labels.shape)
      # print("vinput.shape")
      # print(vinput.shape)
      # print("vinput")
      # print(str_from_3d_tensor(vinput))

      # print("vlabel.shape")
      # print(vlabel.shape)
      # print("vlabel")
      # print(str_from_3d_tensor(vlabel))

      # print("output")
      # print(str_from_3d_tensor(output))


      
      # convert the labels into the format required by cross entropy loss
      compressed = compress_labels(vlabel) 
      # print(compressed.shape)

      if HAS_CUDA:
        # l = criterion(output.cuda(), .cuda())
        l = criterion(torch.transpose(output, 1, 2).cuda(), compressed.cuda())
      else:
        l = criterion(torch.transpose(output, 1, 2), compressed)

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

  return output, loss.item() / (2000 / BATCH_SIZE)

def train(model):
  print("Starting training")
  total_loss = 0
  for epoch in range(EPOCHS): # 2 epochs
    output, loss = train_epoch(model)
    total_loss += loss

    if epoch % 20 == 0:
      print('Epoch: {} Loss: {}'.format(epoch, loss))

    if epoch % 50 == 0 and epoch != 0:
      save_model(model)

def test(model):
  h = torch.zeros([1, 1, N], device=device)
  c = torch.zeros([1, 1, N], device=device)


  for letter_index in range(N):
    inputs = torch.zeros([1, 1, N], device=device)
    inputs[0][0][letter_index] = 1
    starting_letter = get_letter(inputs, randomize=False)
    print("Starting letter: {}".format(starting_letter))
    generated_name = starting_letter

    for i in range(SEQ_LEN):

      # print("inputs.shape")
      # print(inputs.shape)
      # print("inputs")
      # print(str_from_3d_tensor(inputs))


      output, (h, c) = model(inputs, (h, c))
      # print("Output")
      # print("output.shape")
      # print(output.shape)
      # print(output)
      indices = torch.tensor([i], device=device)
      last_letter_tensor = torch.index_select(output, 1, indices)

      inputs = torch.cat((inputs, last_letter_tensor), 1)
      # print(output[0][0])
      # print(torch.argmax(output[0][0]).item())

      letter = get_letter(last_letter_tensor)
    
      generated_name = generated_name + letter
      
      if letter == EON:
        pass
        # break

    print("Generated name:")
    print(generated_name)
    # print("Length of generated name: {}".format(len(generated_name)))


if __name__ == "__main__":
  
  training_set = NamesDataset(train=True)
#   test_set = ImageDataset("test", get_image_tensor, get_label_tensor)

  training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
#   test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

  # for i in range(33):
    # print(get_char_from_tensor(training_set.__getitem__(i)))

  model = RNN()

  if LOAD_WEIGHTS:
    load_weights(model)
  else:
    print("Initialzing with random weights.")

  model.to(device)

  # criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss() # handle for cuda as well
  criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss() # handle for cuda as well

  optimizer = optim.Adam(model.parameters(), lr=ETA)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
  # model.

  train(model)
  # test(model)

  


  

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


