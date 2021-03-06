import torch
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(False)

# good model: model_v3_2021-11-09 03_39_08.650717
TEST = True
EPOCHS = 1000 
SAVE_WEIGHTS = True
# MODEL_FILENAME = "/home/jason/Repos/NN/hw7/" + "model_v3_2021-11-09 03_39_08.650717.pth" # load the model from this file
MODEL_FILENAME = "0702-676827009-Pereira.pth"
LOAD_WEIGHTS = False
HAS_CUDA = torch.cuda.is_available()
ETA = 0.001
EON = '<EON>'
N = 27
SEQ_LEN = 11
BATCH_SIZE = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: {}".format(device))


def get_max_index(tensor, randomize=False): # (,27)
  max_index = torch.argmax(tensor)
  max_values, max_indices = torch.topk(tensor, 3)
  # print(max_indices)
  if not randomize:
    return max_indices[0][0][0]
  else:
    random_number = random.random()
    if 0 < random_number <= 0.3:
      return max_indices[0][0][0]
    elif 0.3 < random_number <= 0.6:
      return max_indices[0][0][1]
    else:
      return max_indices[0][0][2]

def maxify(tensor, max_index): # (,27)
  ret = torch.zeros(N, device=device)
  ret[max_index] = 1
  return ret


def word2tensor(word, from_index, to_index):
  length = to_index - from_index + 1
  tensor = torch.zeros((1, length, N), device=device)
  # print(word)
  for i in range(from_index, to_index + 1):
    ch_index = (ord(word[i]) - 97 + 1) if i < len(word) else 0 # EON has index 0
    tensor[0][i][ch_index] = 1
  
  return tensor

def word2tensor_full(word):
  tensor = torch.zeros(size=(1, SEQ_LEN, N), device=device)
  # print(word)
  for i in range(SEQ_LEN):
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
  return "model_v4_" + str(datetime.datetime.now()) + '.pth'

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

    f = open('/home/jason/Repos/NN/hw7/names.txt', 'r')
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
    # return 5
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

  def forward(self, x, prev_state):
    h, c = prev_state
  
    output, (hn, cn) = self.lstm(x, (h, c))

    output = self.i2o(output)

    return output, (hn, cn)

def train_epoch(model):
  prev_state = torch.zeros([BATCH_SIZE, SEQ_LEN, N], device=device)
  h = torch.zeros([1, BATCH_SIZE, N], device=device)
  c = torch.zeros([1, BATCH_SIZE, N], device=device)

  model.zero_grad()
  loss = 0

  for i, data in enumerate(training_dataloader, 0):
    inputs, labels = data
  
    word = inputs[0]

    vinput = word2tensor_full(inputs[0])
    vlabel = word2tensor_full(labels[0])

  
    output, (h, c) = model(vinput, (h, c))



    
    # convert the labels into the format required by cross entropy loss
    compressed = compress_labels(vlabel) 

    if HAS_CUDA:
      l = criterion(torch.transpose(output, 1, 2).cuda(), compressed.cuda())
    else:
      l = criterion(torch.transpose(output, 1, 2), compressed)


    loss = loss + l


  # optimizer.step() # throwing an error
  # scheduler.step()

    if i % 100 == 0:
      pass
     

  loss.backward()

  for p in model.parameters():
    p.data.add_(p.grad.data, alpha=-ETA)

  return output, loss.item() / (2000 / BATCH_SIZE)

def train(model):
  print("Starting training")
  print(model)
  total_loss = 0
  for epoch in range(EPOCHS): 
    output, loss = train_epoch(model)
    total_loss += loss

    if epoch % 1 == 0:
      print('Epoch: {} Loss: {}'.format(epoch, loss))

    if epoch % 20 == 0 and epoch != 0:
      save_model(model)

def test(model, letter_index):
  h = torch.zeros([1, 1, N], device=device)
  c = torch.zeros([1, 1, N], device=device)

  max_index = letter_index
  inputs = torch.zeros([1, 0, N], device=device)
  starting_letter = EON if letter_index == 0 else chr(letter_index + 97 - 1)
  print("Starting letter: {}".format(starting_letter))
  generated_name = starting_letter

  for i in range(SEQ_LEN-1):

    inputs = torch.cat((inputs, torch.zeros([1,1,N])), dim=1)
    inputs[0][-1][max_index] = 1




    output, (h, c) = model(inputs, (h, c))
    indices = torch.tensor([i+1], device=device)
    last_letter_tensor = output[0][-1].unsqueeze(0).unsqueeze(0)



    max_index = get_max_index(last_letter_tensor, randomize=True).item()
    letter = EON if max_index == 0 else chr(max_index + 97 - 1)
  
   
    
    if letter == EON:
      # pass
      break

    generated_name = generated_name + letter

  print(generated_name)


if __name__ == "__main__":
  
  training_set = NamesDataset(train=True)

  training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)


  model = RNN()

  if LOAD_WEIGHTS:
    load_weights(model)
  else:
    print("Initialzing with random weights.")

  model.to(device)

  criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss() # handle for cuda as well

  optimizer = optim.Adam(model.parameters(), lr=ETA)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
  # model.

  
  if TEST:
    while True:
      user_input = input("Please enter a starting letter: ")

      if len(user_input) != 1:
        continue

      if not user_input.isalpha():
        continue
      letter_index = ord(user_input.lower()) - 97 + 1
      for i in range(20):
        test(model, letter_index)
  else:
    train(model)


# References:
# 1) https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html