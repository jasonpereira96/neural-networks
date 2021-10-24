import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
# import pandas as pd
from torchvision import transforms
from torchvision.io import ImageReadMode
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

BATCH_SIZE = 64
TRAINING_SET_LENGTH = 8000
TEST_SET_LENGTH = 2000
WIDTH = 200
HEIGHT = 200
BLACK = np.uint8(0)
WHITE = np.uint8(255)
DATASET_PATH = 'geometry_dataset/output'
TRAINING_SET_PATH = 'geometry_dataset/train'
TEST_SET_PATH = 'geometry_dataset/test'
TRAINING_LABELS_PATH = 'geometry_dataset/training.file'
TEST_LABELS_PATH = 'geometry_dataset/testing.file'

classes = [
  'Circle',
  'Square',
  'Octagon',
  'Heptagon',
  'Nonagon',
  'Star',
  'Hexagon',
  'Pentagon',
  'Triangle'
]

random.seed(55)
def get_image_tensor(img):
  # print("called")
  return img[None :].float()

def get_label_tensor(filename):
  label = get_label_from_filename(filename)
  # tensor = torch.zeros([len(classes), 1])
  # tensor[classes.index(label)] = 1
  # return tensor.transpose(0, 1)
  # tensor = torch.zeros([1,1])
  # tensor[0][0] = classes.index(label)         # OPTIMIZE WITH A HASH TABLE
  # print(tensor.shape)
  return classes.index(label)
  # return tensor
  


def get_file_stem(filename):
  return Path(filename).stem

def build_bw_png(rgb_im, min_color, max_color):
  new_image_array = np.zeros((200, 200))

  for index in range(len(new_image_array)):
    new_image_array[index] = [0] * HEIGHT

  for row_index in range(WIDTH):
    for col_index in range(HEIGHT):
      r, g, b = rgb_im.getpixel((row_index, col_index))
      color = (r, g, b)
      if color == min_color:
        new_image_array[row_index][col_index] = BLACK
      else:
        new_image_array[row_index][col_index] = WHITE

  return new_image_array
  

def add_color(color, set):
  if color in set:
    set[color] += 1
  else:
    set[color] = 1

def convert_to_bw(input_filename, output_filename):

  color_counts = {}

  im = Image.open(input_filename)
  rgb_im = im.convert('RGB')
  # print(im.size)
  
  for row_index in range(WIDTH):
    for col_index in range(HEIGHT):
      r, g, b = rgb_im.getpixel((row_index, col_index))
      color = (r, g, b)
      add_color(color, color_counts)

  color1 = list(color_counts.keys())[0]
  color2 = list(color_counts.keys())[1]

  min_color = color1 if color_counts[color1] < color_counts[color2] else color2
  max_color = color1 if color_counts[color1] > color_counts[color2] else color2

  # print(color_counts)
  bw_image_array = build_bw_png(rgb_im, min_color, max_color)

  # print(bw_image_array)

  new_im = Image.fromarray(bw_image_array.transpose())
  new_im = new_im.convert('RGB')
  new_im.save(output_filename)
  # print("Conversion complete of {}".format(input_filename))

def generate_datasets():
  return False
  file_names = {
    'Circle': [],
    'Square': [],
    'Octagon': [],
    'Heptagon': [],
    'Nonagon': [],
    'Star': [],
    'Hexagon': [],
    'Pentagon': [],
    'Triangle': []
  }

  counter = 0

  all_file_names = os.listdir(DATASET_PATH)
  print(all_file_names)
  for file_name in all_file_names:
    if file_name.endswith('.png'):
      for start_word in list(file_names.keys()):
        if file_name.startswith(start_word):
          file_names[start_word].append(file_name)
  
  print(file_names)

  if not os.path.isdir(TRAINING_SET_PATH):
    os.mkdir(TRAINING_SET_PATH)

  if not os.path.isdir(TEST_SET_PATH):
    os.mkdir(TEST_SET_PATH)

  for shape in list(file_names.keys()):
    shape_file_names = file_names[shape]
    training_set = shape_file_names[:TRAINING_SET_LENGTH]
    test_set = shape_file_names[TRAINING_SET_LENGTH: TRAINING_SET_LENGTH + TEST_SET_LENGTH]


    for filename in training_set:
      stem = get_file_stem(filename=filename)
      source_filename = '{}/{}'.format(DATASET_PATH, filename)
      target_filename = '{}/{}_bw.png'.format(TRAINING_SET_PATH ,stem)
      
      if not os.path.exists(target_filename):
        convert_to_bw(source_filename, target_filename)

      counter += 1
      if counter % 100 == 0:
        print("{} files done.".format(counter))

    for filename in test_set:
      stem = get_file_stem(filename=filename)
      source_filename = '{}/{}'.format(DATASET_PATH,filename)
      target_filename = '{}/{}_bw.png'.format(TEST_SET_PATH,stem)

      if not os.path.exists(target_filename):
        convert_to_bw(source_filename, target_filename)
        
      counter += 1  
      if counter % 100 == 0:
        print("{} files done.".format(counter))


def get_label_from_filename(filename):
  return filename.split('_')[0]

class ImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    with open(annotations_file, 'r') as labels_file:
      self.img_labels = [j.strip() for j in labels_file.readlines()]

    # self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, index):
    img_path = os.path.join(self.img_dir, self.img_labels[index])
    # print(img_path)
    image = read_image(img_path, ImageReadMode.GRAY)
    label = self.img_labels[index]

    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(64*98*98, 128) # how do these numbers appear?
    self.fc2 = nn.Linear(128, 10)

  # x represents our data
  def forward(self, x):
    # Pass data through conv1
    x = self.conv1(x)
    # print(1)
    # print(x.shape)
    # Use the rectified-linear activation function over x
    x = F.relu(x)
    # print(2)
    # print(x.shape)


    x = self.conv2(x)
    x = F.relu(x)
    # print(3)
    # print(x.shape)

    # Run max pooling over x
    x = F.max_pool2d(x, 2)
    # Pass data through dropout1
    x = self.dropout1(x)
    # Flatten x with start_dim=1
    x = torch.flatten(x, 1)

    # print(4)
    # print(x.shape)
    # Pass data through fc1
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)

    # Apply softmax to x
    output = F.log_softmax(x, dim=1)
    return output

if __name__ == "__main__":
  # generate_datasets()

  training_set = ImageDataset(TRAINING_LABELS_PATH, TRAINING_SET_PATH, get_image_tensor, get_label_tensor)
  test_set = ImageDataset(TEST_LABELS_PATH, TEST_SET_LENGTH, get_image_tensor, get_label_tensor)

  training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False)
  test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

  # sample_order = list(range(0, TRAINING_SET_LENGTH))
  # random.shuffle(sample_order)

  
  image, label = training_set[0]
  print(label)
  image, label = training_set[4]
  print(label)
  image, label = training_set[14000]
  print(label)

  network = CNN()

  '''
  train_features, train_labels = next(iter(training_dataloader))
  print(f"Feature batch shape: {train_features.size()}")
  print(f"Labels length: {len(train_labels)}")
  img = train_features[2]
  print(img)
  label = train_labels[2]
  # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
  # plt.show()
  print(f"Label: {label}")
  print(get_label_tensor(label))

  # input_ = img
  # input_ = img[None, None, :].float()
  # i2 = torch.rand((1, 1, 28, 28))
  print(img.shape)
  # print(i2.shape)
  network = CNN()
  result = network(img)
  print (result)
  '''
  
  print ('+++++++++++++++++++++++++++++++++++++++++++++++++++')


  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_dataloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = network(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print('Finished Training')












# conv
# avg pooling
# conv
# relu
# conv
# dropout
# max pooling
# y
