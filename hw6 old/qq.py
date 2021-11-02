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
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from zipfile import ZipFile

def is_dataset_local():
  # return DATASET_LOCATION == 'local'
  return True
  # return DATASET_LOCATION == 'drive'

DEVICE = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(DEVICE))

DATASET_LOCATION = 'local'
MODEL_FILENAME = "model_weights_v2_2021-11-01 08 42 55.pth"
SAVE_MODEL = False
LOAD_WEIGHTS = True
BATCH_SIZE = 128
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
IMAGES_DIR = './'



# Google Drive data
DRIVE_ROOT = "drive/MyDrive/"

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

# if not is_dataset_local():
  # drive.mount('/content/drive')


def get_filenames():
  file_names = []
  all_file_names = os.listdir(IMAGES_DIR)

  for file_name in all_file_names:
    if file_name.endswith('.png'):
      file_names.append(file_name)

  print("{} PNG files to classify".format(len(file_names)))
  return file_names


def print_image(x):
  for i in range(200):
    line = ''
    for j in range(200):
      line = line + str(x[0][0][i][j].item())
    print(line)

# def save_checkpoint(network):
#   checkpoint = {'input_size': WIDTH * HEIGHT,
#               'output_size': 9,
#               'hidden_layers': [each.out_features for each in network.hidden_layers],
#               'state_dict': network.state_dict()}

#   torch.save(checkpoint, MODEL_FILENAME)

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = fc_model.Network(checkpoint['input_size'],
#                              checkpoint['output_size'],
#                              checkpoint['hidden_layers'])
#     model.load_state_dict(checkpoint['state_dict'])
    
#     return model
def get_current_timestamp():
  return str(datetime.datetime.now()).split('.')[0]

def get_image_tensor(img):
  # print("called")
  # return img
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

def get_label_from_index(index):
  return classes[index]

def get_output_label(output_tensor):
  # print("Output tensor:")
  # print(output_tensor)
  # max_value = output_tensor[0]
  # max_index = 0
  # for i, value in enumerate(output_tensor):
  #   if value > max_value:
  #     max_index = i

  max_index = max(range(len(output_tensor)), key=output_tensor.__getitem__)
  return classes[max_index]
  
def get_output_labels(output_tensors): # for a whole batch
  labels = []
  for output_tensor in output_tensors:
    labels.append(get_output_label(output_tensor))
  return labels
  

def get_file_stem(filename):
  return Path(filename).stem

def build_bw_png(rgb_im, min_color, max_color):
  new_image_array = np.zeros((200, 200))
  # print("in bbp")
  # print("{} {}".format(min_color, max_color))

  # for index in range(len(new_image_array)):
  #   new_image_array[index] = [0] * HEIGHT

  for row_index in range(WIDTH):
    for col_index in range(HEIGHT):
      r, g, b = rgb_im.getpixel((row_index, col_index))
      color = (r, g, b)
      if color == min_color:
        new_image_array[row_index][col_index] = BLACK
      else:
        new_image_array[row_index][col_index] = WHITE
  
  # print("nia")
  # print(new_image_array)

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

  # print('color_counts')
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
  def __init__(self, mode="test" ,transform=None, target_transform=None):
    self.mode = mode

    # self.zip_file = ZipFile('drive/MyDrive/NN/hw6/data/hw6.zip')

    print("Pre-processing images...")
    # create a ZipFile object

    count = 0

    self.zipped_filenames = []

    zip_ = ZipFile('temp.zip', 'w')
    # zip_ = ZipFile('temp.zip', 'r')

    for filename in get_filenames():
      new_filename = get_file_stem(filename) + '_bw.png'
      convert_to_bw(os.path.join(IMAGES_DIR, filename), new_filename)
      # zip_.write(filename, new_filename)
      zip_.write(new_filename)

      self.zipped_filenames.append(new_filename)
      count += 1

      if count % 100 == 0:
        print("{} files pre-processed.".format(count))
    # Add multiple files to the zip

    # close the Zip File
    # zip_.close()

    self.zip_file = zip_

    print("Processing images done.")


    # self.img_labels = [j.strip() for j in labels_file.readlines()]
    # self.img_labels = list(map(lambda b_label: b_label.decode("utf-8"), self.img_labels))
    # print(self.img_labels)

    self.N = len(self.zipped_filenames)

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    # return 1

    return self.N

  def __getitem__(self, index):
    # print("__getitem__({})".format(index))
    # if is_dataset_local():
    #   img_path = os.path.join(self.img_dir, self.img_labels[index])
    #   image = read_image(img_path, ImageReadMode.GRAY)
    # else:
    # print(img_path)
    if self.mode == "test":
      pass
      # img_path = self.img_labels[index]
    # else:
      # img_path = 'hw6/geometry_dataset/train/' + self.img_labels[index]
    # img_path = 'hw6/geometry_dataset/test/' + self.img_labels[index] if self.mode == "test" else 'hw6/geometry_dataset/train/' + self.img_labels[index]
    img_path = self.zipped_filenames[index]
    # img_path = 'hw6/geometry_dataset/train/' + self.img_labels[index]

    image_file = self.zip_file.open(img_path)
    # image_file = self.zip_file.open('Triangle_ffa8bf6c-2a83-11ea-8123-8363a7ec19e6_bw.png')

    
    # image = read_image(img_path, ImageReadMode.GRAY)
    image = mpimg.imread(image_file)
    # print("B")
    
    # print(image[0][0])
    image = np.delete(image,[1,2], 2)
    image = np.swapaxes(image, 0, 2)
    image = torch.from_numpy(image).to(DEVICE)
    # print("C")
    # print(image.shape)

    
    # label = self.img_labels[index]

    if self.transform:
      image = self.transform(image)
    # if self.target_transform:
      # label = self.target_transform(label)

    # print(self.zipped_filenames[index])

    return image, self.zipped_filenames[index].split('_')[0]


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, 1)
    self.conv2 = nn.Conv2d(16, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    # self.dropout2 = nn.Dropout2d(0.5)
    # self.fc1 = nn.Linear(64*98*98, 128) # 64 filters * (each filter produces an output of 98*)
    self.fc1 = nn.Linear(64*98*98, 128) # 64 filters * (each filter produces an output of 98*)
    self.fc2 = nn.Linear(128, 9)

  # x represents our data
  def forward(self, x):
    # Pass data through conv1
    # print("A")
    # print(x.shape)
    # print(x)

    # print_image(x)

    x = self.conv1(x)
    # print(1)
    # print(x.shape)
    # Use the rectified-linear activation function over x
    x = F.relu(x)
    # print(2)
    # print(x.shape)


    x = self.conv2(x)
    # print(x.shape)

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
    # x = self.dropout2(x)
    x = self.fc2(x)

    # Apply softmax to x
    output = F.log_softmax(x, dim=1)
    return output


if __name__ == "__main__":
  # generate_datasets()

  # training_set = ImageDataset("train", get_image_tensor, get_label_tensor)
  test_set = ImageDataset("test", get_image_tensor, get_label_tensor)

  # training_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

  # sample_order = list(range(0, TRAINING_SET_LENGTH))
  # random.shuffle(sample_order)

  
  # image, label = training_set[0]
  # print(label)
  # # plt.imshow(image)
  # # plt.show()
  # image, label = training_set[4]
  # print(label)
  # image, label = training_set[14000]
  # print(label)

  network = CNN()
  if LOAD_WEIGHTS:
    print("Loading weights from {}".format(MODEL_FILENAME))
    network.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device('cpu')))
    print("Loaded weights from {}".format(MODEL_FILENAME))

  network.eval()
  network.to(DEVICE)
  
  print ('+++++++++++++++++++++++++++++++++++++++++++++++++++')


  # criterion = nn.CrossEntropyLoss()
  criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
  optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)


  def test():
    matches = 0
    total = 0
    for i, data in enumerate(test_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print("F")
        # print(type(labels))

        # print("H")
        # print(labels)
        

        
        # forward + backward + optimize
        outputs = network(inputs)
        # print("X")
        # print(type(outputs))
        # print(outputs)


        output_labels = get_output_labels(outputs)
        # print("G")
        # print(type(output_labels))
        # print(output_labels)

        
        for index in range(1):
          expected_output = labels[index]
          actual_output = output_labels[index]
          is_match = actual_output == expected_output

          if is_match:
            matches +=1
          total += 1

          print("Label: {} | Output label: {} | running acc: {}".format(expected_output, actual_output, (matches*100)/total))
          # print(outputs)

        # print(inputs[0])

    # network.zip_file.close()


  

  def train():
    print('Starting Training')

    # training loop
    for epoch in range(25):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(training_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)

        if torch.cuda.is_available():
          loss = criterion(outputs.cuda(), labels.cuda())
        else:
          loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
          print("Epoch: {} running loss: {}".format(epoch, running_loss / 2000))
          running_loss = 0.0

          # if SAVE_MODEL:
          #   torch.save(network.state_dict(), MODEL_FILENAME)
          #   print("Model saved to {}".format(MODEL_FILENAME))

          print("Finished batch: {}".format(i))

      if SAVE_MODEL:
        print("Saving model...")
        # model_filename = MODEL_FILENAME + '_' + get_current_timestamp()
        model_filename = 'drive/MyDrive/NN/hw6/model/model_weights_v2_' + get_current_timestamp() + '.pth'
        torch.save(network.state_dict(), model_filename)
        print("Model saved to {}".format(model_filename))

    print('Finished Training')

  test()
  # train()



'''
Ideas

load the entire dataset in memory
use a hash table to optimize get label
Cache cuda.is_available()
https://medium.com/dair-ai/pytorch-1-2-quickstart-with-google-colab-6690a30c38d
https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

'''







# conv
# avg pooling
# conv
# relu
# conv
# dropout
# max pooling
# y
