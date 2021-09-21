import gzip
import numpy as np
import matplotlib.pyplot as plt

image_size = 28
DIMENSIONS = image_size * image_size
N = 5

def step(x):
  return 1 if x >= 0 else 0

def argmax(V):
  return np.argmax(V)

def vectorize(image):
  return image.reshape(DIMENSIONS, 1)

def get_desired_output_matrix(digit):
  matrix = np.zeros((10, 1))
  matrix[digit] = 1
  return matrix

def read_label_data():
  f = gzip.open('train-labels-idx1-ubyte.gz','r')
  # reading the # magic number,  number of labels
  a = f.read(8)
  training_labels = []
  for i in range(N):
    label = int.from_bytes(f.read(1), signed=False, byteorder='big')
    training_labels.append(label)
  # print(training_labels)

  f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  # reading the # magic number,  number of labels
  a = f.read(8)
  test_labels = []
  for i in range(N):
    label = int.from_bytes(f.read(1), signed=False, byteorder='big')
    test_labels.append(label)

  return training_labels, test_labels
  
def read_image_data():
  f = gzip.open('train-images-idx3-ubyte.gz','r')
  # reading the # magic number,  number of images,  rows , columns
  a = f.read(16)
  buf = f.read(image_size * image_size * N)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  training_images = data.reshape(N, image_size, image_size)
  image = np.asarray(training_images[4]).squeeze()
  plt.imshow(image)
  # plt.show()
  f = gzip.open('t10k-images-idx3-ubyte.gz','r')
  # reading the # magic number,  number of images,  rows , columns
  a = f.read(16)
  buf = f.read(image_size * image_size * N)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  test_images = data.reshape(N, image_size, image_size)
  image = np.asarray(test_images[3]).squeeze()
  # plt.imshow(image)
  return training_images, test_images
  

# add a case if gzip files are not present
# change to prevent plagarism

def read_data():
  training_images, test_images = read_image_data()
  training_labels, test_labels = read_label_data()
  return training_images, test_images, training_labels, test_labels


class HW3:
  def __init__(self, training_images, test_images, training_labels, test_labels):
    self.training_images = training_images
    self.test_images = test_images
    self.training_labels = training_labels
    self.test_labels = test_labels
    self.W = np.random.rand(10, DIMENSIONS)
    self.errors_per_epoch = []
    self.epoch = 0
  
  def update_weights(self):
    W = self.W

    for index in range(1):
      X = vectorize(self.training_images[index])
      desired_output_digit = self.training_labels[index].reshape()
      desired_output = get_desired_output_matrix(desired_output_digit)
      
      W = W + eta * (desired_output - step(matmul(W, X)))

  def run_epoch(self):
    errors = 0
    for index in range(1):
      X = vectorize(self.training_images[index])
      W = self.W
      V = np.matmul(W, X)
      print(V)
      actual_output = argmax(V)
      desired_output = argmax(self.training_labels[index])
      delta = desired_output - actual_output

      if delta != 0:
        errors = errors + 1

    return errors
  
    # image = training_images[2]

    # for row in range(28):
    #   line = ''
    #   for col in range(28):
    #     pixel = image[row][col]
    #     if pixel != 0:
    #       line = line + '@'
    #     else:
    #       line = line + '.'
    #   print(line)

  def train(self):
    count = 1

    while count > 0: # True
      errors = self.run_epoch()
      self.errors_per_epoch.append(errors)
      self.epoch = self.epoch + 1

      if errors == 0:
        print('training complete')
        break
      else:
        self.update_weights()

      count = 0

training_images, test_images, training_labels, test_labels = read_data()
hw3 = HW3(training_images, test_images, training_labels, test_labels)
hw3.train()