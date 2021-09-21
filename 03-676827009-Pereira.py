import gzip
import numpy as np
import matplotlib.pyplot as plt

image_size = 28
DIMENSIONS = image_size * image_size
N = 50
eta =  1 # learning rate
EPSILON = 0.01

def step(x):
  u = lambda x: 1 if x >= 0 else 0
  if isinstance(x, int):
    return u(x)
  return np.vectorize(u)(x)

def argmax(V):
  return np.argmax(V)

def vectorize(image):
  return image.reshape(DIMENSIONS, 1)

def get_desired_output_matrix(digit):
  matrix = np.zeros((10, 1))
  matrix[digit] = 1
  return matrix

def print_delta(delta):
  over_zero = []
  for x in np.nditer(delta):
    if x > 0:
      over_zero.append(x)
  print(over_zero)

def get_weights_diff(W1, W2):
  diffs = []
  for row_index in range(10):
    for col_index in range(DIMENSIONS):
      delta = W1[row_index][col_index] - W2[row_index][col_index]
      if delta != 0:
        diffs.append(delta)
  return diffs

def print_image(image):
  # image = training_images[2]
  for row in range(28):
    line = ''
    for col in range(28):
      pixel = image[row][col]
      if pixel != 0:
        line = line + '@'
      else:
        line = line + '.'
    print(line)

def get_initial_weights():
  # return np.random.rand(10, DIMENSIONS)
  weights = np.zeros((10, DIMENSIONS), dtype=float)

  for row_index in range(10):
    for col_index in range(DIMENSIONS):
      weights[row_index][col_index] = np.random.uniform(-1, 1)

  return weights

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
    # self.W = np.random.rand(10, DIMENSIONS)
    self.W = get_initial_weights()
    self.errors_per_epoch = []
    self.epoch = 0
  
  def update_weights(self):
    for index in range(N):
      old_weights = np.copy(self.W)
      image = self.training_images[index]
      X = vectorize(image)
      desired_output_digit = self.training_labels[index]
      desired_output = get_desired_output_matrix(desired_output_digit)
      # print_image(image)
      
      WX = np.matmul(self.W, X)
      delta = eta * np.matmul((desired_output - step(WX)), np.transpose(X))
      # print('delta')
      # print_delta(delta)
      self.W = self.W + eta * np.matmul((desired_output - step(WX)), np.transpose(X))

      print(get_weights_diff(old_weights, self.W))

  def run_epoch(self):
    errors = 0
    for index in range(N):
      X = vectorize(self.training_images[index])
      V = np.matmul(self.W, X)
      # print(V)
      actual_output = argmax(V)
      desired_output = self.training_labels[index]

      if desired_output != actual_output:
        errors = errors + 1

    return errors  

  def train(self):
    print('initial W')
    print(self.W)
    print('------------------')

    while True: # True
      errors = self.run_epoch()
      self.errors_per_epoch.append(errors)
      self.epoch = self.epoch + 1

      if (errors / N) < EPSILON:
        print('training complete')
        break
      else:
        self.update_weights()

      if self.epoch % 10 == 0:
        print(self.epoch)
        # print(self.W)
        print('errors: ' + str(errors))

training_images, test_images, training_labels, test_labels = read_data()
hw3 = HW3(training_images, test_images, training_labels, test_labels)
hw3.train()