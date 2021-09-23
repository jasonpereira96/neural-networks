import gzip
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists

# Note:
# The gzip files must be present in the same directory for this script to run

FILENAME = 'weights.npy'
IMAGE_SIZE = 28
DIMENSIONS = IMAGE_SIZE * IMAGE_SIZE

# Change the values of these parameters according to what is required
N = 1000
test_N = 10000
convergence_margin = 0
eta = 0.7 # learning rate
EPSILON = 0

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

def get_weights_diff(W1, W2):
  diffs = []
  for row_index in range(10):
    for col_index in range(DIMENSIONS):
      delta = W1[row_index][col_index] - W2[row_index][col_index]
      if delta != 0:
        diffs.append(delta)
  return diffs

# function to print an image on to the console
def print_image(image):
  for row in range(28):
    line = ''
    for col in range(28):
      pixel = image[row][col]
      if pixel != 0:
        line = line + '@'
      else:
        line = line + '.'
    print(line)

# function to save the weights to a file for future use
def save_weights(W):
  print('saving weights to file')
  np.save(FILENAME, W)

# function to initialize W. If from_file is True, the we try to load the weights from a file
def get_initial_weights(from_file=False):
  if from_file and exists(FILENAME):
    print('loading initial weights from file')
    return np.load(FILENAME)

  weights = np.zeros((10, DIMENSIONS), dtype=float)
  for row_index in range(10):
    for col_index in range(DIMENSIONS):
      weights[row_index][col_index] = np.random.uniform(-1, 1)

  return weights

# functions to read data from the gzip files. The gzip files must be present in the same directory
def read_label_data():
  f = gzip.open('train-labels-idx1-ubyte.gz','r')
  # reading the # magic number, number of labels
  a = f.read(8)
  training_labels = []
  for i in range(N):
    label = int.from_bytes(f.read(1), signed=False, byteorder='big')
    training_labels.append(label)

  f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  # reading the # magic number, number of labels
  a = f.read(8)
  test_labels = []
  for i in range(test_N):
    label = int.from_bytes(f.read(1), signed=False, byteorder='big')
    test_labels.append(label)
  
  return training_labels, test_labels
  
def read_image_data():
  f = gzip.open('train-images-idx3-ubyte.gz','r')
  # reading the # magic number, number of images, rows, columns
  a = f.read(16)
  buf = f.read(IMAGE_SIZE * IMAGE_SIZE * N)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  training_images = data.reshape(N, IMAGE_SIZE, IMAGE_SIZE)

  f = gzip.open('t10k-images-idx3-ubyte.gz','r')
  # reading the # magic number, number of images, rows, columns
  a = f.read(16)
  buf = f.read(IMAGE_SIZE * IMAGE_SIZE * test_N)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  test_images = data.reshape(test_N, IMAGE_SIZE, IMAGE_SIZE)
  return training_images, test_images
  
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
    self.W = get_initial_weights(from_file=False)
    self.errors_per_epoch = []
    self.epoch = 0
  
  def update_weights(self):
    for index in range(N):
      image = self.training_images[index]
      X = vectorize(image)
      desired_output_digit = self.training_labels[index]
      desired_output = get_desired_output_matrix(desired_output_digit)
      
      WX = np.matmul(self.W, X)

      self.W = self.W + eta * np.matmul((desired_output - step(WX)), np.transpose(X))

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

  def test(self):
    errors = 0
    for index in range(test_N):
      X = vectorize(self.test_images[index])
      V = np.matmul(self.W, X)
      # print(V)
      actual_output = argmax(V)
      desired_output = self.test_labels[index]

      if desired_output != actual_output:
        errors = errors + 1

    error_perc = (errors / test_N) * 100
    print('Percentage of misclassified samples = ' + str(error_perc) + '%')
    return errors  


  def train(self):
    print('initial W')
    print(self.W)
    print('------------------')

    while True:
      errors = self.run_epoch()
      self.errors_per_epoch.append(errors)
      self.epoch = self.epoch + 1

      if (errors / self.epoch) <= eta:
        print('Training complete')
        print('Errors per epoch:')
        print(self.errors_per_epoch)
        # save_weights(self.W)
        break
      else:
        self.update_weights()

      if self.epoch % 10 == 0:
        print('Epoch: ' + str(self.epoch))
        print('errors: ' + str(errors))
        # saving the weights to file periodically
        # save_weights(self.W)

  def plot_errors_vs_epoch(self):
    plt.clf()
    number_of_epochs = len(self.errors_per_epoch)
    x = list(range(number_of_epochs))
    y = self.errors_per_epoch
    plt.plot(x, y, label="Misclassifications per epoch")
    plt.suptitle('Misclassifications per epoch (\u03B7 = ' + str(eta) + ' and N = ' + str(N) + ' and \u0190 = ' + str(EPSILON) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Misclassifications')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
  training_images, test_images, training_labels, test_labels = read_data()
  hw3 = HW3(training_images, test_images, training_labels, test_labels)
  hw3.train()
  hw3.plot_errors_vs_epoch()
  hw3.test()