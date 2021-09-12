import random
import numpy as np
from string import Template
import matplotlib.pyplot as plt

# we define u(x) = 1 if x >= 0, and otherwise, we define u(x) = 0 if x < 0
def step(x):
  return 1 if x >= 0 else 0

def initialize(N):
  w0 = random.uniform(-0.25, 0.25)
  w1 = random.uniform(-1, 1)
  w2 = random.uniform(-1, 1)

  X = np.zeros(shape=(N, 2))

  for row_index in range(N):
    for col_index in range(2):
      X[row_index][col_index] = random.uniform(-1, 1)

  return w0, w1, w2, X

def plot_initial_graph():
  S0 = [] # (e)
  S1 = [] # (f)

  for point in X:
    x1, x2 = point

    if w1*x1 + w2*x2 + w0 >= 0:
      S1.append(point)
    else:
      S0.append(point)
      

  x = list(map(lambda point : point[0], S0))
  y = list(map(lambda point : point[1], S0))
  plt.scatter(x, y, alpha=0.5, label="$S_0$")

  x = list(map(lambda point : point[0], S1))
  y = list(map(lambda point : point[1], S1))
  plt.scatter(x, y, alpha=0.5, label="$S_1$")

  x = np.linspace(-1, 1, 100)
  y = (-w0 - w1*x) / w2
  plt.plot(x, y, label="Boundary")
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.suptitle('Graphical intepretation of a perceptron')
  plt.legend()
  plt.show() # (g)

w0_ = random.uniform(-1, 1)
w1_ = random.uniform(-1, 1)
w2_ = random.uniform(-1, 1)

initial_weights = [w0_, w1_, w2_].copy()


def get_num_of_misclassifications(correct_weights, actual_weights, points):
  w0, w1, w2 = correct_weights
  w0_, w1_, w2_ = actual_weights

  misclassifications = 0
  correct_classifications = 0
  count = 1

  for point in points:
    x1, x2 = point
    
    c_desired = w0 + w1*x1 + w2*x2
    c_real = w0_ + w1_*x1 + w2_*x2

    both_positive = (c_desired >= 0) and (c_real >= 0)
    both_negative = (c_desired < 0) and (c_real < 0)


    if both_positive or both_negative:
      correct_classifications = correct_classifications + 1
    else:
      misclassifications = misclassifications + 1

    p1 = 'positive' if c_desired >= 0 else 'negative'
    p2 = 'positive' if c_real >= 0 else 'negative'

    if both_positive or both_negative:
      p3 = ''
    else:
      p3 = 'misclassification'

    t = Template('$p1 | $p2 | $p3')

    # print(str(count) + ' ' + t.substitute(p1=p1, p2=p2, p3=p3))
    count = count + 1

  return misclassifications

def train():
  global w0_, w1_, w2_, w0, w1, w2
  # w0, w1, w2 = correct_weights
  # w0_, w1_, w2_ = actual_weights
  epoch = 0
  misclassifications_per_epoch = []

  while True:
    misclassifications = 0
    for point in X:
      x1, x2 = point

      desired_output = step(w0 + w1*x1 + w2*x2)
      actual_output = step(w0_ + w1_*x1 + w2_*x2)

      if actual_output != desired_output:
        w0_ = w0_ + eta * 1 * (desired_output - actual_output)
        w1_ = w1_ + eta * x1 * (desired_output - actual_output)
        w2_ = w2_ + eta * x2 * (desired_output - actual_output)
        misclassifications = misclassifications + 1

    if misclassifications == 0:
      misclassifications_per_epoch.append(0)
      break
    
    misclassifications_per_epoch.append(misclassifications)
    epoch = epoch + 1

  return misclassifications_per_epoch

def show_misclassifications_graph(misclassifications_per_epoch):
  plt.clf()
  number_of_epochs = len(misclassifications_per_epoch)
  x = list(range(number_of_epochs))
  y = misclassifications_per_epoch
  plt.plot(x, y, label="Misclassifications per epoch")
  plt.suptitle('Misclassifications per epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Misclassifications')
  plt.legend()
  plt.show()

def print_weights_comparison():
  print('initial weights: ' + str(initial_weights))
  print('final weights: ' + str([w0_, w1_, w2_]))
  print('Difference between initial and final w0: ' + str(abs(initial_weights[0] - w0_)))
  print('Difference between initial and final w1: ' + str(abs(initial_weights[1] - w1_)))
  print('Difference between initial and final w2: ' + str(abs(initial_weights[2] - w2_)))


N = 100

w0, w1, w2, X = initialize(N)
plot_initial_graph()

eta = 1 # i
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)

print_weights_comparison()

w0_, w1_, w2_ = initial_weights
eta = 10
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)


w0_, w1_, w2_ = initial_weights
eta = 0.1
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)

N = 1000
w0, w1, w2, X = initialize(N)
plot_initial_graph()

eta = 1 # i
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)

w0_, w1_, w2_ = initial_weights
eta = 10
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)


w0_, w1_, w2_ = initial_weights
eta = 0.1
misclassifications_per_epoch = train()
show_misclassifications_graph(misclassifications_per_epoch)