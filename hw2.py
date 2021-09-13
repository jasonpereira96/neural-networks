import random
import numpy as np
from string import Template
import matplotlib.pyplot as plt

# we define u(x) = 1 if x >= 0, and otherwise, we define u(x) = 0 if x < 0
def step(x):
  return 1 if x >= 0 else 0

def initialize(N):
  # picking optimal weights at random
  w0 = random.uniform(-0.25, 0.25) # (a)
  w1 = random.uniform(-1, 1) # (b)
  w2 = random.uniform(-1, 1) # (c)

  X = np.zeros(shape=(N, 2)) # (d)

  for row_index in range(N):
    for col_index in range(2):
      X[row_index][col_index] = random.uniform(-1, 1) # creating the dataset

  return w0, w1, w2, X

def plot_initial_graph(w0, w1, w2, X):
  plt.clf()

  # dividing S into S0 and S1
  S0 = [] # (e)
  S1 = [] # (f)

  for point in X:
    x1, x2 = point

    if w1*x1 + w2*x2 + w0 >= 0:
      S1.append(point)
    else:
      S0.append(point)
      
  # plotting the initial graph
  # plotting a scatter plot (of the dataset) and the perceptron line
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
  plt.suptitle('Graphical intepretation of a perceptron (N = ' + str(N) + ')')
  plt.legend()
  plt.show() # (g)

# picking initial weights w0', w1', w2'
w0_ = random.uniform(-1, 1) # (h)ii
w1_ = random.uniform(-1, 1)
w2_ = random.uniform(-1, 1)

initial_weights = [w0_, w1_, w2_].copy()

# function to count the number of misclassifications
def get_num_of_misclassifications(correct_weights, actual_weights, points):
  w0, w1, w2 = correct_weights
  w0_, w1_, w2_ = actual_weights

  misclassifications = 0
  correct_classifications = 0

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

  return misclassifications

def train():
  global w0_, w1_, w2_, w0, w1, w2
  epoch = 0
  misclassifications_per_epoch = []

  # the perceptron training algorithm
  # it will run till all the points are classified
  while True:
    misclassifications = 0
    for point in X:
      x1, x2 = point
      desired_output = step(w0 + w1*x1 + w2*x2)
      actual_output = step(w0_ + w1_*x1 + w2_*x2)
      
      # comparing the desired output to the actual output
      # if there is a mismatch, then update the weights
      if actual_output != desired_output:
        w0_ = w0_ + eta * 1 * (desired_output - actual_output)
        w1_ = w1_ + eta * x1 * (desired_output - actual_output)
        w2_ = w2_ + eta * x2 * (desired_output - actual_output)
        misclassifications = misclassifications + 1

    # if all points are correctly classified, then stop training
    if misclassifications == 0:
      misclassifications_per_epoch.append(0)
      break
    
    misclassifications_per_epoch.append(misclassifications)
    epoch = epoch + 1

  return misclassifications_per_epoch

# function to plot a graph of misclassifications vs epoch number
def show_misclassifications_graph(misclassifications_per_epoch):
  plt.clf()
  number_of_epochs = len(misclassifications_per_epoch)
  x = list(range(number_of_epochs))
  y = misclassifications_per_epoch
  plt.plot(x, y, label="Misclassifications per epoch")
  plt.suptitle('Misclassifications per epoch (\u03B7 = ' + str(eta) + ' and N = ' + str(N) + ')')
  plt.xlabel('Epoch')
  plt.ylabel('Misclassifications')
  plt.legend()
  plt.show()

# printing the optimal and final weights
def print_weights_comparison():
  print('Optimal weights: ' + str([w0, w1, w2]))
  print('Final weights: ' + str([w0_, w1_, w2_]))
  print('Difference between optimal and final w0: ' + str(abs(abs(w0) - abs(w0_))))
  print('Difference between optimal and final w1: ' + str(abs(abs(w1) - abs(w1_))))
  print('Difference between optimal and final w2: ' + str(abs(abs(w2) - abs(w2_))))


rows = []
N = 0
eta = 0

def run():
  row = []
  global N, w0, w1, w2, w0_, w1_, w2_, X, eta

  # set N = 100 and run the algorithm for eta = 1, 10, 0.1
  N = 100
  w0, w1, w2, X = initialize(N)
  plot_initial_graph(w0, w1, w2, X)

  eta = 1 # (h)i
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))

  print_weights_comparison()

  w0_, w1_, w2_ = initial_weights
  eta = 10
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))


  w0_, w1_, w2_ = initial_weights
  eta = 0.1
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))

  # set N = 1000 and run the algorithm for eta = 1, 10, 0.1
  N = 1000
  w0, w1, w2, X = initialize(N)
  plot_initial_graph(w0, w1, w2, X)


  eta = 1 # i
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))


  w0_, w1_, w2_ = initial_weights
  eta = 10
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))



  w0_, w1_, w2_ = initial_weights
  eta = 0.1
  misclassifications_per_epoch = train()
  show_misclassifications_graph(misclassifications_per_epoch)
  row.append(len(misclassifications_per_epoch))

  rows.append(row)


for i in range(1):
  run()
  if i % 10 == 0:
    print(i)

print(rows)

'''
(h) vii
initial weights: [-0.03384998158526087, -0.29792879896350755, -0.8273876976548766]
final weights: [-1.033849981585261, -3.1701347875342694, 1.2139549075551104]
Difference between initial and final w0: 1.0000000000000002
Difference between initial and final w1: 2.872205988570762
Difference between initial and final w2: 2.0413426052099872

The initial weights w0', w1' and w2' can be different from the optimal weights. 
However, the lines w0 + w1x + w2y = 0 and w0' + w1'x + w2'y = 0 represent almost the same
line when viewed on a graph


After running the perceptron training algorithm 2000 times, the following results were observed:

Parameters: N = 100 and eta = 1
Average epochs needed until convergence: 80
Median epochs needed until convergence: 12

Parameters: N = 100 and eta = 10
Average epochs needed until convergence: 60
Median epochs needed until convergence: 13

Parameters: N = 100 and eta = 0.1
Average epochs needed until convergence: 55
Median epochs needed until convergence: 11

Parameters: N = 1000 and eta = 1
Average epochs needed until convergence: 251
Median epochs needed until convergence: 34

Parameters: N = 1000 and eta = 10
Average epochs needed until convergence: 282
Median epochs needed until convergence: 34

Parameters: N = 1000 and eta = 0.1
Average epochs needed until convergence: 241
Median epochs needed until convergence: 33

(l)
Changing the value of eta does not have too much effect on the number of epochs needed until convergence.

(m)
Sometimes our initial choice of weights is indeed unlucky and thus the algorithm takes a rather long time to converge.
However, this happens only rarely. If we start with a different set of initial weights, then training time does vary slightly.
However, as long as we do not change eta or N, it is not a large difference.

(n)
As N increases, the number of epochs needed until convergence increases propotionally.
'''

