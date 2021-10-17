import numpy as np
import matplotlib.pyplot as plt

N = 24
POINTS = 300
eta = 0.0001

def average(V):
  return sum(V) / len(V)

def phi(x):
  return np.tanh(x)

def phi_dash(x):
  return 1 / (np.cosh(x)**2)

def get_random():
  return np.random.uniform(0, 1)

def plot_mse_graph(mse_per_epoch):
  epochs = len(mse_per_epoch)
  plt.clf()
  plt.plot(range(epochs), mse_per_epoch,label="MSE vs epoch")

  plt.suptitle("MSE vs epoch")
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.show()

def plot_graph(X, D, network=None):
  plt.clf()
  plt.scatter(X, D, label="The curve")

  if network is not None:
    Y = list(map(lambda x: network.get_network_output(x), X))
    plt.scatter(X, Y, label="The fitted curve")

  plt.suptitle("The curve")
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.show()


class HW5:

  def get_network_output(self, x):
    I = np.tanh(self.W1.transpose() * x + self.B)
    Y = np.matmul(self.W2.transpose(), I) + self.b
    return Y

  def mse_avg(self):
    X, D, get_network_output = self.X, self.D, self.get_network_output
    return average([(D[i] - get_network_output(X[i]))**2 for i in range(POINTS)])

  def mse(self, d, x):
    return (d - self.get_network_output(x))**2

  def run_epoch(self, index):
    global eta
    x, d = self.X[index], self.D[index]
    W1, W2, B, b = self.W1, self.W2, self.B, self.b

    # print("x: {}".format(x))

    initial_mse = self.mse_avg()
    
    y = self.get_network_output(x)

    error_signal = (d - y)

    del_E_by_del_b = -(1)*(error_signal)

    del_E_by_del_B = -(1)*((error_signal)*W2*phi_dash(W1*x+B))

    del_E_by_del_W1 = -(x) * ((error_signal)*W2*phi_dash(W1*x+B))

    del_E_by_del_W2 = -(phi(W1*x+B))*(error_signal)

    self.b = b - eta * del_E_by_del_b
    self.B = B - eta * del_E_by_del_B

    self.W1 = W1 - eta * del_E_by_del_W1 
    self.W2 = W2 - eta * del_E_by_del_W2

    y = self.get_network_output(x)
    self.mse_per_epoch.append(y)

    current_mse = self.mse_avg()

    if current_mse > initial_mse:
      eta = 0.9 * eta

  def __init__(self):
    self.X = np.array([get_random() for i in range(POINTS)])
    self.V = np.array([np.random.uniform(-0.1, 0.1) for i in range(POINTS)])

    D = []
    for index in range(POINTS):
      x, v = self.X[index], self.V[index]
      D.append(np.sin(20 * x) + 3*x + v)
    self.D = np.array(D)

    self.W1 = np.array([get_random() for i in range(N)])
    self.W2 = np.array([get_random() for i in range(N)])
    self.B = np.array([get_random() for i in range(N)])
    self.b = get_random()

    self.mse_per_epoch = []

  def run(self):
    # plot_graph(self.X, self.D, self)
    # print(self.get_network_output(2))
    print("MSE: {}".format(self.mse_avg()))

    count = 0
    while count < POINTS * 5:
      self.run_epoch(count % POINTS)
      count = count + 1

    print("MSE: {}".format(self.mse_avg()))
    plot_graph(self.X, self.D, self)
    plot_mse_graph(self.mse_per_epoch)

hw5 = HW5()
hw5.run()





