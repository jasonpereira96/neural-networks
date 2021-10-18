import numpy as np
import matplotlib.pyplot as plt

N = 24
POINTS = 300
eta = 0.01

def average(V):
  return sum(V) / len(V)

def phi(x):
  return np.tanh(x)

def phi_dash(x):
  return 1 - np.power(np.tanh(x), 2)
  # return 1 / (np.cosh(x)**2)

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
    Z = list(zip(X, Y))
    Z.sort()
    X = list(map(lambda x:x[0], Z))
    Y = list(map(lambda x:x[1], Z))
    plt.plot(X, Y, label="The fitted curve", color="red")

  plt.suptitle("The curve")
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.show()


class HW5:

  def get_network_output(self, x):
    I = np.tanh(self.W1.transpose() * x + self.B)
    Y = np.dot(self.W2.transpose(), I) + self.b
    return Y

  def FeedForward(self,x): # This method feeds forward the input x and returs the predicted output
    # I use the same notation as Haykin book
    self.v_1 = x*(self.W1) + self.B #Local Induced Fileds of the hidden layer
    self.y_1 = np.tanh(self.v_1)
    self.v_2 = self.y_1.T.dot(self.W2) + self.b
    self.o = self.v_2 # output of the network
    return self.o

  def mse_avg(self):
    X, D, get_network_output = self.X, self.D, self.get_network_output
    return average([(D[i] - get_network_output(X[i]))**2 for i in range(POINTS)])

  def mse(self, d, x):
    return (d - self.get_network_output(x))**2

  def loss(self,X,D): # Calculates the cost function of the network for a 'vector' of the inputs/outputs
    #x : input vector
    #d : desired output
    temp = np.zeros(len(X))
    for i in range(len(X)):
        temp[i] = self.D[i] - self.FeedForward(X[i])
    self.cost = np.mean(np.square(temp))
    return self.cost

  def run_epoch(self, index):
    global eta
    x, d = self.X[index], self.D[index]

    # print("x: {}".format(x))

    # initial_mse = self.mse()

    v_1 = x*(self.W1) + self.B #Local Induced Fileds of the hidden layer
    y_1 = np.tanh(v_1)
    v_2 = y_1.T.dot(self.W2) + self.b
    y = v_2 # output of the network
    print("in run_epoch() x:{} d:{} y:{}".format(x, d, y))

    
    # y = self.get_network_output(x)

    error_signal = (d - y)

    del_E_by_del_b = -(1)*(error_signal)
    
    del_E_by_del_W2 = -(phi(v_1))*(error_signal)
    print("in RE")
    print("b update: {}".format(del_E_by_del_b))
    print("W2 update: {}".format(del_E_by_del_W2))
    print("")


    return

    self.W2 = self.W2 - eta * del_E_by_del_W2
    self.b = self.b - eta * del_E_by_del_b

    
    del_E_by_del_B = -(1)*((error_signal)*self.W2*phi_dash(self.W1*x+self.B))
    del_E_by_del_W1 = -(x) * ((error_signal)*self.W2*phi_dash(self.W1*x+self.B))
    
    self.W1 = self.W1 - eta * del_E_by_del_W1 
    self.B = self.B - eta * del_E_by_del_B

    y = self.get_network_output(x)
    self.mse_per_epoch.append(y)

    current_mse = self.mse_avg()

  def BackPropagate(self,x,y,d): 
    # Given the input, desired output, and predicted output 
    # this method update the weights accordingly
    # I used the same notation as in Haykin: (4.13)
    self.delta_out = (d-y)*1 # 1: phi' of the output at the local induced field
    # self.W2 += eta*self.delta_out*self.y_1
    # self.b += eta*self.delta_out

    print("in BP")

    print("in BP() x:{} d:{} y:{}".format(x, d, y))
    print("b update: {}".format(self.delta_out))
    print("W2 update: {}".format(self.delta_out*self.y_1))

    return
    
    self.delta_1 = (1 - np.power(np.tanh(self.v_1), 2))*(self.w_2)*self.delta_out
    self.w_1 += eta*x*self.delta_1
    self.b_1 += eta*self.delta_1

    # if current_mse > initial_mse:
      # eta = 0.9 * eta
      # pass

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

    epoch = 0
    while epoch < 3000:
      index = epoch % POINTS
      self.run_epoch(index)
      epoch += 1

    print("MSE: {}".format(self.mse_avg()))
    plot_graph(self.X, self.D, self)
    plot_mse_graph(self.mse_per_epoch)

  def run2(self):
    print("get_network_output(): {}".format(self.get_network_output(self.X[3])))
    print("FF(): {}".format(self.FeedForward(self.X[3])))

    print("mse(): {}".format(self.mse_avg()))
    print("loss(): {}".format(self.loss(self.X, self.D)))

    self.run_epoch(3)
    print("get_network_output(): {}".format(self.get_network_output(self.X[3])))
    self.FeedForward(self.X[3])
    print("get_network_output(): {}".format(self.get_network_output(self.X[3])))
    self.BackPropagate(self.X[3], self.get_network_output(self.X[3]), self.D[3])
    print("get_network_output(): {}".format(self.get_network_output(self.X[3])))

hw5 = HW5()
hw5.run2()





