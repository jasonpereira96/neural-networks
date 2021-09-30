import numpy as np
import matplotlib.pyplot as plt

N = 50
EPSILON = 0.001
eta = 0.0001
min_value, min_point = np.Inf, [8, 8]

def sigma(values):
  return sum(values)

def get_initial_points():
  return np.random.uniform(-1, 1), np.random.uniform(-1, 1)

def has_converged(energies):
  if len(energies) < 2:
    return False
  else:
    return (energies[-2] - energies[-1]) <= EPSILON
  
class Question2:
  def f(self, w0, w1):
    # return ((y - w0 - w1*x) ** 2)
    cost = 0
    for index in range(N):
      x, y = self.X[index], self.Y[index]
      cost = cost + ((y - w0 - w1*x) ** 2)
    return cost / N

  def del_f_wrt_w0(self, w0, w1):
    # return (-2 * (y - w0 - w1*x))
    total = 0
    for index in range(N):
      x, y = self.X[index], self.Y[index]
      total = total + (-2 * (y - w0 - w1*x))
    return total / N

  def del_f_wrt_w1(self, w0, w1):
    # return ((-2 * (y - w0 - w1*x)) * x)
    total = 0
    for index in range(N):
      x, y = self.X[index], self.Y[index]
      total = total + ((-2 * (y - w0 - w1*x)) * x)
    return total / N

  def gradient_descent(self):
    global min_point, min_value
    w0, w1 = get_initial_points()
    energies = []
    epoch = 0

    while not has_converged(energies):
      F = self.f(w0, w1)
      energies.append(F)

      if F < min_value:
        min_point = [w0, w1]
        min_value = F

      w0_temp, w1_temp = w0, w1

      w0 = w0 - eta * self.del_f_wrt_w0(w0_temp, w1_temp)
      w1 = w1 - eta * self.del_f_wrt_w1(w0_temp, w1_temp)

      epoch = epoch + 1

    self.energies = energies
    return w0, w1
  
  def __init__(self):
    self.X = [i for i in range(1, N + 1)]
    self.Y = [i + np.random.uniform(-1, 1) for i in range(1, N + 1)]

  def get_coefficients(self):
    Y = self.Y
    X = self.X
    X_square = [x**2 for x in X]
    XY = [X[index] * Y[index] for index in range(N)]

    w0 = (sigma(Y)*sigma(X_square) - sigma(X)*sigma(XY)) / (N * sigma(X_square) - (sigma(X))**2)

    w1 = (N * sigma(XY) - sigma(X) * sigma(Y)) / (N * sigma(X_square) - sigma(X) ** 2)

    return w0, w1

  def plot_graph(self, lr_coefficients, gd_coefficients):
    # plot both graphs side by side and put in report
    # plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Optimizing cost function')

    ax1.set(xlabel='X', ylabel="Y")
    ax2.set(xlabel='X', ylabel="Y")

    w0, w1 = lr_coefficients
    ax1.scatter(self.X, self.Y, label="Data points")
    x = np.linspace(0, 50)
    y = w0 + w1 * x
    ax1.plot(x, y, c="red", label="Regression line using least squares")
    ax1.legend()

    w0_, w1_ = gd_coefficients
    ax2.scatter(self.X, self.Y, label="Data points")

    x = np.linspace(0, 50)
    y = w0_ + w1_ * x
    ax2.plot(x, y, c="yellow", label="Regression line using gradient descent")
    
    plt.legend()
    plt.show()  

  def plot_graph_merged(self, lr_coefficients, gd_coefficients):
    plt.clf()
    plt.suptitle('Least squares vs gradient descent (Zoomed in)')
    ax = plt.gca()

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.scatter(self.X, self.Y, label="Data points")

    w0, w1 = lr_coefficients
    x = np.linspace(0, 50)
    y = w0 + w1 * x
    plt.plot(x, y, c="red", label="Regression line using least squares")

    w0_, w1_ = gd_coefficients
    x = np.linspace(0, 50)
    y = w0_ + w1_ * x
    plt.plot(x, y, c="yellow", label="Regression line using gradient descent")
    
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 5)
    plt.legend()
    plt.show()  

  def run(self):
    w0, w1 = self.get_coefficients()
    print("Least squares weights w0: {} and w1: {}".format(w0,w1))
    # self.plot_graph([w0, w1])
    w0_, w1_ = self.gradient_descent()
    print("Gradient descent weights w0: {} and w1: {}".format(w0_ ,w1_))
    self.plot_graph([w0, w1], [w0_, w1_])
    self.plot_graph_merged([w0, w1], [w0_, w1_])
    print("Min value of cost function: {}".format(min_value))
    # print("Energies: {}".format(self.energies))

ls = Question2()
ls.run()
