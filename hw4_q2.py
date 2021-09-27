import numpy as np
import matplotlib.pyplot as plt

N = 50
EPOCHS = 1000
eta = 0.0001
min_value, min_point = np.Inf, [8, 8]

def sigma(values):
  return sum(values)

def get_initial_points():
  return np.random.uniform(-1, 1), np.random.uniform(-1, 1)
  
# https://www.statisticshowto.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/
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

    while epoch < EPOCHS:
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
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Optimizing cost function')

    ax1.set(xlabel='X', ylabel="Y")
    ax2.set(xlabel='X', ylabel="Y")

    w0, w1 = lr_coefficients
    ax1.scatter(self.X, self.Y, label="Data points")
    x = np.linspace(0, 50)
    y = w0 + w1 * x
    ax1.plot(x, y, c="red", label="Regression line")

    w0, w1 = gd_coefficients
    ax2.scatter(self.X, self.Y, label="Data points")
    x = np.linspace(0, 50)
    y = w0 + w1 * x
    ax2.plot(x, y, c="yellow", label="Regression line")
    fig.show()
    

  def run(self):
    w0, w1 = self.get_coefficients()
    print("w0: {} and w1: {}".format(w0,w1))
    # self.plot_graph([w0, w1])
    w0_, w1_ = self.gradient_descent()
    print("w0: {} and w1: {}".format(w0,w1))
    self.plot_graph([w0, w1], [w0_, w1_])
    print("Min value: {}".format(min_value))
    print("Energies: {}".format(self.energies))

ls = Question2()
ls.run()
