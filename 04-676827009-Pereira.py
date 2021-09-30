import numpy as np
import matplotlib.pyplot as plt

eta = 0.001
EPSILON = 0.01

def log(x):
  return np.log(x)

def inverse(matrix):
  return np.linalg.inv(matrix)

def is_valid(x, y):
  return x + y < 1 and x > 0 and y > 0

def f(x, y):
  if is_valid(x, y):
    return -log(1 - x - y) - log(x) -log(y)
  else:
    return None

def del_f_wrt_x(x, y):
  return (1 / (1 - x - y)) - (1 / x)

def del_f_wrt_y(x, y):
  return (1 / (1 - x - y)) - (1 / y)

def get_initial_points():
  return np.random.uniform(0.01, 0.49), np.random.uniform(0.01, 0.49)

def has_converged(energies):
  if len(energies) < 2:
    return False
  else:
    return (energies[-2] - energies[-1]) <= EPSILON

min_value = np.Inf
min_point = [1,2]

def plot_graph(x, y, energies, title, trajectory, reset_points):
  is_gradient_descent = not ("Newton" in title)
  plt.clf()
  number_of_epochs = len(energies)
  x = list(range(number_of_epochs))
  y = energies
  plt.plot(x, y, label="f(x, y) vs Epoch")
  plt.suptitle(title)
  plt.xlabel('Epoch Number')
  plt.ylabel('f(x, y)')

  if reset_points is not None:
    plt.scatter(list(map(lambda x: x[0], reset_points)), list(map(lambda x: x[1], reset_points)), c="red", label="Reset points")
  
  plt.legend()
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x = list(map(lambda v: v[0], trajectory))
  y = list(map(lambda v: v[0], trajectory))
  z = np.linspace(0, number_of_epochs, number_of_epochs)
  fig.suptitle("Trajectory of f(x, y) {}".format("Gradient Descent" if is_gradient_descent else "Newton's Method"))
  ax.plot(x, y, z, label="Trajectory of f(x, y)")

  plt.legend()
  plt.show()


# Gradient descent
def train():
  global min_point, min_value
  x, y = get_initial_points()
  energies = []
  reset_points = []
  trajectory = []
  epoch = 0

  while not has_converged(energies):
    if not is_valid(x, y):
      x, y = get_initial_points()
      reset_points.append([epoch, f(x, y)])
    
    F = f(x, y)
    energies.append(F)
    trajectory.append((x, y))

    if F < min_value:
      min_point = [x, y]
      min_value = F

    x_temp, y_temp = x, y

    y = y - eta * del_f_wrt_y(x_temp, y_temp)
    x = x - eta * del_f_wrt_x(x_temp, y_temp)

    epoch = epoch + 1
  
  return x, y, energies, reset_points, trajectory


for i in range(1):
  x, y, energies, reset_points, trajectory = train()
  # print(trajectory)
  plot_graph(x, y, energies, "f(x ,y) vs Epoch number (Gradient descent) and \u03B7 = {}".format(eta), trajectory=trajectory, reset_points=reset_points)

print("global minima: {}".format(f(1/3, 1/3)))
print("")
print("Gradient descent")
print("Minima found using gradient descent: {} at point {}".format(min_value, min_point))


# Newton's method
class NewtonsMethod:
  def __init__(self):
    self.eta = 0.001
    self.min_point = None
    self.min_value = np.Inf

  def del_f_wrt_x_2(self, x, y):
    return (1 / (1 - x - y)**2) + (1 / x**2)

  def del_f_wrt_y_2(self, x, y):
    return (1 / (1 - x - y)**2) + (1 / y**2)

  def del_f_wrt_xy(self, x, y):
    return 1 / (1 - x - y)**2

  def hessian(self, x, y):
    return np.array([
      [self.del_f_wrt_x_2(x, y), self.del_f_wrt_xy(x, y)],
      [self.del_f_wrt_xy(x, y), self.del_f_wrt_y_2(x, y)],
    ])

  def train(self):
    x, y = get_initial_points()
    energies = []
    reset_points = []
    trajectory = []
    epoch = 0

    while not has_converged(energies):
      if not is_valid(x, y):
        x, y = get_initial_points()
        reset_points.append([epoch, f(x, y)])
      
      F = f(x, y)
      energies.append(F)
      trajectory.append((x, y))

      if F < self.min_value:
        self.min_point = [x, y]
        self.min_value = F

      old_point = np.array([x, y]).reshape((2, 1)) 
      gradient = np.array([del_f_wrt_x(x, y), del_f_wrt_y(x, y)]).reshape((2, 1))
      hessian = self.hessian(x, y)

      new_point = old_point - np.matmul(inverse(hessian), gradient)

      x = new_point[0][0]
      y = new_point[1][0]

      epoch = epoch + 1
    return x, y, energies, reset_points, trajectory

  def run(self):
    x, y, energies, reset_points, trajectory = self.train()
    print("")
    print("Newton's Method")
    print("Minima found using Newton's Method: {}".format(self.min_value))
    print("Min point: {}".format(self.min_point))
    # print("x: {} y: {}".format(x, y))
    # print("Energies: {}".format(energies))
    plot_graph(x, y, energies, "f(x, y) vs Epoch number (Newton's Method) and \u03B7 = {}".format(self.eta), reset_points=reset_points, trajectory=trajectory)

newtons_method = NewtonsMethod()
newtons_method.run()