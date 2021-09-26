import numpy as np
import matplotlib.pyplot as plt

eta = 0.1

def log(x):
  return np.log(x)

def is_valid(x, y):
  return x + y < 1 and x > 0 and y > 0

def f(x, y):
  if is_valid(x, y):
    return -log(1 - x - y) - log(x) -log(y)
    # return x ** 2  + y ** 2
  else:
    return None

def del_f_wrt_x(x, y):
  return (1 / (1 - x - y)) - (1 / x)
  # return 2*x

def del_f_wrt_y(x, y):
  return (1 / (1 - x - y)) - (1 / y)
  # return 2*y

def nabla(x, y):
  return np.array([[
    del_f_wrt_y(x, y),
    del_f_wrt_x(x, y)
  ]])

def get_initial_points():
  return np.random.uniform(0.01, 0.49), np.random.uniform(0.01, 0.49)

min_value = np.Inf
min_point = [1,2]

def run():
  global min_point
  x, y = get_initial_points()
  energies = []
  reset_points = []
  epoch = 0

  while epoch < 50:
    if not is_valid(x, y):
      x, y = get_initial_points()
      reset_points.append([epoch, f(x, y)])
    
    F = f(x, y)
    energies.append(F)

    if F < min_value:
      min_point = [x, y]

    x_temp, y_temp = x, y

    y = y - eta * del_f_wrt_y(x_temp, y_temp)
    x = x - eta * del_f_wrt_x(x_temp, y_temp)

    epoch = epoch + 1

# plt.clf()
# number_of_epochs = len(energies)
# x = list(range(number_of_epochs))
# y = energies
# plt.plot(x, y, label="f(x, y)")
# plt.scatter(list(map(lambda point: point[0], reset_points)), list(map(lambda point: point[1], reset_points)), label="reset points", c="red")
# plt.suptitle('f(x, y)')
# plt.xlabel('Epoch')
# plt.ylabel('f(x, y)')
# plt.legend()
# plt.show()


for i in range(1000):
  # print(i)
  run()

print("global minima: {}".format(f(0.333, 0.333)))
print(min_point)
print(f(min_point[0], min_point[1]))


  




