import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
DEFAULT_REWARD = 0
G = 3
N = 5
Y = 0.9
LR = 0.1
actions = [UP, DOWN, LEFT, RIGHT]
START_STATE = (1,1,0)
HOME = (1, N)
T = 100

n_states = N * N * (G+1)
n_entries = n_states * len(actions)


def init_q_table():
  board = {}
  entries = np.random.normal(0, 1, n_entries)
  counter = 0
  for x in range(1,N+1):
    for y in range(1, N+1):
      for g in range(G+1):
        for action in actions:
          state = (x, y, g, action)
          board[state] = entries[counter]
          counter += 1

  return board

q_table = init_q_table()

def is_valid(state):
  x, y, g = state
  return 0 <= g <= G and 1 <= x <= N and 1 <= y <= N

def plot_graph(X, Y, Z, C):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.scatter(X, Y, Z, c=C, lw=0, s=30, cmap='coolwarm', vmin=min(C), vmax=max(C))
  plt.show()


def plot_q_table():
  X, Y, Z, C = [],[],[],[]
  for k in q_table.keys():
    x, y, g, action = k
    q_value = q_table[k]
    if action == RIGHT:
      X.append(x)
      Y.append(y)
      C.append(q_value)
      Z.append(g)
  plot_graph(X, Y, Z, C)

class Board:
  def __init__(self, N):
    self.N = N
    self.board = [0] * N
    for i in range(N):
      self.board[i] = ['a'] * N

  def __str__(self):
    res = ''
    for i in range(self.N):
      res = res + str(self.board[i]) + '\n'
    return res
     
  def get_phys(self, x, y):
    x = x - 1
    y = y - 1

    N = self.N
    phys_y = y
    phys_x = N - x -1

    return phys_x, phys_y

  def set_value(self, x, y, v):
    phys_x, phys_y = self.get_phys(x, y)
    self.board[phys_x][phys_y] = v
  
  def get_value(self, x, y):
    phys_x, phys_y = self.get_phys(x, y)
    return self.board[phys_x][phys_y]

def ____reward(from_state, to_state):
  x1, y1, g1 = from_state
  x2, y2, g2 = to_state

  if x2 == 1 and y2 == N and g2 == 0:
    return g1
  
  return DEFAULT_REWARD
      

def get_coordinates(x, y, action):
  if action == UP:
    return x, y + 1
  if action == DOWN:
    return x, y - 1
  if action == LEFT:
    return x - 1, y
  if action == RIGHT:
    return x + 1, y
  

def get_next_action(state):
  x, y, g = state

  most_profitable_action = None
  max_q_value = -np.Inf

  for action in actions:
    x_new, y_new = get_coordinates(x, y, action)
    if is_valid((x_new, y_new, g)) and max_q_value < q_table[(x_new, y_new, g, action)]:
      most_profitable_action = action
      max_q_value =  q_table[(x_new, y_new, g, action)]

  ran = np.random.random()
  if 0 < ran < 0.6 and True:
    return most_profitable_action
  else:
    return actions[np.random.randint(low=0, high=10000) % 4]


def get_next_state(state, action):
  x, y, g = state
  # near home
  # H1
  if x == 1 and y == N-1 and action == UP and g > 0:
    return (x, y, 0), g

  # H2
  if x == 2 and y == N and action == LEFT and g > 0:
    return (x, y, 0), g

  # near mine
  # M1
  if x == N - 1 and y == 1 and action == RIGHT and g < G:
    return (x, y, g+1), 0

  # M1
  if x == N and y == 2 and action == DOWN and g < G:
    return (x, y, g+1), 0

  new_x, new_y = get_coordinates(x, y, action)

  if is_valid((new_x, new_y, g)):
    return (new_x, new_y, g), 0
  else:
    return (x, y, g), 0

def get_estimate_of_optimal_future_value(state):
  x, y, g = state
  max_optimal_future_value = -np.Inf
  

  for action in actions:
    optimal_future_value = q_table[(x, y, g, action)]
    max_optimal_future_value = max(max_optimal_future_value, optimal_future_value)

  return max_optimal_future_value

# A policy is a function that maps states to actions
def policy(state):
  x, y, g = state
  max_q_value = -np.Inf
  best_action = None
  for action in actions:
    if q_table[(x, y, g, action)] > max_q_value:
      best_action = action
      max_q_value = q_table[(x, y, g, action)]

  return best_action

def print_q_table():
  for k in q_table.keys():
    print("{}: {}".format(k, q_table[k]))


print_q_table()


for episode in range(1, 20000):
  if episode % 1000 == 0:
    print("episode: {}".format(episode))

  s = START_STATE
  for t in range(1, T):
    x, y, g = s
    action_to_take = get_next_action(s)
    next_state, r = get_next_state(s, action_to_take)
    q_old = q_table[(x, y, g, action_to_take)]
    estimate_of_optimal_future_value = get_estimate_of_optimal_future_value(next_state)
    update_value = LR * (r + Y**t * estimate_of_optimal_future_value - q_old)
    q_table[(x, y, g, action_to_take)] = q_old + update_value
    # print((x[0], x[1], x[2], action_to_take))
    # print(update_value)
    s = next_state

def print_policy():
  state = START_STATE
  print("Policy:")
  for i in range(50):
    x, y, g = state
    best_action = policy(state)
    next_state, r = get_next_state(state, best_action)
    print("x: {} y: {} g: {}".format(x, y, g))
    state = next_state


print_q_table()
print_policy()
# plot_q_table()