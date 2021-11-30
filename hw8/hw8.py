import numpy as np

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
DEFAULT_REWARD = 0
EPISODES = 50000
G = 3
N = 5
Y = 0.6
LR = 0.1
actions = [UP, DOWN, LEFT, RIGHT]
START_STATE = (1,1,0)
HOME = (1, N)
T = 100
ALPHA = 0.6

pure_exploitation = False

n_states = N * N * (G+1)
n_entries = n_states * len(actions)

np.random.seed(55)

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
  if 0 < ran < ALPHA or pure_exploitation:
    return most_profitable_action
  else:
    rem_actions = list(filter(lambda a: a != most_profitable_action, actions))
    return rem_actions[np.random.randint(low=0, high=10000) % 3]


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



def train():
  for episode in range(1, EPISODES):
    if episode % 1000 == 0:
      print("episode: {}".format(episode))

    s = START_STATE
    for t in range(1, T):
      x, y, g = s
      action_to_take = get_next_action(s)
      next_state, r = get_next_state(s, action_to_take)
      q_old = q_table[(x, y, g, action_to_take)]
      estimate_of_optimal_future_value = get_estimate_of_optimal_future_value(next_state)
      update_value = LR * (r + Y**1 * estimate_of_optimal_future_value - q_old)
      q_table[(x, y, g, action_to_take)] = q_old + update_value
      # print(update_value)
      s = next_state


def print_policy():
  state = START_STATE
  total_reward = 0
  print("Policy:")
  for t in range(40):
    x, y, g = state
    best_action = policy(state)
    next_state, r = get_next_state(state, best_action)
    total_reward += r * (Y ** t)
    print("{}) {}  x: {} y: {} g: {}".format(t+1, best_action, next_state[0], next_state[1], next_state[2]))
    # print("x: {} y: {} g: {}".format(x, y, g))
    state = next_state

  print("Total reward: {}".format(total_reward))

train()

print_q_table()
print("n = {}, G = {}, alpha = {}, lr = {}, Y = {}".format(N, G, ALPHA, LR, Y))
print_policy()