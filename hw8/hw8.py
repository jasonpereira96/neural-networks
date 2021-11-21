import numpy as np

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
N = 5
Y = 0.9
LR = 0.1

actions = [UP, DOWN, LEFT, RIGHT]

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


def init_q_table():
  board = Board(N)
  n_entries = N * N * len(actions)
  entries = np.random.normal(0, 1, n_entries)
  counter = 0
  for i in range(N):
    for j in range(N):
      values = {}
      for action in actions:
        values[action] = entries[counter]
        counter += 1
      board.set_value(i+1, j+1, values)

  return board
      

board = init_q_table()
print(board)



# fuck
# instead of N*N states, may require N*N*2 or N*N*3 states