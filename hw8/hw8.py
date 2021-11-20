import math

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
     
    
  def set_value(self, x, y, v):
    x = x - 1
    y = y - 1

    N = self.N
    phys_y = y
    phys_x = N - x -1

    print("x: {} y: {} px: {} py: {}".format(x, y, phys_x, phys_y))
    self.board[phys_x][phys_y] = v

N = 4
board = Board(N)
counter = 1
for i in range(N):
  for j in range(N):
    board.set_value(i+1, j+1, i+1+j+1)
    counter += 1

print(board)


