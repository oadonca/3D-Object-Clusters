import numpy as np
x = np.array([True, False, True])
y = np.array([1, 1, 1])

print(x * y)


def hello(x):
  h_list = [1, 2, 3, x]
  def bye(y):
    h_list.append(y)
  bye(5)
  return h_list

print(hello(4))

x = np.array([[[1, 2, 3], [0,0,0], [4, 5, 6]], [[0, 0, 1], [3,2,1], [0, 0, 0]], [[1, 4, 1], [0,0,0], [1, 1, 6]]])
inds = np.where(((x[:,:,0] > 0) | (x[:,:,1] > 0) | (x[:,:,2] > 0)))
print('X\n', x)
print('x inds\n', x[inds])
print('x inds reshape\n', x[inds].reshape((-1, 3)))