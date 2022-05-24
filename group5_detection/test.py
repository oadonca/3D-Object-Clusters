import numpy as np
import time
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

def test_func(x, y, z, test = None, fun=None):
  return z, y, z

def time_function(function, args, kwargs = ()):
  start = time.perf_counter()
  func_return = function(*args, **kwargs)
  end = time.perf_counter() - start
  return func_return, end

args = [1, 2, 3]
kwargs = {'test': 'Fun', 'fun': 'test'}
func_ret, exec_time = time_function(test_func, args, kwargs)

x, y, z = func_ret

print(x, y, z)