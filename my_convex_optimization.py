import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

f = lambda x : (x - 1) ** 4 + x ** 2
f_prime = lambda x : 4*((x-1)**3) + 2*x

def print_a_function(f, values):
  x = np.linspace(min(values), max(values), 1000)
  y = [f(v) for v in x]

  plt.plot(x, y)
  plt.xlabel("X")
  plt.ylabel("f(x)")
  plt.title("Function Plot")
  plt.show()


def find_root_bisection(f, min, max, tol=0.001, max_iters=100):

  a, b = min, max
  for _ in range(max_iters):
    midpoint = (a + b) / 2
    fa, fm, fb = f(a), f(midpoint), f(b)

    if abs(fm) <= tol:
      return midpoint

    if fa * fb < 0:
      b = midpoint
    else:
      a = midpoint
  return midpoint

def find_root_newton_raphson(f, f_deriv, start, tol=0.001, max_iters=100):
  x = start
  for _ in range(max_iters):
    fx = f(x)
    if abs(fx) <= tol:
      return x

    dx = -fx / f_deriv(x)
    x += dx

  return None

def gradient_descent(f, f_prime, start, learning_rate=0.01):

  x = start
  while True:
    dx = -learning_rate * f_prime(x)
    new_x = x + dx
    if abs(f(new_x) - f(x)) < 1e-5:
      break
    x = new_x

  return x

def solve_linear_problem(A, b, c):
    r = linprog(c, A, b)
    return round(r.fun), r.x 


def find_root(f, a, b):
  precision = 0.001
  max_iters = 100 

  for _ in range(max_iters):
    midpoint = (a + b) / 2
    fa, fm, fb = f(a), f(midpoint), f(b)

    if abs(fm) <= precision:
      return midpoint

    if fa * fb < 0:
      b = midpoint
    else:
      a = midpoint

  return None