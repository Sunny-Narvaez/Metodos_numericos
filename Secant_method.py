import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sin, cos, tan, cot, sec, csc 

def secant_method(f, x0, x1, tol, max_iter):
    results = []
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            print("Error: Division by zero")
            break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs((x2 - x1) / x2)
        results.append((i, x2, f(x2), error))
        if error < tol:
            break
        x0, x1 = x1, x2
    return results

def print_results(results):
    print("Iteración\t x\t\t f(x)\t\t Error relativo")
    for result in results:
        print(f"{result[0]}\t\t {result[1]:.6f}\t {result[2]:.6f}\t {result[3]:.6f}")

def plot_function(f, x0, x1, root, results):
    x_vals = np.linspace(x0, x1, 400)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.scatter([root], [f(root)], color='red')
    plt.title('Secant Method')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()