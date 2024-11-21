import numpy as np

def jacobi_iteration(A, b, x):
    n = len(A)
    x_new = np.zeros_like(x)
    for i in range(n):
        s = sum(A[i][j] * x[j] for j in range(n) if j != i)
        x_new[i] = (b[i] - s) / A[i][i]
    return x_new

def jacobi(A, b, x0, tol=1e-10, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = jacobi_iteration(A, b, x)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def Jacobi():
    n = int(input("Ingrese el tamaño de la matriz: "))
    A = np.zeros((n, n))
    b = np.zeros(n)
    print("Ingrese los elementos de la matriz A:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1}][{j+1}] = "))
    print("Ingrese los elementos del vector b:")
    for i in range(n):
        b[i] = float(input(f"b[{i+1}] = "))
    x0 = np.zeros(n)
    tol = float(input("Ingrese la tolerancia: "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))
    x = jacobi(A, b, x0, tol, max_iter)
    print("Solución aproximada:")
    print(x)