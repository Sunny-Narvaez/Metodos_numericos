import numpy as np

def resolver_gauss(AB):
    n = len(AB)
    for i in range(n):
        for j in range(i+1, n):
            factor = AB[j][i] / AB[i][i]
            for k in range(i, n+1):
                AB[j][k] -= factor * AB[i][k]
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        suma = sum(AB[i][j] * x[j] for j in range(i+1, n))
        x[i] = (AB[i][n] - suma) / AB[i][i]
    print("Solución del sistema:")
    for i in range(n):
        print(f"x{i+1} = {x[i]}")
    return x

def pedir_matriz():
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
    return A, b

def pedir_particion(A, b):
    n = len(A)
    start_row = int(input(f"Ingrese la fila inicial de la partición (0 a {n-1}): "))
    start_col = int(input(f"Ingrese la columna inicial de la partición (0 a {n-1}): "))
    part_size = int(input("Ingrese el tamaño de la partición: "))
    if start_row + part_size > n or start_col + part_size > n:
        raise ValueError("La partición excede los límites de la matriz.")
    partition = A[start_row:start_row+part_size, start_col:start_col+part_size]
    b_partition = b[start_row:start_row+part_size]
    return partition, b_partition

def main():
    A, b = pedir_matriz()
    partition, b_partition = pedir_particion(A, b)
    print("La partición de la matriz es:")
    print(partition)
    # Combine the partition with vector b to form the augmented matrix
    AB = np.hstack((partition, b_partition.reshape(-1, 1)))
    resolver_gauss(AB)

if __name__ == "__main__":
    main()