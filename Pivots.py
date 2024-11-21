import numpy as np

def display_pivot_menu():
    print("Pivoteo")
    print("1. Pivoteo Máximo")
    print("2. Pivoteo Total")
    print("3. Pivoteo Parcial")
    print("4. Volver al menú principal")

def pedir_matriz_aumentada():
    n = int(input("Ingrese el tamaño de la matriz: "))
    AB = np.zeros((n, n + 1))
    print("Ingrese los elementos de la matriz aumentada:")
    for i in range(n):
        for j in range(n + 1):
            AB[i, j] = float(input(f"AB[{i+1}][{j+1}] = "))
    return AB

def pivoteo_maximo(AB):
    n = len(AB)
    for i in range(n):
        max_index = np.argmax(np.abs(AB[i:, i])) + i
        if i != max_index:
            AB[[i, max_index]] = AB[[max_index, i]]
        for j in range(i + 1, n):
            factor = AB[j][i] / AB[i][i]
            AB[j] = AB[j] - factor * AB[i]
    print("Matriz después del pivoteo máximo:")
    for fila in AB:
        print(fila)
    return AB

def pivoteo_total(AB):
    n = len(AB)
    for i in range(n):
        max_index = np.unravel_index(np.argmax(np.abs(AB[i:, i:n]), axis=None), AB[i:, i:n].shape)
        max_index = (max_index[0] + i, max_index[1] + i)
        if i != max_index[0]:
            AB[[i, max_index[0]]] = AB[[max_index[0], i]]
        if i != max_index[1]:
            AB[:, [i, max_index[1]]] = AB[:, [max_index[1], i]]
        for j in range(i + 1, n):
            factor = AB[j][i] / AB[i][i]
            AB[j] = AB[j] - factor * AB[i]
    print("Matriz después del pivoteo total:")
    for fila in AB:
        print(fila)
    return AB

def pivoteo_parcial(AB):
    n = len(AB)
    for i in range(n):
        max_index = i + np.argmax(np.abs(AB[i:, i]))
        if i != max_index:
            AB[[i, max_index]] = AB[[max_index, i]]
        for j in range(i + 1, n):
            factor = AB[j][i] / AB[i][i]
            AB[j] = AB[j] - factor * AB[i]
    print("Matriz después del pivoteo parcial:")
    for fila in AB:
        print(fila)
    return AB
