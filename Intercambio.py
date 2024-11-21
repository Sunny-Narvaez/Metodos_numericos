import numpy as np

def intercambio_inversa(A):

    n = A.shape[0]
    # Comprobación de que la matriz sea cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    # Augmentamos la matriz A con la identidad
    A = np.hstack((A, np.eye(n)))

    # Método de Intercambio
    for k in range(n):
        # Seleccionar pivote
        pivote = A[k, k]
        if np.abs(pivote) < 1e-12:
            raise ValueError("El pivote es cero, la matriz no es invertible.")
        
        # Paso 1: Dividir fila del pivote por el valor del pivote
        A[k, :] = A[k, :] / pivote
        
        # Paso 2: Actualizar filas restantes
        for i in range(n):
            if i != k:
                factor = A[i, k]
                A[i, :] -= factor * A[k, :]
    
    # Extraer la matriz inversa (segunda mitad)
    inversa = A[:, n:]
    return inversa

def exchange_method(matrix, vector):
    """Resuelve un sistema de ecuaciones lineales utilizando el método de intercambio."""
    n = len(matrix)
    augmented_matrix = np.hstack((matrix, vector.reshape(-1, 1)))

    for k in range(n):
        # Selección del pivote mayor en valor absoluto
        max_row, max_col = divmod(np.argmax(np.abs(augmented_matrix[k:, k:n])), n - k)
        max_row += k
        max_col += k

        # Intercambio de filas y columnas
        augmented_matrix[[k, max_row]] = augmented_matrix[[max_row, k]]
        augmented_matrix[:, [k, max_col]] = augmented_matrix[:, [max_col, k]]

        # Operaciones de intercambio
        pivot = augmented_matrix[k, k]
        if pivot == 0:
            raise ValueError("El sistema no tiene solución única.")

        # Paso 1: Dividir el renglón del pivote
        augmented_matrix[k, k:] /= pivot
        augmented_matrix[k, k] = 1 / pivot  # Reciproco del pivote

        # Paso 2: Ajustar los demás renglones
        for i in range(n):
            if i != k:
                factor = augmented_matrix[i, k]
                augmented_matrix[i, k:] -= factor * augmented_matrix[k, k:]
                augmented_matrix[i, k] = factor / pivot

    # Extraer la solución del sistema
    solution = augmented_matrix[:, -1]
    return solution

def main():
    # Solicitar el tamaño de la matriz
    n = int(input("Ingrese el tamaño de la matriz (n x n): "))
    
    # Crear una matriz vacía de tamaño n x n
    matrix = np.zeros((n, n))
    
    # Solicitar los elementos de la matriz
    print("Ingrese los elementos de la matriz A:")
    for i in range(n):
        matrix[i] = list(map(float, input().split()))
    
    # Solicitar el vector de términos independientes
    print("Ingrese el vector de términos independientes:")
    vector = np.array(list(map(float, input().split())))

    try:
        solution = exchange_method(matrix, vector)
        print("Solución del sistema:")
        for i, x in enumerate(solution, start=1):
            print(f"x{i} = {x:.4f}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
