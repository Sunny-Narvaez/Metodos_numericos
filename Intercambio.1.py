import numpy as np

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

def interactive_exchange_solver():
    """Programa interactivo para resolver sistemas de ecuaciones con el método de intercambio."""
    print("=== Método de Intercambio ===")
    n = int(input("Ingrese el número de ecuaciones: "))
    print("Ingrese la matriz de coeficientes (fila por fila):")
    matrix = np.array([list(map(float, input().split())) for _ in range(n)])
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
    interactive_exchange_solver()