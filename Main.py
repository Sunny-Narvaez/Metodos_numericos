import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
    x = np.linspace(-10, 10, 400)
    y = f(x)
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter([x0, x1, root], [f(x0), f(x1), f(root)], color='red')
    plt.title('Secant Method')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def display_menu():
    print("Métodos numéricos")
    print("1. Método de la secante")
    print("2. Gauss Jordan")
    print("3. Gauss con pivoteo")
    print("4. Gauss Método de Intercambio")
    print("5. Método de Jacobi")
    print("6. Salir")

def display_pivot_menu():
    print("Gauss con Pivoteo")
    print("1. Pivoteo máximo")
    print("2. Pivoteo total")
    print("3. Volver al menú principal")

def pedir_matriz_aumentada():
    n = int(input("Ingrese el tamaño de la matriz cuadrada: "))
    A = []
    b = []
    print("Ingrese los elementos de la matriz A por renglón:")
    for i in range(n):
        fila = []
        for j in range(n):
            elemento = float(input(f"A[{i+1}][{j+1}]: "))
            fila.append(elemento)
        A.append(fila)
    print("Ingrese los elementos de la matriz de resultados b:")
    for i in range(n):
        elemento = float(input(f"b[{i+1}]: "))
        b.append(elemento)
    AB = []
    for i in range(n):
        AB.append(A[i] + [b[i]])
    print("Matriz aumentada:")
    for fila in AB:
        print(fila)
    return AB

def pivoteo_maximo(AB):
    n = len(AB)
    max_index = 0
    max_value = abs(AB[0][0])
    for i in range(1, n):
        if abs(AB[i][0]) > max_value:
            max_value = abs(AB[i][0])
            max_index = i
    if max_index != 0:
        AB[0], AB[max_index] = AB[max_index], AB[0]
    print("Matriz después del pivoteo máximo:")
    for fila in AB:
        print(fila)
    return AB

def pivoteo_total(AB):
    n = len(AB)
    max_row, max_col = 0, 0
    max_value = abs(AB[0][0])
    for i in range(n):
        for j in range(n):
            if abs(AB[i][j]) > max_value:
                max_value = abs(AB[i][j])
                max_row, max_col = i, j
    if max_col != 0:
        for i in range(n):
            AB[i][0], AB[i][max_col] = AB[i][max_col], AB[i][0]
    if max_row != 0:
        AB[0], AB[max_row] = AB[max_row], AB[0]
    print("Matriz después del pivoteo total:")
    for fila in AB:
        print(fila)
    return AB

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

def pedir_grado_polinomio():
    grado = int(input("Ingrese el grado del polinomio: "))
    return grado

def capturar_polinomio(grado):
    matriz = [[0] * grado for _ in range(grado)]
    B = []
    for i in range(grado):
        polinomio = input(f"Ingrese el polinomio {i + 1} (ejemplo: 13x4-18x3-5x2+4x1): ")
        terminos = polinomio.replace('-', '+-').split('+')
        terminos = [t.strip() for t in terminos if t.strip()]
        for termino in terminos:
            if 'x' in termino:
                coeficiente, incognita = termino.split('x')
                coeficiente = int(coeficiente)
                incognita = int(incognita)
                matriz[i][grado - incognita] = coeficiente
            else:
                print("Formato incorrecto. Por favor ingrese el término en el formato correcto (ejemplo: 13x4-18x3-5x2+4x1).")
                return capturar_polinomio(grado)
        resultado = int(input(f"Ingrese el resultado del polinomio {i + 1}: "))
        B.append(resultado)
    return matriz, B

def imprimir_matriz_aumentada(matriz, B):
    matriz_aumentada = [fila + [B[i]] for i, fila in enumerate(matriz)]
    print("Matriz aumentada:")
    for fila in matriz_aumentada:
        print(fila)

def despejar_fila(matriz, B, fila):
    m = len(matriz[0])
    for j in range(m):
        if matriz[fila][j] != 0:
            an1 = matriz[fila][j]
            break
    else:
        raise ValueError(f"No se encontró un elemento distinto de cero en la fila {fila}.")
    resultado = B[fila]
    ecuacion = f"({resultado} / {an1})"
    for k in range(m):
        if k != j:
            coef = matriz[fila][k]
            ecuacion += f" - ({coef} / {an1})x{m-k}"
    return ecuacion, j

def reemplazar_en_filas(matriz, B, ecuacion, columna, grado, fila_actual):
    for i in range(grado):
        if i == fila_actual:
            continue
        nuevo_polinomio = ""
        for j in range(grado):
            if j == columna:
                coeficiente = matriz[i][j]
                nuevo_polinomio += f"{coeficiente}*({ecuacion})"
            else:
                coeficiente = matriz[i][j]
                nuevo_polinomio += f"{coeficiente}x{grado-j}"
            nuevo_polinomio += " + "
        nuevo_polinomio = nuevo_polinomio[:-3]
        print(f"Nuevo polinomio {i + 1}: {nuevo_polinomio} = {B[i]}")

def resolver_sistema_polinomios(matriz, B, grado):
    for fila_actual in range(grado-1, -1, -1):
        ecuacion_despejada, columna = despejar_fila(matriz, B, fila_actual)
        print(f"Ecuación despejada: x{grado-columna} = {ecuacion_despejada}")
        reemplazar_en_filas(matriz, B, ecuacion_despejada, columna, grado, fila_actual)

def ingresar_matriz():
    n = int(input("Ingrese el tamaño de la matriz: "))
    A = np.zeros((n, n))
    b = np.zeros(n)
    print("Ingrese los elementos de la matriz A:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i}][{j}] = "))
    print("Ingrese los elementos del vector b:")
    for i in range(n):
        b[i] = float(input(f"b[{i}] = "))
    return A, b

def mostrar_sistema_ecuaciones(A, b):
    n = len(b)
    print("Sistema de ecuaciones:")
    for i in range(n):
        ecuacion = " + ".join([f"{A[i, j]}*x{j+1}" for j in range(n)])
        print(f"{ecuacion} = {b[i]}")
    print()

def mostrar_forma_matricial(A, b):
    print("Sistema de ecuaciones en su forma matricial:")
    print("A =")
    print(A)
    print("b =")
    print(b)
    print()

def gauss_eliminaAdelante(AB, vertabla=False, lu=False, casicero=1e-15):
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]
    L = np.identity(n, dtype=float)
    if vertabla:
        print('Elimina hacia adelante:')
    for i in range(0, n, 1):
        pivote = AB[i, i]
        adelante = i+1
        if vertabla:
            print(' fila', i, 'pivote: ', pivote)
        for k in range(adelante, n, 1):
            if np.abs(pivote) >= casicero:
                factor = AB[k, i] / pivote
                AB[k, :] = AB[k, :] - factor * AB[i, :]
                L[k, i] = factor
                if vertabla:
                    print('   factor: ', factor, ' para fila: ', k)
            else:
                print('  pivote:', pivote, 'en fila:', i,
                      'genera division para cero')
    respuesta = AB
    if vertabla:
        print("Matriz U:")
        print(AB)
        print("Matriz L:")
        print(L)
    return L, AB[:, :-1], AB[:, -1]

def resolver_sistema(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def comprobar_solucion(A, x, b):
    print("Comprobación de la solución:")
    b_calculado = np.dot(A, x)
    print("b calculado =")
    print(b_calculado)
    print("b original =")
    print(b)
    print("Diferencia =")
    print(b - b_calculado)
    print()

def Jacobi():
    A, b = ingresar_matriz()
    mostrar_sistema_ecuaciones(A, b)
    mostrar_forma_matricial(A, b)
    
    AB = np.hstack([A, b.reshape(-1, 1)])
    L, U, b = gauss_eliminaAdelante(AB, vertabla=True)
    
    print("Matrices factorizadas:")
    print("L =")
    print(L)
    print("U =")
    print(U)
    print()
    
    x = resolver_sistema(U, b)
    print("Solución del Sistema de ecuaciones:")
    print("x =")
    print(x)
    print()
    
    comprobar_solucion(A, x, b)

def gaussJordan(a, b):
    n, _ = np.shape(a)
    A = np.c_[a, b]
    for i in range(n):
        for j in range(n):
            if A[j, i] != 0 and A[i, i] != 0 and i != j:
                f = A[j, i] / A[i, i]
                A[j, i + 1:n + 1] = A[j, i + 1:n + 1] - f * A[i, i + 1:n + 1]
    x = np.zeros(n)
    for i in range(n):
        x[i] = A[i, n] / A[i, i]
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

# Función principal
def main():
    while True:
        display_menu()
        choice = input("Seleccione una opción: ")
        if choice == '1':
            func_str = input("Ingrese la función f(x) (por ejemplo, 'x**3 - 6*x**2 + 11*x - 6.1'): ")
            try:
                x = sp.symbols('x')
                f_sympy = sp.sympify(func_str)
                f = sp.lambdify(x, f_sympy, 'numpy')
                x0 = float(input("Ingrese el valor inicial x0: "))
                x1 = float(input("Ingrese el valor inicial x1: "))
                tol = 1e-6
                max_iter = int(input("Ingrese el número máximo de iteraciones: "))
                results = secant_method(f, x0, x1, tol, max_iter)
                print_results(results)
                if results:
                    root = results[-1][1]
                    plot_function(f, x0, x1, root, results)
                    print(f"Raíz aproximada: {root:.6f}")
                else:
                    print("El método no convergió")
            except sp.SympifyError:
                print("Error: La función ingresada no es válida. Intente de nuevo.")
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '2':
            A, b = pedir_matriz()
            n, c = np.shape(A)
            r = np.linalg.matrix_rank(A)
            ab = np.c_[A, b]
            ra = np.linalg.matrix_rank(ab)
            print('rango (A)={} rango (Ab) ={} n ={}'.format(r, ra, n))
            if r == ra == n:
                print('solución única')
                x = gaussJordan(A, b)
                print(x)
            elif r == ra < n:
                print('múltiples soluciones')
            elif r < ra:
                print('sin solución')
        elif choice == '3':
            while True:
                display_pivot_menu()
                pivot_choice = input("Seleccione una opción: ")
                if pivot_choice == '1':
                    AB = pedir_matriz_aumentada()
                    AB = pivoteo_maximo(AB)
                    resolver_gauss(AB)
                elif pivot_choice == '2':
                    AB = pedir_matriz_aumentada()
                    AB = pivoteo_total(AB)
                    resolver_gauss(AB)
                elif pivot_choice == '3':
                    break
                else:
                    print("Opción no válida. Intente de nuevo.")
        elif choice == '4':
            grado = pedir_grado_polinomio()
            matriz, B = capturar_polinomio(grado)
            print("Matriz de coeficientes:", matriz)
            print("Matriz de resultados B:", B)
            imprimir_matriz_aumentada(matriz, B)
            resolver_sistema_polinomios(matriz, B, grado)
        elif choice == '5':
            Jacobi()
        elif choice == '6':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()