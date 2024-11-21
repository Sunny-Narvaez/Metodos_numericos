import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sin, cos, tan, cot, sec, csc
from Pivots import display_pivot_menu, pedir_matriz_aumentada, pivoteo_maximo, pivoteo_total, pivoteo_parcial
from Secant_method import secant_method, print_results, plot_function
from Jacobi import Jacobi
from Gauss_Jordan import pedir_matriz, resolver_gauss, pedir_particion
from Intercambio import intercambio_inversa, exchange_method  # Import the functions

def display_menu():
    print("Métodos numéricos")
    print("1. Método de la secante")
    print("2. Gauss Jordan / Partición de Matriz")
    print("3. Gauss con pivoteo")
    print("4. Gauss Método de Intercambio")
    print("5. Método de Jacobi")
    print("6. Salir")

# Función principal
def main():
    while True:
        display_menu()
        choice = input("Seleccione una opción: ")
        if choice == '1':
            func_str = input("Ingrese la función f(x) (por ejemplo, 'x**3 - 6*x**2 + 11*x - 6.1' o 'sin(x) - 0.5'): ")
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
            sub_choice = input("Seleccione una opción: \n1. Gauss Jordan\n2. Partición de Matriz\n")
            if sub_choice == '1':
                n, c = np.shape(A)
                r = np.linalg.matrix_rank(A)
                ab = np.c_[A, b]
                ra = np.linalg.matrix_rank(ab)
                print('rango (A)={} rango (Ab) ={} n ={}'.format(r, ra, n))
                if r == ra == n:
                    print('solución única')
                    x = resolver_gauss(np.hstack((A, b.reshape(-1, 1))))
                    print(x)
                elif r == ra < n:
                    print('múltiples soluciones')
                elif r < ra:
                    print('sin solución')
            elif sub_choice == '2':
                partition, b_partition = pedir_particion(A, b)
                print("La partición de la matriz es:")
                print(partition)

                # Combine the partition with the corresponding part of b to form the augmented matrix
                AB = np.hstack((partition, b_partition.reshape(-1, 1)))
                resolver_gauss(AB)
            else:
                print("Opción no válida. Intente de nuevo.")
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
                    AB = pedir_matriz_aumentada()
                    AB = pivoteo_parcial(AB)
                    resolver_gauss(AB)
                elif pivot_choice == '4':
                    break
                else:
                    print("Opción no válida. Intente de nuevo.")
        elif choice == '4':
            n = int(input("Ingrese el tamaño de la matriz (n x n): "))
            A = np.zeros((n, n))
            
            # Solicitar los elementos de la matriz
            print("Ingrese los elementos de la matriz A:")
            for i in range(n):
                for j in range(n):
                    A[i, j] = float(input(f"A[{i+1}][{j+1}] = "))
            
            # Solicitar el vector de términos independientes
            print("Ingrese el vector de términos independientes:")
            vector = np.array([float(input(f"b[{i+1}] = ")) for i in range(n)])
            
            # Calcular la inversa usando el método de intercambio
            try:
                A_inv = intercambio_inversa(A)
                print("Matriz original:")
                print(A)
                print("Matriz inversa:")
                print(A_inv)
                
                # Resolver el sistema de ecuaciones usando el método de intercambio
                solution = exchange_method(A, vector)
                print("Solución del sistema:")
                for i, x in enumerate(solution, start=1):
                    print(f"x{i} = {x:.4f}")
            except ValueError as e:
                print("Error:", e)
        elif choice == '5':
            Jacobi()
        elif choice == '6':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()