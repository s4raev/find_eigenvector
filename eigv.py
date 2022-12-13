import numpy as np
from copy import copy
import sympy as sp


def gauss(ext_matrix: np.ndarray) -> np.ndarray:
    """
    Простой алгоритм Гаусса, принимающий расширенную матрицу системы и
    использующий максимальные элементы в качестве ведущих.
    Возвращает список корней системы уравнений.
    """
    ext_matrix = copy(ext_matrix)
    for i in range(len(ext_matrix)):

        max_i = np.argmax(ext_matrix[i:, i]) + i
        ext_matrix[[i, max_i]] = ext_matrix[[max_i, i]]

        ext_matrix[i] /= ext_matrix[i][i]
        for j in range(i + 1, len(ext_matrix)):
            ext_matrix[j, i:] -= ext_matrix[i, i:] * ext_matrix[j][i]

    x = np.zeros(len(ext_matrix))
    for i in range(len(x) - 1, -1, -1):
        x[i] = ext_matrix[i][-1] - x[i:] @ ext_matrix[i, i:-1]
    return x


def get_vector(A: np.ndarray, eig_value: float) -> list[np.ndarray]:
    """
    Для заданного собственного числа ищет базис собственного подпространства.

    Используется алгоритм Гаусса и аккуратно обрабатываются линейно зависимые
    уравнения в системе.
    """
    eig_vector = []
    n = A.shape[1]

    for i in range(n):
        A[i, i] -= eig_value
    A, k = sp.Matrix(A).rref(iszerofunc=lambda x: abs(x) < 1e-2,
                             normalize_last=False)
    A = np.array(A)
    A = np.delete(A, k, axis=1)

    for x in range(A.shape[1]):
        B = A.copy()
        X = np.zeros(B.shape[1])
        X[x] = 1
        ans = np.zeros(n)
        true_index = 0
        for i in range(B.shape[1]):
            B[:, i] *= -X[i]
        for i in range(n):
            if (i in k):
                ans[i] = np.sum(B[i])
                true_index += 1
                continue
            ans[i] = X[i - true_index]
        eig_vector.append(ans)
    return eig_vector


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Возвращает нормализованный вектор, коллинеарный поданному на вход.
    Для удобства, первый коэффициент всегда приводится к положительному.
    """
    if x[0] < 0:
        x *= -1
    norm = np.sqrt((x ** 2).sum())
    if norm == 0:
        return x
    return x / norm


def get_eig_vectors(eig_values: list[float], A: np.ndarray) -> (list[float], list[np.ndarray]):
    """
    Находит собственные вектора для заданных уникальных собственных чисел.
    Возвращает список собственных чисел, скорректированный с учётом их
    кратности, и "плоский" список собственных векторов (как numpy.linalg.eig).
    """
    eig_vectors = []
    eig_values_multiplicity = []
    for eig_value in eig_values:
        for x in get_vector(A.copy(), eig_value):
            eig_values_multiplicity.append(eig_value)
            eig_vectors.append(normalize(x))
    return eig_values_multiplicity, eig_vectors


a = sp.symbols('a')


def get_function(pol_coef: list[float]) -> sp.core.add.Add:
    """
    Вспомогательная функция для получения sympy функции
    из массива коэффициентов многочлена.
    """
    f = pol_coef[0]
    for i in range(1, len(pol_coef)):
        f += pol_coef[i] * a ** i
    return f


def get_solve(pol_coef: list[float], eps=0.1) -> list[float]:
    """
    Решает нелинейное уравнение с заданной точностью.

    Используется модифицированный метод Ньютона, который может работать
    в случае наличия кратных корней.
    """
    f = get_function(pol_coef)
    n_pol = len(pol_coef) - 1

    n_iter = 50
    x = n_iter * [0]
    x_solve = []

    x[0] = 4
    sum_p_root = 0
    for s in range(n_pol):
        for n in range(n_iter):
            x[n + 1] = (x[n] - f.subs(a, x[n]) / sp.diff(f, a).subs(a, x[n])
                        ).evalf()
            if n >= 1 and abs((x[n + 1] - x[n]) / (1 - (x[n + 1] - x[n])
                                                   / (x[n] - x[n - 1]))) < eps:
                k = n + 1
                break
        p = (1 / (1 - (x[k] - x[k - 1]) / (x[k - 1] - x[k - 2]))).evalf()
        x_solve.append(x[k])
        f = f / pow(a - x[k], p)
        p = round(p, 3)
        sum_p_root += p + 0.01
        if sum_p_root >= n_pol:
            break
    x_solve = [float(x) if type(x) == 'sympy.core.numbers.Float'
               else float(x.as_real_imag()[0]) for x in x_solve]
    return x_solve


def get_eig(A: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Вычисляет собственные числа и вектора переданной на вход матрицы.
    Формат вывода совместим с np.linalg.eig

    Используется метод непосредственной развёртки. Метод Крылова ищет
    коэффициенты характеристического многочлена, затем модифицированный метод
    Ньютона находит собственные числа и далее для каждого ищется базис
    собственного подпространства.
    """
    n = len(A)
    Y = np.zeros([n, n])
    y0 = np.zeros(n)
    y0[0] = 1
    Y[:, -1] = y0
    for i in range(n - 2, -1, -1):
        Y[:, i] = A @ Y[:, i + 1]
    ext_Y = np.hstack((Y, A @ -Y[:, [0]]))
    # Коэффициенты характеристического многочлена от младшего к старшему
    p = np.hstack(([1], gauss(ext_Y)))[::-1]

    lam = get_solve(p)  # собственные числа без учёта кратности
    # собственные числа с учётом кратности и собственные вектора
    lam_multiplicity, vectors = get_eig_vectors(lam, A)
    return np.array(lam_multiplicity), vectors


def check_eigvector(A: np.ndarray, x: np.ndarray) -> bool:
    """
    Проверяет, является ли вектор x собственным для матрицы A
    """
    return max(abs(normalize(A @ x) - normalize(x))) < 1e-2


def test_matrix(A: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Проверяет работу get_eig для заданной матрицы, печатая результаты
    в человекочитаемом виде в консоль.
    Возвращает вычисленные собственные числа и вектора.
    """
    print("Найти собственные числа и вектора для матрицы A:")
    print(A)
    val, vec = get_eig(A)
    print(f"Собственные числа:\n{'; '.join(map(str, val))}")
    newline = '\n'
    print(f"Собственные вектора:\n{newline.join(map(str, vec))}")
    for x in vec:
        if not check_eigvector(A, x):
            print(f"Проверка провалена для {x}")
            break
    else:
        print("Проверки прошли успешно")
    return val, vec


if __name__ == "__main__":
    A1 = np.array([
        [1, 2, 3, 4],
        [2, 1, 2, 3],
        [3, 2, 1, 2],
        [4, 3, 2, 1]
    ], dtype=float)
    A2 = np.array([
        [3, -2],
        [-4, 1]
    ], dtype=float)
    A3 = np.array([
        [5, 1, 2],
        [1, 4, 1],
        [2, 1, 3]
    ], dtype=float)
    test_matrix(A1)
    print("----")
    test_matrix(A2)
    print("----")
    test_matrix(A3)
