import numpy as np
import matplotlib.pyplot as plt
import time as t
import numba as nb
import copy
from progress.bar import IncrementalBar
from numba.typed import List

start_time = t.perf_counter()
matrix_size = 10#int(input('Введите размер матрицы: '))
coord_point_array = []
repeat_points = []

@nb.njit(cache=True)
def func_1(x, y):
    """ функция 1 """
    return -x*(x**2 - 5) - y
@nb.njit(cache=True)
def func_2(x, y):
    """ функция 2 """
    return x + 1.3

def calc_start_point(quantity, segment):
    """Numba-optimized function for calculating start points"""
    start_point = np.empty((quantity, 2))
    for row in range(quantity):
        start_point[row, 0] = np.random.uniform(segment[0][0], segment[0][1])
        start_point[row, 1] = np.random.uniform(segment[1][0], segment[1][1])
    return start_point

def draw_grafic(X, Y):
    """Numba-optimized function for plotting the system"""
    plt.plot(X, Y, linewidth=0.2)

def draw_repeat_points(cell_size):
    """Numba-optimized function for plotting repeat points"""
    for el in repeat_points: 
        x_1 = (el[1] - int(matrix_size / 2)) * cell_size
        x = [x_1, x_1 + cell_size, x_1 + cell_size, x_1, x_1]
        y_1 = -(el[0] - int(matrix_size / 2)) * cell_size
        y = [y_1, y_1, y_1 + cell_size, y_1 + cell_size, y_1]
        plt.plot(x, y, 'red', linewidth=0.02)

def calc_cell_size(X, Y,  matrix_size):
    """ Рассчет размера клетки: массив точек по Х, массив точек по У, размер матрицы """
    cell_size = [((np.max(X) - np.min(X)) / (matrix_size - 1)), ((np.max(Y) - np.min(Y)) / (matrix_size - 1))]
    return np.max(cell_size)

def error_len(x, y, grid):
    """ Ошибка размера матрицы """
    if(x > (len(grid) - 1) or y > (len(grid) - 1) or x < 0 or y < 0):
        return False  # если x или у 
    else:             # вышли за границы матрицы 
        return True   # вывести ошибку, иначе продолжить

def save_coord_point(coord_point):
    """ Сохранение точек матрицы в массив: массив координат """
    coord_point_array.append(coord_point)

def comparison_point(arr, grid, cell_size):
    """ Заполнение точками: массив с точками, матрица """
    for i in nb.prange(1, len(arr) - 1):
        if(arr[i-1][1] < arr[i][1]):
            k = 0
        elif(arr[i-1][1] > arr[i][1]):
            k = 2
        if(arr[i-1][0] < arr[i][0]):
            k = 3
        elif(arr[i-1][0] > arr[i][0]):
            k = 1
        if(arr[i+1][1] < arr[i][1]):
            if(grid[arr[i][1]][arr[i][0]][0][k] == 0 or grid[arr[i][1]][arr[i][0]][0][k] == 1):
                grid[arr[i][1]][arr[i][0]][0][k] = 1
                grid[arr[i][1]][arr[i][0]][1] += 1
            else:
                point_temp = copy.deepcopy(grid[arr[i][1]][arr[i][0]][0])
                point_temp[k] = 1
                if len(repeat_points) != 0: 
                    if point_temp in repeat_points:
                        continue
                    else:
                        repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
                else:
                    repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
        elif(arr[i+1][1] > arr[i][1]):
            if(grid[arr[i][1]][arr[i][0]][0][k] == 0 or grid[arr[i][1]][arr[i][0]][0][k] == 3):
                grid[arr[i][1]][arr[i][0]][0][k] = 3
                grid[arr[i][1]][arr[i][0]][1] += 1
            else:
                point_temp = copy.deepcopy(grid[arr[i][1]][arr[i][0]][0])
                point_temp[k] = 3
                if len(repeat_points) != 0: 
                    if point_temp in repeat_points:
                        continue
                    else:
                        repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
                else:
                    repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
        if(arr[i+1][0] < arr[i][0]):
            if(grid[arr[i][1]][arr[i][0]][0][k] == 0 or grid[arr[i][1]][arr[i][0]][0][k] == 4):
                grid[arr[i][1]][arr[i][0]][0][k] = 4
                grid[arr[i][1]][arr[i][0]][1] += 1
            else:
                point_temp = copy.deepcopy(grid[arr[i][1]][arr[i][0]][0])
                point_temp[k] = 4
                if len(repeat_points) != 0: 
                    if point_temp in repeat_points:
                        continue
                    else:
                        repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
                else:
                    repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
        elif(arr[i+1][0] > arr[i][0]):
            if(grid[arr[i][1]][arr[i][0]][0][k] == 0 or grid[arr[i][1]][arr[i][0]][0][k] == 2):
                grid[arr[i][1]][arr[i][0]][0][k] = 2
                grid[arr[i][1]][arr[i][0]][1] += 1
            else:
                point_temp = copy.deepcopy(grid[arr[i][1]][arr[i][0]][0])
                point_temp[k] = 2
                if len(repeat_points) != 0: 
                    if point_temp in repeat_points:
                        continue
                    else: 
                        repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])
                else:
                    repeat_points.append([arr[i][1], arr[i][0], grid[arr[i][1]][arr[i][0]][0], point_temp])

def create_grid(matrix_size, func_1, func_2, cell_size=0):
    """ Принимает: размер матрицы, функция 1, функция 2, размер клетки(необязательно) """
    """ Начальные условия """
    segment = [0, 10]
    h = 0.001
    n = int((segment[1] - segment[0]) / h)
    quantity_start_point = 50#int(input('Количество стартовых точек: '))
    start_point = calc_start_point(quantity_start_point, [[-2.6, 2.6], [-4.3, 4.3]])
    mu = 0.1
    progress_bar = IncrementalBar('Start Point', max = quantity_start_point)
    try:
        grid = [[[[0, 0, 0, 0], 0] for col in nb.prange(matrix_size)] for row in nb.prange(matrix_size)] # заполнение матрицы нулями
        for j in nb.prange(len(start_point)):
            point = [[], []] # массив содержащий решения X и Y
            for i in nb.prange(n):
                if(i == 0):
                    point[0].append(start_point[j][0] + func_1(start_point[j][0], start_point[j][1]) * h / mu)
                    point[1].append(start_point[j][1] + func_1(start_point[j][0], start_point[j][1]) * h / mu)
                point[0].append(point[0][i] + func_1(point[0][i], point[1][i]) * h / mu)
                point[1].append(point[1][i] + func_2(point[0][i], point[1][i]) * h)
            draw_grafic(point[0], point[1])
            if(cell_size == 0):
                cell_size = calc_cell_size(point[0], point[1], matrix_size)
            x_temp = int( (matrix_size - 1) / 2 ) + int( point[0][0] / cell_size ) # рассчет первых точек Х и У
            y_temp = int( (matrix_size - 1) / 2 ) + int( point[0][0] / cell_size ) # относительно центра матрицы
            """ Заполнение матрицы по траектории движения """
            for i in nb.prange(2, n):
                if(error_len(x_temp, y_temp, grid)):
                    x_coor = int( (matrix_size + 1) / 2 ) + int( point[0][i] / cell_size ) # следующие координаты 
                    y_coor = int( (matrix_size + 1) / 2 ) - int( point[1][i] / cell_size ) # X и Y

                    if(error_len(x_coor, y_coor, grid)):
                        if([x_temp, y_temp] != [x_coor, y_coor]):
                            save_coord_point([x_temp, y_temp])
                    else: 
                        continue
                    x_temp = x_coor # переприсваем точки х и у
                    y_temp = y_coor # для дальнейших операций в цикле
                else: 
                    continue
            comparison_point(coord_point_array, grid, cell_size)
            progress_bar.next()
        draw_repeat_points(cell_size)            
        progress_bar.finish()
        print("Размер клетки:", cell_size)
        return grid
    except ValueError:
        print(str(ValueError))
matrix = create_grid(matrix_size, func_1, func_2)
plt.savefig(f'{matrix_size}.svg')
my_file = open(f'{matrix_size}.txt', "w+")
for string in matrix:
    my_file.write(str(string) + '\n')
my_file.close()
file_repeat = open(f'repeatOf{matrix_size}.txt', "w+")
for string in repeat_points:
    file_repeat.write(str(string) + "\n")
#print(repeat_points)
print('Время выполнения программы:', t.perf_counter() - start_time)