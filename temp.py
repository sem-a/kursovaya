import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time as t

start_time = t.time()

def draw_grid_for_grafic(cell_size, start_point_polygon):
    running = True
    i = 0
    while running:
        line_x = (start_point_polygon[0, 0] - cell_size) + i * cell_size
        line_y = (start_point_polygon[1, 1] + cell_size) - i * cell_size
        i += 1
        if line_x < start_point_polygon[0, 1]:
            plt.plot([line_x, line_x], [start_point_polygon[1, 0] - 1, start_point_polygon[1, 1] + 1], '-', linewidth=0.2, color='grey')
        if line_y > start_point_polygon[1, 0]:
            plt.plot([start_point_polygon[0, 0] - 1, start_point_polygon[0, 1] + 1], [line_y, line_y], '-', linewidth=0.2, color="grey")
        if line_x > start_point_polygon[0, 1] + 1 and line_y < start_point_polygon[1, 0] - 1:
            running = True
            break


def draw_grafic(point):
    X = point[0, 1:]
    Y = point[1, 1:]
    plt.plot(X, Y)

def draw_matrix(coord_point_arr):
    X = []
    Y = []
    for i in range(len(coord_point_arr)):
        X.append(coord_point_arr[i][0])
        Y.append(coord_point_arr[i][1])

    plt.plot(X, Y)

@nb.njit(cache=True)
def f_1(x, y):
  return -x*(x**2 - 5) - y
@nb.njit(cache=True)
def f_2(x, y):
  return x

@nb.njit(cache=True)
def method_euler(start_point, n, h, mu):
    """ Решение методом Эйлера """
    point = np.zeros( (2, n+1) )
    x, y = start_point[0], start_point[1]
    point[0, 0], point[1, 0] = x, y
    for i in range(n):
        dx = f_1(x, y) * h / mu
        dy = f_2(x, y) * h
        x += dx
        y += dy
        point[0, i+1], point[1, i+1] = x, y
    return point

@nb.njit(cache=True)
def calc_start_point_of_cell(quantity, polygon, grid_size, cell_size):
    """ Рассчет стартовых точек по центрам клеток """
    x_min, x_max = polygon[0][0], polygon[0][1]
    y_min, y_max = polygon[1][0], polygon[1][1]
    start_point = np.empty( (quantity, 2) )
    x_offset = 0
    y_offset = 0
    for i in nb.prange(quantity):
        x = x_min + (cell_size / 2) + x_offset * cell_size
        if (x_offset == grid_size - 1):
            x_offset = 0
            y_offset += 1
        y = y_min + (cell_size / 2) + y_offset * cell_size
        x_offset += 1
        start_point[i, 0] = x
        start_point[i, 1] = y
        if (start_point[i, 0] > x_max and start_point[i, 1] > y_max):
            return start_point
    return start_point

    

@nb.njit(cache=True)
def calc_start_point(quantity, polygon):
    """ Рассчет стартовых точек """
    x_min, x_max = polygon[0][0], polygon[0][1]
    y_min, y_max = polygon[1][0], polygon[1][1]
    start_point = np.empty( (quantity, 2) )
    for i in nb.prange(quantity):
        start_point[i, 0] = np.random.uniform(x_min, x_max)
        start_point[i, 1] = np.random.uniform(y_min, y_max)
    return start_point

@nb.njit(cache=True)
def calc_cell_size(point, size):
    """ Рассчет размера клетки: массив точек, размер матрицы """
    x_min, x_max = np.min(point[0]), np.max(point[0])
    y_min, y_max = np.min(point[1]), np.max(point[1])
    cell_size_x = (x_max - x_min) / (size - 1)
    cell_size_y = (y_max - y_min) / (size - 1)
    return max(cell_size_x, cell_size_y)

@nb.njit(cache=True)
def error_len(x, y, grid_size):
    """ Ошибка длины массива """
    return 0 <= x < grid_size and \
           0 <= y < grid_size

@nb.njit(cache=True)
def save_coord_point(coord_point_arr, grid_size, cell_size, point, n):
    """ Преобразование точек в координаты матрицы """
    prev_arr = np.array([[
        int( (grid_size - 1) / 2 ) + int( point[0, 0] / cell_size ), # рассчет точек 
        int( (grid_size - 1) / 2 ) + int( point[1, 0] / cell_size )  # относительно центра
    ]])                                                              # матрицы
    for i in nb.prange(1, n):
        curr_x = int( (grid_size + 1) / 2 ) + int( point[0, i] / cell_size )
        curr_y = int( (grid_size + 1) / 2 ) + int( point[1, i] / cell_size )

        if(error_len(prev_arr[0][0], prev_arr[0][1], grid_size) and error_len(curr_x, curr_y, grid_size)):
            if prev_arr[0][0] != curr_x or prev_arr[0][1] != curr_y:
                coord_point_arr = np.vstack((coord_point_arr, prev_arr.reshape(1, 2)))
            prev_arr[0][0], prev_arr[0][1] = curr_x, curr_y
    return coord_point_arr[1:]

@nb.njit(cache=True)
def is_in(e, arr):
    len_arr = len(arr)
    if(len_arr > 0):
        for i in nb.prange(len_arr):
            if e['y'] == arr[i]['y'] and e['x'] == arr[i]['x'] and \
                np.all(e['curr_arr'] == arr[i]['curr_arr']) and np.all(e['edited_arr'] == arr[i]['edited_arr']):
                return True
    return False

@nb.njit(cache=True)
def numba_vstack(arr1, arr2):
    combined_lenght = len(arr1) + len(arr2)
    result = np.zeros((combined_lenght), dtype=dtype_for_repeat_points)
    for i in nb.prange(len(arr1)):
        result[i] = arr1[i]
    for i in nb.prange(len(arr2)):
        result[len(arr1) + i] = arr2[i]
    return result

@nb.njit(cache=True)
def calc_middle_iteration(grid, grid_size):
    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row, col]['iteration'] != 0:
                res = int(grid[row, col]['iteration'] / grid[row, col]['pathway'])
                grid[row, col]['iteration'] = res
                
@nb.njit(cache=True)
def update_grid_and_repeat_points(grid, point_coords, k, new_value, repeat_points, repeat_points_temp, pathway_counter):
    """ Обновляет grid и repeat_points с новым значением """
    x, y = int(point_coords[0]), int(point_coords[1])
    if grid[y, x]['array'][k] in {0, new_value}:
        grid[y, x]['array'][k] = new_value
        grid[y, x]['iteration'] += 1.
        if grid[y, x]['pathway'] <= pathway_counter:
            grid[y, x]['pathway'] += 1
    else:
        point_temp = np.copy(grid[y, x]['array'])
        point_temp[k] = new_value
        is_unique = True
        for rp in repeat_points:
            if np.array_equal(point_temp, rp[3]):
                is_unique = False
                break
        if is_unique:
            repeat_points_temp[0]['y'], repeat_points_temp[0]['x'] = y, x
            repeat_points_temp[0]['curr_arr'] = np.copy(grid[y, x]['array'])
            repeat_points_temp[0]['edited_arr'] = point_temp
            if not is_in(repeat_points_temp[0], repeat_points):
                repeat_points = numba_vstack(repeat_points, repeat_points_temp)
    return repeat_points

@nb.njit(cache=True)
def comparison_point(arr, grid, repeat_points, repeat_points_temp, pathway_counter):
    """ Заполнение точками: массив с точками, матрица """
    for i in nb.prange(1, len(arr) - 1):
        k = 0
        if arr[i-1, 0] < arr[i, 0]:
            k = 3
        elif arr[i-1, 0] > arr[i, 0]:
            k = 1
        if arr[i-1, 1] < arr[i, 1]:
            k = 2
        elif arr[i-1, 1] > arr[i, 1]:
            k = 0
        if arr[i+1, 1] < arr[i, 1]:
            repeat_points = update_grid_and_repeat_points(grid, arr[i], k, 3,repeat_points,
                                                          repeat_points_temp, pathway_counter)
        elif arr[i+1, 1] > arr[i, 1]:
            repeat_points = update_grid_and_repeat_points(grid, arr[i], k, 1, repeat_points,
                                                          repeat_points_temp, pathway_counter)
        elif arr[i+1, 0] < arr[i, 0]:
            repeat_points = update_grid_and_repeat_points(grid, arr[i], k, 4, repeat_points,
                                                          repeat_points_temp, pathway_counter)
        elif arr[i+1, 0] > arr[i, 0]:
            repeat_points = update_grid_and_repeat_points(grid, arr[i], k, 2, repeat_points,
                                                          repeat_points_temp, pathway_counter)
    return repeat_points

#@nb.njit(cache=True)
def create_grid(grid, repeat_points, grid_size, repeat_points_temp, cell_size = 0):
    """ Принимает: размер матрицы, функция 1, функция 2, размер клетки """
    coord_point_arr = np.zeros( (0, 2) ) # массив для координат относительно матрицы
    for i in nb.prange(len(start_point_arr)):
        start_point = start_point_arr[i]
        pathway_counter = i
        point = method_euler(start_point, n, h, mu)
        draw_grafic(point)
        if cell_size == 0 and i == 0:
            cell_size = calc_cell_size(point, grid_size)
        coord_point_arr = save_coord_point(coord_point_arr, grid_size, 
                                           cell_size, point, n)
        repeat_points = comparison_point(coord_point_arr, grid, repeat_points, 
                                         repeat_points_temp, pathway_counter)
    calc_middle_iteration(grid, grid_size)
    return grid, point, coord_point_arr, repeat_points, cell_size

""" Глобальные переменные """

segment = np.array([0, 10])
h = 0.001 # step
mu = 0.1 # parameter
n = int( (segment[1] - segment[0]) / h )
quantity_start_point = 2000 # number of starting points
start_point_polygon = np.array( [ [-2.6, 2.6], [-4.7, 4.7] ] ) # a segment for
                                                               # selecting starting points
# start_point_arr = calc_start_point(quantity_start_point, start_point_polygon)

matrix_size = 100

start_point_arr = calc_start_point_of_cell(quantity_start_point, start_point_polygon, matrix_size, cell_size=0.09)

"""***********************"""

""" Создание матрицы в глобальной области """

dtype_for_matrix = np.dtype( [
    ('array', np.int16, (4)),
    ('iteration', np.int16),
    ('pathway', np.int16)
] )
grid = np.zeros((matrix_size, matrix_size), dtype=dtype_for_matrix)

dtype_for_repeat_points = np.dtype( [
    ('y', np.int16), ('x', np.int16),
    ('curr_arr', np.int16, (4)),
    ('edited_arr', np.int16, (4))
] )
repeat_points = np.zeros((0), dtype=dtype_for_repeat_points) # an array of points 
                                                             # that change repeatedly
repeat_points_temp = np.zeros((1), dtype=dtype_for_repeat_points)
"""***************************************"""

grid, point, coord_point_arr, repeat_points, cell_size = create_grid(grid, repeat_points, matrix_size, repeat_points_temp, cell_size=0.09)


my_file = open(f'{matrix_size}.txt', "w+")
for string in reversed(grid):
    my_file.write(str(string) + '\n')
my_file.close()

my_file = open(f'repeat_of_{matrix_size}.txt', "w+")
for string in reversed(repeat_points):
    my_file.write(str(string) + '\n')
my_file.close()

end_time = t.time()
elapsed_time = end_time - start_time
print(f"Время выполнения программы: {elapsed_time} секунд")
print(f"Размер клетки: {cell_size}")



draw_grid_for_grafic(cell_size, start_point_polygon)
plt.show()