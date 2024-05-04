import numpy as np 
import numba as nb
import matplotlib.pyplot as plt
import time as t

@nb.njit(cache=True)
def f_1(x, y):
  return -x*(x**2 - 5) - y

@nb.njit(cache=True)
def f_2(x, y):
  return x

def draw_grafic(point):
    X = point[0]
    Y = point[1]
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X, Y)

def draw_grid():
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

@nb.njit(cache=True)
def method_euler(start_point, n, h, mu):
    """ Решение методом Эйлера """
    point = np.zeros( (2, n+1) )
    x, y = start_point[0], start_point[1]
    point[0, 0], point[1, 0] = x, y
    for i in nb.prange(n):
        dx = f_1(x, y) * h / mu
        dy = f_2(x, y) * h
        x += dx
        y += dy
        point[0, i+1], point[1, i+1] = x, y
    return point

@nb.njit(cache=True)
def calc_start_point(polygon, grid_size, cell_size):
    quantity = grid_size * grid_size
    x_min, x_max = polygon[0, 0], polygon[0, 1]
    y_min, y_max = polygon[1, 0], polygon[1, 1]
    start_point = np.empty( (quantity, 2) )
    x_offset, y_offset = 0, 0 # индекс смещения по х и по у
    for i in range(quantity):
        x = x_min + (cell_size / 2) + x_offset * cell_size
        y = y_min + (cell_size / 2) + y_offset * cell_size
        x_offset += 1
        if x > x_max:
            x_offset = 0
            y_offset += 1
        start_point[i, 0] = x
        start_point[i, 1] = y
    return start_point

@nb.njit(cache=True)
def calc_grid_size(cell_size, solution):
    x_min, x_max = min(solution[0]), max(solution[0])
    y_min, y_max = min(solution[1]), max(solution[1])
    x_offset, y_offset = x_max - x_min, y_max - y_min
    grid_size = int(max(x_offset, y_offset) // cell_size + 1)
    if max(x_offset, y_offset) == x_offset:
        start_point_polygon = np.array([ [x_min, x_max], [x_min, x_max] ])
    elif max(x_offset, y_offset) == y_offset:
        start_point_polygon = np.array([ [y_min, y_max], [y_min, y_max] ])
    return grid_size, start_point_polygon

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
def numba_vstack(arr1, arr2, dtype):
    combined_length = len(arr1) + len(arr2)
    result = np.empty(combined_length, dtype=dtype)
    result[:len(arr1)] = arr1
    result[len(arr1):] = arr2
    return result

@nb.njit(cache=True)
def calc_middle_iteration(grid):
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell = grid[row, col]
            if cell['iteration'] != 0:
                cell['iteration'] //= cell['pathway']

@nb.njit(cache=True)   
def update_grid_and_repeat_points(grid, point_coords, k, new_value, repeat_points, repeat_points_temp, pathway_counter, dtype):
    y, x = int(point_coords[1]), int(point_coords[0])
    cell = grid[y, x]
    if cell['array'][k] in (0, new_value):
        cell['array'][k] = new_value
        cell['iteration'] += 1
        if cell['pathway'] <= pathway_counter:
            cell['pathway'] += 1
    else:
        point_temp = cell['array'].copy()
        point_temp[k] = new_value
        is_unique = True
        for rp in repeat_points:
            if np.array_equal(point_temp, rp['edited_arr']):
                is_unique = False
                break
        if is_unique:
            repeat_points_temp[0]['y'], repeat_points_temp[0]['x'] = y, x
            repeat_points_temp[0]['curr_arr'] = cell['array'].copy()
            repeat_points_temp[0]['edited_arr'] = point_temp
            if not is_in(repeat_points_temp[0], repeat_points):
                repeat_points = numba_vstack(repeat_points, repeat_points_temp, dtype)
    return repeat_points

@nb.njit(cache=True)
def comparison_point(arr, grid, repeat_points, repeat_points_temp, pathway_counter, dtype):
    for i in nb.prange(1, len(arr) - 1):
        prev_x, prev_y = arr[i - 1]
        curr_x, curr_y = arr[i]
        next_x, next_y = arr[i + 1]
        
        # Определение направления движения
        k = 0
        if prev_x < curr_x:
            k = 3
        elif prev_x > curr_x:
            k = 1
        if prev_y < curr_y:
            k = 2
        elif prev_y > curr_y:
            k = 0
        
        # Определение нового значения на основе следующей точки
        new_value = 0
        if next_y < curr_y:
            new_value = 3
        elif next_y > curr_y:
            new_value = 1
        elif next_x < curr_x:
            new_value = 4
        elif next_x > curr_x:
            new_value = 2
        
        # Обновление grid и repeat_points
        repeat_points = update_grid_and_repeat_points(grid, arr[i], k, new_value, repeat_points,
                                                      repeat_points_temp, pathway_counter, dtype)
    return repeat_points


#@nb.njit(cache=True)
def create_grid(grid, repeat_points, grid_size, repeat_points_temp, cell_size, start_point_arr, dtype):
    max_points = grid_size * grid_size * n  # Предполагаемое максимальное количество точек
    coord_point_arr = np.empty((max_points, 2), dtype=np.float64)  # Используйте тип данных, соответствующий вашему случаю
    point_count = 0
    
    for i in range(len(start_point_arr)):
        start_point = start_point_arr[i]
        pathway_counter = i
        point = method_euler(start_point, n, h, mu)
        draw_grafic(point)
        # Сохранение координат точек
        new_points = save_coord_point(coord_point_arr[point_count:], grid_size, cell_size, point, n)
        point_count += len(new_points)
        
        # Обновление repeat_points
        repeat_points = comparison_point(new_points, grid, repeat_points, repeat_points_temp, pathway_counter, dtype)
    
    # Обрезка неиспользованной части coord_point_arr
    coord_point_arr = coord_point_arr[:point_count]
    
    # Вычисление среднего количества итераций
    calc_middle_iteration(grid)
    
    return grid, repeat_points

start_time = t.time()


""" Глобальный переменные """

segment = np.array([0, 10])
h = 0.001 # шаг
mu = 0.1 # параметр
n = int( (segment[1] - segment[0]) / h )
#quantity_start_point = 100 # количество стартовых точек

start_point_arr = np.array( [[1, 1]] )

cell_size = 1.02 # размер клетки

solution = method_euler(start_point_arr[0], n, h, mu)

matrix_size, start_point_polygon = calc_grid_size(cell_size, solution)

start_point_arr = calc_start_point(start_point_polygon, matrix_size, cell_size)

dtype_for_matrix = np.dtype( [      # Тип для
    ('array', np.int16, (4)),       # матрицы 
    ('iteration', np.int32),
    ('pathway', np.int16)
] )

grid = np.zeros((matrix_size, matrix_size), dtype=dtype_for_matrix)

dtype_for_repeat_points = np.dtype( [       # Тип для 
    ('y', np.int16), ('x', np.int16),       # массива с
    ('curr_arr', np.int16, (4)),            # повторяющимися точками
    ('edited_arr', np.int16, (4))
] )
repeat_points = np.zeros((0), dtype=dtype_for_repeat_points) # an array of points 
                                                             # that change repeatedly
repeat_points_temp = np.zeros((1), dtype=dtype_for_repeat_points)

"""***************************************"""

grid, repeat_points = create_grid(grid, repeat_points, matrix_size, repeat_points_temp, cell_size, start_point_arr, dtype_for_repeat_points)
#draw_grid()

# # Сохранение данных grid в файл
np.savetxt(f'{matrix_size}.txt', grid[::-1], fmt='%s')

# # Сохранение данных repeat_points в файл
# np.savetxt(f'repeat_of_{matrix_size}.txt', repeat_points[::-1], fmt='%s')

#np.save(f'gridSize_{matrix_size}_cellSize_{cell_size}.npy', grid)

# Запись данных repeat_points в файл
with open(f'repeat_of_{matrix_size}.txt', "w") as my_file:
    for string in reversed(repeat_points):
        my_file.write(f'{string}\n')

end_time = t.time()
print(f"Время выполнения программы: {end_time - start_time} сек.\nРазмер клетки: {cell_size}\nРазмер матрицы: {len(grid)}")

plt.show()