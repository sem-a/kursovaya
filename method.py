# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:24:04 2023

@author: aleks
"""

import numpy as np
import time
import random as r
import math as m
import matplotlib.pyplot as plt


start_time = time.perf_counter()

coord_point_array = []


def func_1(x, y):
    # Функция 1 
    return -x*(x**2 - 5) - y
def func_2(x, y):
    # Функция 2
    return x

def graficSystem(X, Y):
    # Рисует график системы
    plt.plot(X, Y)

def calculation_cell_size(X, Y, matrix_size):
    # Расчет размера клетки по размеру матрицы
    cell_size = [((np.max(X) - np.min(X)) / (matrix_size - 5)), ((np.max(Y) - np.min(Y)) / (matrix_size - 5))]
    return np.max(cell_size)


def error_len(x, y, grid): 
    # Принимает: координата х, координата у, матрица
    if(x > (len(grid) - 1) or y > (len(grid) - 1) or x < 0 or y < 0):
        #print('Ошибка: Слишком маленький размер матрицы! Увеличьте размер матрицы')
        return False
    else:
        return True

    
def print_matrix(matrix):
    # Распечатать матрицу matrix
    if (matrix != 0):
        for i in range(len(matrix)):
            print(matrix[i])
    else: 
        print('Ошибка чтения матрицы')


def save_coord_point(coord_point):
    # Сохраеить точки в матрице
    coord_point_array.append(coord_point)

    
def comparison_point(arr, grid):
    # Принимает: массив с точками, матрицу
    for i in range(len(arr)-1):
        if(grid[arr[i][1]][arr[i][0]] == 0):
            grid[arr[i][1]][arr[i][0]] = [0, 0, 0, 0]
    # Заполнение массива точками
    for i in range(1, len(arr)-1):
        if(arr[i-1][1] < arr[i][1]):
            k = 0
        if(arr[i-1][1] > arr[i][1]):
            k = 2
        if(arr[i-1][0] < arr[i][0]):
            k = 3
        if(arr[i-1][0] > arr[i][0]):
            k = 1
        if(arr[i+1][1] < arr[i][1]):
            grid[arr[i][1]][arr[i][0]][k] = 1
        if(arr[i+1][1] > arr[i][1]):
            grid[arr[i][1]][arr[i][0]][k] = 3
        if(arr[i+1][0] < arr[i][0]):
            grid[arr[i][1]][arr[i][0]][k] = 4
        if(arr[i+1][0] > arr[i][0]):
            grid[arr[i][1]][arr[i][0]][k] = 2
    
def check_arr(arr, grid):
    # Принимает: массив с точками, матрицу
    for i in range(len(arr)-1):
        grid[arr[i][1]][arr[i][0]] = 1
    
    
def calc_start_point(quantity, segment):
    start_point = [[None, None]] * quantity
    for i in range(quantity):
        start_point[i] = [r.uniform(segment[0][0], segment[0][1]), r.uniform(segment[1][0], segment[1][1])]
    return start_point
    
def create_grid(matrix_size, func_1, func_2, cell_size=0):
    # Принимает: размер матрицы, размер клетки, функция 1, функция 2, размер клетки(необязательно)
    
    # Начальные условия:
    segment = [0,10]
    h = 0.001
    n = int( (segment[1] - segment[0]) / h )
    start_point = calc_start_point(1000, [[-2.6, 2.6], [-4.3, 4.3,]])
    mu = 0.1
    
    try:
        grid = [[[0, 0, 0, 0] for col in range(matrix_size)] for row in range(matrix_size)] # заполнение сетки нулями
        
        # Заполнение массивов X, Y
        for j in range(len(start_point)):
            point = [[], []] # массив содержащий [X, Y]
            for i in range(n):
                if(i == 0):
                    point[0].append(start_point[j][0] + func_1(start_point[j][0], start_point[j][1]) * h / mu)
                    point[1].append(start_point[j][1] + func_1(start_point[j][0], start_point[j][1]) * h / mu)
                point[0].append(point[0][i] + func_1(point[0][i], point[1][i]) * h / mu)
                point[1].append(point[1][i] + func_2(point[0][i], point[1][i]) * h)
        
            graficSystem(point[0], point[1])
        
            if(cell_size == 0):
                cell_size = calculation_cell_size(point[0], point[1], matrix_size) # размер клетки
                print('Размер клетки:', cell_size)
            
        
            x_temp = int( (matrix_size - 1) / 2 ) + int( point[0][0] / cell_size ) # Координата X первой точки 
            y_temp = int( (matrix_size - 1) / 2 ) + int( point[1][0] / cell_size ) # Координата У первой точки    
            # Заполнение матрицы по траектории движения    
            for i in range(2, n):
                if(error_len(x_temp, y_temp, grid)):
                    x_coor = int( (matrix_size + 1) / 2 ) + int( point[0][i] / cell_size ) # Координата Х следующей точки
                    y_coor = int( (matrix_size + 1) / 2 ) - int( point[1][i] / cell_size ) # Координата У следующей точки
                    
                    if(error_len(x_coor, y_coor, grid)):
                        if([x_temp, y_temp] != [x_coor, y_coor]):
                            save_coord_point([x_temp, y_temp]) # Сохранение точек в массив
                            
                    else:
                        continue
                    
                    x_temp = x_coor # Переприсваиваем Х
                    y_temp = y_coor # Переприсваиваем У
                else:
                    continue
            
            #check_arr(coord_point_array, grid)
            comparison_point(coord_point_array, grid) 
        
        return grid
    except:
        print('Возникла непредвиденная ошибка!')
    
        
print_matrix(create_grid(30, func_1, func_2))
plt.show()
print('Время выполнения программы:', time.perf_counter() - start_time)
