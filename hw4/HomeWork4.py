#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[4]:


# Перемножить матрицы 5x3 и 3x2
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
ar = np.dot(np.ones((5,3)), np.ones((3,2)))
print(ar)


# In[6]:


# Дан массив, поменять знак у элементов, значения которых между 3 и 8
ar = np.arange(11)
print(ar)
ar[(3 < ar) & (ar <= 8)] *= -1
print(" ")
print(ar)


# In[7]:


# Создать 5x5 матрицу со значениями в строках от 0 до 4
ar = np.zeros((5,5))
ar = ar + np.arange(5)
print(ar)


# In[10]:


# Есть генератор, сделать с его помощью массив
# https://habr.com/ru/post/132554/     yield, what is it?
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)


# In[11]:


# Создать вектор размера 10 со значениями от 0 до 1, не включая ни то, ни другое
# 12 чисел от 0 до 1 включительно дальше крайние удаляем
ar = np.linspace(0,1,12)[1:-1]
print(ar)


# In[12]:


# Отсортировать вектор
vec = np.random.random(10)
vec.sort()
print(vec)


# In[14]:


# Проверить, одинаковы ли 2 numpy массива
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B) # Returns True if the two arrays are equal within the given tolerance; False otherwise.
print(equal)


# In[15]:


# Сделать массив неизменяемым
ar = np.zeros(10)
ar.flags.writeable = False
ar[0] = 1 # exception on this step


# In[17]:


# Дан массив 10x2 (точки в декартовой системе координат), преобразовать в полярную
ar = np.random.random((10,2))
X,Y = ar[:,0], ar[:,1]
R = np.hypot(X, Y)
T = np.arctan2(Y,X)
print(R)
print(" ")
print(T)


# In[20]:


# Заменить максимальный элемент на ноль
ar = np.random.random(10)
print(ar)
print(" ")
ar[ar.argmax()] = 0
print(ar)


# In[23]:


# Создать структурированный массив с координатами x, y на сетке в квадрате [0,1]x[0,1]
ar = np.zeros((10,10), [('x',float),('y',float)])

ar['x'], ar['y'] = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))

print(ar)


# In[24]:


# Из двух массивов сделать матрицу Коши C (Cij = 1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


# In[26]:


# Найти минимальное и максимальное значение, принимаемое каждым числовым типом numpy
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
   print(" ")
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
   print(" ")


# In[40]:


#  Напечатать все значения в массиве
np.set_printoptions(threshold=500)
Z = np.zeros((15,15))
print(Z)


# In[42]:


# Найти ближайшее к заданному значению число в заданном массиве
Z = np.arange(100)
print(Z)
print(" ")
v = np.random.uniform(0,100)
print(v)
print(" ")
index = (np.abs(Z-v)).argmin()
print(Z[index])


# In[46]:


# Создать структурированный массив, представляющий координату (x,y) и цвет (r,g,b)
ar = np.zeros(10, [ ('pos', [ ('x', float, 1), ('y', float, 1)]),('color',    [ ('r', float, 1),('g', float, 1),('b', float, 1)])])
print(ar)


# In[48]:


# Дан массив (100,2) координат, найти расстояние от каждой точки до каждой
import scipy.spatial as sp

ar = np.random.random((10,2))
print(ar)
print(" ")
D = sp.distance.cdist(ar,ar)
print(D)


# In[50]:


# Преобразовать массив из float в int
ar = np.arange(10, dtype=np.int32)
print(ar)
ar = ar.astype(np.float32, copy=False) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
print(ar)


# In[51]:


# Дан файл:
# Как прочитать его?
#1,2,3,4,5
#6,,,7,8
#,,9,10,11
#
# ar = np.genfromtxt("missing.dat", delimiter=",")


# In[53]:


# Каков эквивалент функции enumerate для numpy массивов?
ar = np.arange(9).reshape(3,3)
print(ar)
print(" ")
for index, value in np.ndenumerate(ar):
    print(index, value)
print(" ")
for index in np.ndindex(ar.shape):
    print(index, ar[index])


# In[54]:


# Сформировать 2D массив с распределением Гаусса
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.hypot(X, Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
print(G)


# In[58]:


# Случайно расположить p элементов в 2D массив
n = 10
p = 5
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False), 8)
print(Z)


# In[60]:


# Отнять среднее из каждой строки в матрице
ar = np.random.rand(5, 10)
print(ar)
print(" ")
Y = ar - ar.mean(axis=1, keepdims=True)
print(Y)


# In[62]:


# Отсортировать матрицу по n-ому столбцу
ar = np.random.randint(0,10,(3,3))
n = 1  
print(ar)
print(" ")
print(ar[ar[:,n].argsort()])


# In[65]:


# Определить, есть ли в 2D массиве нулевые столбцы
ar = np.random.randint(0,3,(3,10))
print(ar)
print(" ")
print((~ar.any(axis=0)).any())


# In[68]:


# Дан массив, добавить 1 к каждому элементу с индексом, заданным в другом массиве (осторожно с повторами)
ar = np.ones(10)
print(ar)
print(" ")
I = np.random.randint(0,len(ar),20)
print(I)
print(" ")
Z += np.bincount(I, minlength=len(Z))
print(Z)


# In[69]:


# Дан массив (w,h,3) (картинка) dtype = ubyte, посчитать количество различных цветов
w,h = 16,16
I = np.random.randint(0, 2, (h,w,3)).astype(np.ubyte)
F = I[...,0] * 256 * 256 + I[...,1] * 256 + I[...,2]
n = len(np.unique(F))
print(np.unique(I))


# In[70]:


# Дан четырехмерный массив, посчитать сумму по последним двум осям
ar = np.random.randint(0,10, (3,4,3,4))
print(ar)
print(" ")
sum = ar.reshape(ar.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)


# In[77]:


# Найти диагональные элементы произведения матриц
# Slow version
A = np.eye(3)
print(A)
B = np.arange(9).reshape(3,3)
print(B)
x = np.diag(np.dot(A, B))
print(x)
print(" ")
# Fast version
y = np.sum(A * B.T, axis=1)
print(y)


# In[ ]:




