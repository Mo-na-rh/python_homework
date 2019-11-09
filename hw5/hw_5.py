#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[17]:


# Дан вектор [1, 2, 3, 4, 5], построить новый вектор с тремя нулями между каждым значением
vec = np.array([1,2,3,4,5])
print(vec)
nz = 3
newVec = np.zeros((len(vec)-1)*(nz)+len(vec))
newVec[::nz+1] = vec
print(newVec)


# In[6]:


# Поменять 2 строки в матрице
ar = np.arange(25).reshape(5,5)
print(ar)
print(" ")
ar[[0,1]] = ar[[1,0]]
print(ar)


# In[9]:


# Рассмотрим набор из 10 троек, описывающих 10 треугольников (с общими вершинами), найти множество уникальных отрезков, составляющих все треугольники
triangles = np.random.randint(0,100,(10,3))
print(triangles)
print(" ")
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
print(F)
print(" ")
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)


# In[11]:


# Дан массив C; создать массив A, что np.bincount(A) == C
C = np.bincount([1,1,2,3,4,4,6])
print(C)
print(" ")
A = np.repeat(np.arange(len(C)), C)
print(A)
print(" ")
print(np.bincount(A))# повоторение чисел


# In[15]:


# Посчитать среднее, используя плавающее окно
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float) # кумулятивная сумма 
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

r = np.arange(20)
print(r)
print(" ")
print(moving_average(r))


# In[17]:


# Дан вектор Z, построить матрицу, первая строка которой (Z[0],Z[1],Z[2]), каждая последующая сдвинута на 1 (последняя (Z[-3],Z[-2],Z[-1]))
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
#Create a view into the array with the given shape and strides.

r = np.arange(10)
print(r)
print(" ")
Vec = rolling(r, 3)

print(Vec)


# In[20]:


# Инвертировать булево значение, или поменять знак у числового массива без создания нового
Z = np.random.randint(0,2,100)
np.logical_not(arr, out=arr)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(arr, out=arr)


# In[21]:


# Рассмотрим 2 набора точек P0, P1 описания линии (2D) и точку р, как вычислить расстояние от р до каждой линии i (P0[i],P1[i])
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0] - p[...,0]) * T[:,0] + (P0[:,1] - p[...,1]) * T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U * T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))


# In[22]:


#Дан массив. Написать функцию, выделяющую часть массива фиксированного размера с центром в данном элементе (дополненное значением fill если необходимо)
Z = np.random.randint(0,10, (10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P - Rs//2)
Z_stop  = (P + Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = (np.maximum(Z_start, 0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)


# In[26]:


#Посчитать ранг матрицы
Z = np.random.uniform(0,1,(10,10))

rank = np.linalg.matrix_rank(Z)
print(rank)


# In[28]:


# Найти наиболее частое значение в массиве
Z = np.random.randint(0,10,50)
print(Z)
print(" ")
print(np.bincount(Z).argmax())


# In[30]:


# Извлечь все смежные 3x3 блоки из 10x10 матрицы
Z = np.random.randint(0,5,(10,10))
print(Z)
print(" ")
n = 3
i = 1 + (Z.shape[0] - n)
j = 1 + (Z.shape[1] - n)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)


# In[32]:


# Создать подкласс симметричных 2D массивов (Z[i,j] == Z[j,i])
---


# In[33]:


# Рассмотрим множество матриц (n,n) и множество из p векторов (n,1). Посчитать сумму p произведений матриц (результат имеет размерность (n,1))
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)


# In[35]:


# Дан массив 16x16, посчитать сумму по блокам 4x4
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)


# In[36]:


# Написать игру "жизнь"
def iterate(Z):
    # Число соседей
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Применим правила
    birth = (N == 3) & (Z[1:-1,1:-1]==0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1,1:-1] == 1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100):
    print(Z)
    Z = iterate(Z)


# In[37]:


# Найти n наибольших значений в массиве
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

print (Z[np.argpartition(-Z,n)[:n]])


# In[38]:


# Построить прямое произведение массивов (все комбинации с каждым элементом)
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = map(len, arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))


# In[16]:


# Даны 2 массива A (8x3) и B (2x2). Найти строки в A, которые содержат элементы из каждой строки в B, независимо от порядка элементов в B
A = np.random.randint(0,5,(8,3))
print(A)
print(" ")
B = np.random.randint(0,5,(2,2))
print(B)
print(" ")

C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)


# In[15]:


# Дана 10x3 матрица, найти строки из неравных значений (например [2,2,3])
ar = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(ar[:,1:] == ar[:,:-1], axis=1)
U = ar[~E]
print(ar)
print(" ")
print(U)


# In[41]:


# Преобразовать вектор чисел в матрицу бинарных представлений
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))


# In[10]:


# Дан двумерный массив. Найти все различные строки
Z = np.random.randint(0, 2, (6,3))
print(Z)
print(" ")
arr = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(arr, return_index=True)
uZ = Z[idx]
print(uZ)


# In[8]:


# Даны векторы A и B, написать einsum эквиваленты функций inner, outer, sum и mul
A = np.arange(10)
print(A)
print(" ")
B = np.arange(10)
np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j', A, B)    # np.outer(A, B)o read: http://ajcr.net/Basic-guide-to-einsum/

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j', A, B)    # np.outer(A, B)


# In[ ]:




