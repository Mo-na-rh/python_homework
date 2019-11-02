#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Импортировать NumPy под именем np
import numpy as np


# In[29]:


# Напечатать версию и конфигурацию
print(np.__version__)
np.show_config()


# In[4]:


# Создать вектор (одномерный массив) размера 10, заполненный нулями
Z = np.zeros(10)
print(Z)


# In[28]:


# Создать вектор размера 10, заполненный единицами
vec = np.ones(10)
print(vec)


# In[27]:


# Создать вектор размера 10, заполненный числом 2.5
vec = np.full(10, 2.5)
print(vec)


# In[ ]:


# Как получить документацию о функции numpy.add из командной строки?
# python3 -c "import numpy; numpy.info(numpy.add)" 


# In[ ]:


# Создать вектор размера 10, заполненный нулями, но пятый элемент равен 1
vec = np.zeros(10)
vec[4] = 1
print(vec)


# In[26]:


# Создать вектор со значениями от 10 до 49
vec = np.arange(10,50)
print(vec)


# In[25]:


# Развернуть вектор (первый становится последним)
vec = np.arange(50)
print(vec)
print(" ")
vec = vec[::-1]
print(vec)


# In[9]:


# Создать матрицу (двумерный массив) 3x3 со значениями от 0 до 8
ar = np.arange(9).reshape(3,3)
print(ar)


# In[10]:


# Найти индексы ненулевых элементов в [1,2,0,0,4,0]
nz = np.nonzero([1,2,0,0,4,0])
print(nz)


# In[11]:


# Создать 3x3 единичную матрицу
ar = np.eye(3)
print(ar)


# In[13]:


# Создать массив 3x3x3 со случайными значениями
Ar = np.random.random((3,3,3))
print(Ar)


# In[15]:


# Создать массив 10x10 со случайными значениями, найти минимум и максимум
ar = np.random.random((10,10))
print(ar)
arMin, arMax = ar.min(), ar.max()
print(arMin, arMax)


# In[17]:


# Создать случайный вектор размера 30 и найти среднее значение всех элементов
vec = np.random.random(30)
print(vec)
m = vec.mean()
print(m)


# In[19]:


# Создать матрицу с 0 внутри, и 1 на границах
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)


# In[20]:


# Выяснить результат следующих выражений
Z = 0 * np.nan
print(Z)
Z = np.nan == np.nan
print(Z)
Z = np.inf > np.nan
print(Z)
Z = np.nan - np.nan
print(Z)
Z = 0.3 == 3 * 0.1
print(Z)


# In[22]:


# Создать 5x5 матрицу с 1,2,3,4 под диагональю
# https://www.w3resource.com/numpy/array-creation/diag.php
Z = np.diag(np.arange(1, 5), k=-1)
print(Z)


# In[23]:


# Создать 8x8 матрицу и заполнить её в шахматном порядке
Z = np.zeros((8,8), dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


# In[ ]:




