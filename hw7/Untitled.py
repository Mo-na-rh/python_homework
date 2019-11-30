#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import sys
import numpy as np


# In[13]:


# 2. Создать объект pandas Series из листа, объекта NumPy, и словаря
src_list = list('abcde')
src_arr = np.arange(5)
src_dict = dict(zip(src_list, src_arr))
 
s1 = pd.Series(src_list)
s2 = pd.Series(src_arr)
s3 = pd.Series(src_dict)
 
print(s1)
print(" ")
print(s2)
print(" ")
print(s3)
print(" ")


# In[14]:


# 3. Преобразовать объект Series в DataFrame
 
# создаем объект Series
s = pd.Series({'a': 'one', 'b': 'two', 'c': 'three'})
 
# преобразование в DataFrame
s = s.to_frame()
 
print(s)


# In[10]:


# создать объект Series
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)


# In[11]:


# преобразовать в DataFrame
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
df = pd.DataFrame(population, columns=['population'])
print(df)


# In[15]:


# объединить несколько объектов Series в Dataframe
# Словарь площадь штатов
area_dict = {'California': 423967,
             'Texas': 695662,
             'New York': 141297,
             'Florida': 170312,
             'Illinois': 149995}
area = pd.Series(area_dict)
# Словарь население
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
# Вариант 1
print('вариант 1')
states = pd.DataFrame({'population': population, 'area': area})
print(states)
print(" ")
# Вариант 2
df1 = pd.concat([population, area], axis=1)
print('вариант 2')
print(df1)


# In[16]:


# 5. Присвоить имя индексу объекта Series
 
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
 
s.name = 'qwerty'
 
print(s)


# In[18]:


# 6. Получить элементы объекта Series A, которых нет в объекте Series B
 
s1 = pd.Series([1, 6, 3, 98, 5])
s2 = pd.Series([4, 5, 6, 7, 8])
 
# возвращает вместе с индексами
ans = s1[~s1.isin(s2)]
 
# возвращает значения
ans2 = np.setdiff1d(s1, s2, assume_unique=False)
print(ans)


# In[20]:


# 7. Получить не пересекающиеся элементы в двух объектах Series
 
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([4, 5, 6, 7, 8])
 
# возвращает вместе с индексами
 
# получаем объединенный Series без повтороений
s_union = pd.Series(np.union1d(s1, s2))
# получаем пересекающиеся данные
s_intersect = pd.Series(np.intersect1d(s1, s2))
# отбираем все данные, кроме пересекающихся
ans = s_union[~s_union.isin(s_intersect)]
 
# возвращает значения
ans2 = np.setxor1d(s1, s2, assume_unique=False)
 
print(ans)


# In[ ]:




