#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd


# In[3]:


# 1 Получить от объекта Series показатели описательной статистики 
state = np.random.RandomState(46)
 #mu, sigma, 1000
s = pd.Series(state.normal(10, 5, 1000))
pkz = s.describe()
print(pkz)


# In[5]:


# 2 Узнать частоту уникальных элементов объекта Series (гистограмма)
  
data = 'abcdefghikxcv'
len_series = 30
s = pd.Series(np.take(list(data), np.random.randint(len(data), size=len_series)))
 
# считаем число вхождений
ans = s.value_counts()
 
print(ans)


# In[6]:


# 3 Заменить все элементы объекта Series на "Other", кроме двух наиболее часто встречающихся
 
state = np.random.RandomState(48)
s = pd.Series(state.randint(low=1, high=5, size=[13]))
print(s.value_counts())
s[~s.isin(s.value_counts().index[:2])] = 'Other'
print(s)


# In[8]:


# 4 Создать объект Series в индексах дата каждый день 2019 года, в значениях случайное значение
 
# 1. Найти сумму всех вторников
# 2. Для каждого месяца найти среднее значение
 
dti = pd.date_range(start='2019-01-01', end='2019-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)
 
# 1
ans1 = s[s.index.weekday == 2].sum()
print('Сумма всех "вторников"', ans1)
print()
 
# 2
ans2 = s.resample('M').mean()
print('Средние значения по месяцам:\n', ans2)
print()


# In[9]:


# 5 Преобразовать объект Series в DataFrame заданной формы (shape)
 
s = pd.Series(np.random.randint(low=1, high=10, size=[35]))
 
# преобразование в reshape
r = (7, 5)
 
if r[0] * r[1] != len(s):
    sys.exit('не возможно применить reshape')
    
df = pd.DataFrame(s.values.reshape(r))
 
print(df)


# In[10]:


# 6 Найти индексы объекта Series кратные 3
 
s = pd.Series(np.random.randint(low=1, high=10, size=[7]))
 
# вариант 1
ans1 = np.argwhere(s % 3==0).flatten()
print(ans1)
print(" ") 
# вариант 2
ans2 = s[s % 3 == 0].index
print(ans2)


# In[14]:


# 7 Получить данные по индексам объекта Series
 
s = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
p = [0, 4, 8, 14, 20, 10]
 
# вариант 1
ans1 = s[p]
print(ans1)
print(" ") 
# вариант 2 
ans2 = s.take(p)
print(ans2)


# In[15]:


# 8 Объединить два объекта Series вертикально и горизонтально
 
s1 = pd.Series(range(5))
s2 = pd.Series(list('abcde'))
 
ans_vertical = s1.append(s2)
ans_horizontal = pd.concat([s1, s2], axis=1)
 
print(ans_vertical)
print(" ") 
print(ans_horizontal)


# In[16]:


# 9 Получить индексы объекта Series A, данные которых содержатся в объелте Series B
 
s1 = pd.Series([5, 34, 2, 1, 4, 11, 13, 8, 7])
s2 = pd.Series([1, 5, 13, 2])
 
# вариант 1 (медленный)
ans1 = np.asarray([np.where(i == s1)[0].tolist()[0] for i in s2])
print(ans1)
 
# вариант 2 (медленный)
ans2 = np.asarray([pd.Index(s1).get_loc(i) for i in s2])
print(ans2)
 
# вариант 3 (быстрый)
ans3 = np.argwhere(s1.isin(s2)).flatten()
print(ans3)


# In[18]:


# 10 Получить объект Series B, который содержит элементы без повторений объекта A
 
s = pd.Series(np.random.randint(low=1, high=10, size=[10]))
print(ans)
print(" ")
ans = pd.Series(s.unique())
print(ans)


# In[20]:


# 11 Преобразовать каждый символ объекта Series в верхний регистр
 
s = pd.Series(['life', 'is', 'beautiful'])
 
# преобразование данных Series в строку
s = pd.Series(str(i) for i in s)
 
# вариант 1
ans1 = s.map(lambda x: x.title())
print(ans1)
print(" ") 
# вариант 2
ans2 = pd.Series(i.title() for i in s)
print(ans2)


# In[21]:


# 12 Рассчитать количество символов в объекте Series
 
s = pd.Series(['one', 'two', 'three', 'four', 'five'])

# преобразование в строковый тип
s = pd.Series(str(i) for i in s)
print(s)
print(" ")
# вариант 1
ans1 = np.asarray(s.map(lambda x: len(x)))
print(ans1)
print(" ") 
# вариант 2
ans2 = np.asarray([len(i) for i in s])
print(ans2)


# In[22]:


# 13 Найти разность между объектом Series и смещением объекта Series на n
 
n = 1
 
s = pd.Series([1, 5, 7, 8, 12, 15, 17])
 
ans = s.diff(periods=n)
 
print(ans)


# In[23]:


# 14 Преобразовать разыне форматы строк объекта Series в дату
 
s = pd.Series(['2017/01/01', '2015-02-02', '15 Jan 2019'])
 
ans = pd.to_datetime(s)
 
print(ans)


# In[24]:


#15 все данные должны иметь одинаковый формат (часто бывает выгрузка из SQL)
s = pd.Series(['14.02.2019', '22.01.2019', '01.03.2019'])
 
# преобразование в дату
ans = pd.to_datetime(s, format='%d.%m.%Y')
 
print(ans)


# In[26]:


# 16 Получить год, месяц, день, день недели, номер дня в году от объекта Series (string)
 
from dateutil.parser import parse
 
s = pd.Series(['01 Jan 2019', '02-02-2011', '20120303', '2013/04/04', '2018-12-31'])
 
# парсим в дату и время
s_ts = s.map(lambda x: parse(x, yearfirst=True))
 
# получаем года
print(s_ts.dt.year)
 
# получаем месяца
print(s_ts.dt.month)
 
# получаем дни
print(s_ts.dt.day)
 
# получаем номер недели
print(s_ts.dt.weekofyear)
 
# получаем номер дня в году
print(s_ts.dt.dayofyear)


# In[27]:


# 17 Отобрать элементы объекта Series, кторые содержат не менее двух гласных
 
from collections import Counter
 
s = pd.Series(['Яблоко', 'Orange', 'Plan', 'Python', 'Апельсин', 'Стол', 'Reliance'])
mask = s.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiouаоиеёэыуюя')]) >= 2)
ans = s[mask]
print(ans)


# In[28]:


# 18 Отобрать e-маилы из объекта Series
 
import re
 
emails = pd.Series(['test text @test.com', 'test@mail.ru', 'test.2ru', 'test@pp'])
pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mask = emails.map(lambda x: bool(re.match(pattern, x)))
ans = emails[mask]
print(ans)


# In[29]:


# 19 Получить среднее значение каждого уникального объекта Series s1 через "маску" другого объекта Series s2
n = 10
s1 = pd.Series(np.random.choice(['dog', 'cat', 'horse', 'bird'], n))
s2 = pd.Series(np.linspace(1,n,n))
ans = s2.groupby(s1).mean()
print(ans)


# In[ ]:




