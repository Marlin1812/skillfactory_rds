#!/usr/bin/env python
# coding: utf-8

# Проект UNICEF — международного подразделения ООН
# Миссия - повышение уровня благополучия детей по всему миру.

# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость
# по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
# Для этого нужно создать модель, которая предсказывала бы результаты госэкзамена по математике
# для каждого ученика школы.
# Чтобы определиться с параметрами будущей модели, необходимо провести разведывательный анализ данных
# и составить отчёт по его результатам

# Описание датасета (переменные, которые содержит датасет):
# 
# 1 school — аббревиатура школы, в которой учится ученик
# 2 sex — пол ученика ('F' - женский, 'M' - мужской)
# 3 age — возраст ученика (от 15 до 22)
# 4 address — тип адреса ученика ('U' - городской, 'R' - за городом)
# 5 famsize — размер семьи('LE3' <= 3, 'GT3' >3)
# 6 Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 7 Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 8 Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 9 Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 10 Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 11 reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 12 guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 13 traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 14 studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 15 failures — количество внеучебных неудач (n, если 1<=n<3, иначе 4)
# 16 schoolsup — дополнительная образовательная поддержка (yes или no)
# 17 famsup — семейная образовательная поддержка (yes или no)
# 18 paid — дополнительные платные занятия по математике (yes или no)
# 19 activities — дополнительные внеучебные занятия (yes или no)
# 20 nursery — посещал детский сад (yes или no)
# 21 higher — хочет получить высшее образование (yes или no)
# 22 internet — наличие интернета дома (yes или no)
# 23 romantic — в романтических отношениях (yes или no)
# 24 famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 25 freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 26 goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 27 health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 28 absences — количество пропущенных занятий
# 29 score — баллы по госэкзамену по математике

# Дорожная карта проекта:
# 
# Первичный осмотр данных датасет
# Первичный анализ данных в столбцах
# Преобразовать данные (по необходимости).
# Рассмотреть распределение признака для числовых переменных, устранить выбросы.
# Оценить количество уникальных значений для номинативных переменных
# Провести корреляционный анализ количественных переменных.
# Отобрать не коррелирующие переменные.
# Проанализировать номинативные переменные и устранить те, которые не влияют на предсказываемую величину 'score'
# Cформулировать выводы относительно качества данных и тех переменных, которые будут использованы в построении модели.

# Необходимые для работы библиотеки

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from itertools import combinations
from scipy.stats import ttest_ind
import math


# In[3]:


from jupyterthemes import jtplot
jtplot.style(theme='chesterish', context='notebook', fscale=1.7, ticks=True,
             figsize=(8, 6.5))


# In[4]:


# Загрузка датасета
# data study в дальнейшем 'ds'
ds = pd.read_csv('stud_math.csv')


# In[5]:


# увеличим количество вывода строк и колонок
# показывать больше строк
pd.set_option('display.max_rows', 100)
# показывать больше колонок
pd.set_option('display.max_columns', 35)
# sns.set(style="darkgrid")
sns.set(style="whitegrid")


# In[99]:


# проверим успешно ли импортируются данные и сделаем краткие выводы
display(ds.head(3))


# In[7]:


# все столбцы считались корректно
# проверим все ли столбцы из описания датасета загрузились
ds.info()


# Первичный осмотр данных, краткие выводы:
# всего 395 записей
# загрузилось 30 столбцов (в описании 29)
# числовых - три: age, absences, score
# бинарных -  тринадцать (по два значения каждый): school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
# оценочных - тринадцать (по несколько значений): Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, famrel, freetime, goout, health
# прочие - один: studytime, granular (отсутствует в описании к датасету, имеет отрицательные значения)

# In[8]:


# посмотрим наименования колонок
ds.columns


# In[9]:


# переименуем, для удобства, колонки начинающиеся с большой буквы на маленькую
# и сложные названия в простые (studytime, granular)
ds.rename(columns={'Pstatus': 'pstatus', 'Medu': 'medu', 'Fedu': 'fedu',
                   'Mjob': 'mjob', 'Fjob': 'fjob', 'studytime, granular': 'studytime_g'}, inplace=True)


# In[100]:


# выведим и проверим названия колонок
display(ds.head(3))


# Рассмотрим все колонки датасета по отдельности

# school — школы, в которых учатся ученики

# In[11]:


# проверим сколько и каких значений содержит колонка
pd.DataFrame(ds.school.value_counts())


# In[12]:


ds.loc[:, ['school']].info()


# все данные заполнены, есть два названия школ,которые соответствуют описанию в датасет
# учащихся в школе GP в 7,5 раз больше чем MS

# age — возраст учеников (от 15 до 22)

# In[13]:


ds.age.hist()


# In[14]:


# кажется что пропущены какие-то значения между 17-18 и 19-20
# убедимся что пропусков нет
ds.age.value_counts()


# пропуски отсутствуют

# In[15]:


# построим box-plot чтобы проверить на наличие выбросов
sns.boxplot(data=ds.age)


# In[16]:


# boxp-lot предлагает выкинуть выброс возраста 22
# проведем анализ границ и расчитаем их значения, чтобы определить границу возможного выброса
median = ds.age.median()
IQR = ds.age.quantile(0.75) - ds.age.quantile(0.25)
perc25 = ds.age.quantile(0.25)
perc75 = ds.age.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25),
      '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


ds.age.loc[ds.age <= 22].hist(bins=6,
                              range=(14.5, 22.5),
                              color='red',
                              label='выбросы')

ds.age.loc[ds.age.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=6,
                                                                    range=(
                                                                        14.5, 22.5),
                                                                    color='green',
                                                                    label='IQR')

plt.legend()


# граница выброса действительно на 21

# In[17]:


# удаляем значение 22
ds.loc[ds['age'] == 22.0, 'age'] = np.nan


# In[18]:


# построим box-plot чтобы убедится что выбросов нет
sns.boxplot(data=ds.age)


# распределение признака асимметричное (имеет длинный хвост справа) 
# выбросов нет (значение 22 удалено)
# большинство учеников в возрасте 16-17 лет

# In[19]:


# посмотрим какая связь между возрастом и успеваимостью
display(pd.DataFrame(ds.groupby(['age']).score.agg(
    ['count', 'mean', 'max', 'min', 'median'])))


# можно сделать вывод, что чем старше учащийся тем ниже успеваемость, но нужно проверить корреляцию

# medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[20]:


pd.DataFrame(ds.medu.value_counts())


# In[21]:


ds.loc[:, ['medu']].info()


# содержит пять уникальных значений согласно описанию датасет 
# три строки в которых есть пропуски, но заполнить пропуски пока не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем, так как их влияние минимально

# fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[22]:


pd.DataFrame(ds.fedu.value_counts())


# есть одно значение которое отсутствует в описании датасета 40.0
# похоже на ошибку при внесении информации

# In[23]:


# заменим его на правильное значение 4.0
ds.loc[ds['fedu'] == 40.0, 'fedu'] = 4.0


# In[24]:


# проверим
pd.DataFrame(ds.fedu.value_counts())


# содержит пять уникальных значений как в описании датасет 
# 24 строки в которых есть пропуски, с

# mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)

# In[25]:


pd.DataFrame(ds.mjob.value_counts())


# In[26]:


ds.loc[:, ['mjob']].info()


# содержит пять уникальных значений как в описании датасет 
# 19 строк в которых есть пропуски, но заполнить пропуски пока не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем, так как их влияние минимально

# fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)

# In[27]:


pd.DataFrame(ds.fjob.value_counts())


# In[28]:


ds.loc[:, ['fjob']].info()


# 
# содержит пять уникальных значений как в описании датасет
# 36 строк в которых пропуски 
# можно заполнить наиболее часто встречающимися значениями (mode), заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# лучше провести анализ до конца и потом, если будет необходимость вернутся к этому столбцу чтобы заполнить пропуски
# удалять пропуски пока не будем

# reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)

# In[29]:


pd.DataFrame(ds.reason.value_counts())


# In[30]:


ds.loc[:, ['reason']].info()


# содержит четыре уникальных значения как в описании датасет 
# 17 строк в которых пропуски, но но заполнить пропуски пока не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем

# guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)

# In[31]:


pd.DataFrame(ds.guardian.value_counts())


# In[32]:


ds.loc[:, ['guardian']].info()


# содержит три уникальных значения как в описании датасет
# 31 строка в которых есть пропуски 
# можно заполнить наиболее часто встречающимися значениями (mode), заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# лучше провести анализ до конца и потом, если будет необходимость вернутся к этому столбцу чтобы заполнить пропуски
# удалять пропуски пока не будем

# traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)

# In[33]:


pd.DataFrame(ds.traveltime.value_counts())


# In[34]:


ds.loc[:, ['traveltime']].info()


# содержит четыре уникальных значения как в описании датасет 
# 28 строк в которых есть пропуски 
# можно заполнить наиболее часто встречающимися значениями (mode), заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# лучше провести анализ до конца и потом, если будет необходимость вернутся к этому столбцу чтобы заполнить пропуски
# удалять пропуски пока не будем

# studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов) + studytime_g

# In[35]:


# посмотрим на значения этих колонок
ds.loc[:, ['studytime', 'studytime_g']]


# прослеживается взаимосвязь между значениями этих колонок 

# In[36]:


# посмотрим распределение значений этих столбцов
display(pd.DataFrame(ds.studytime.value_counts()),
        pd.DataFrame(ds.studytime_g.value_counts()))


# In[37]:


# проверим корреляцию двух столбцов
ds['studytime'].corr(ds['studytime_g'])


# присутствует 100% обратная корреляция между значениями studytime и studytime_g

# In[38]:


# проведем дальнейший анализ studytime (studytime_g пока не будем, так как его нет в описательной части датасета)
ds.loc[:, ['studytime']].info()


# studytime - содержит четыре уникальных значения 
# 7 строк в которых есть пропуски, но заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# лучше провести анализ до конца и потом, если будет необходимость вернутся к этому столбцу чтобы заполнить пропуски
# удалять пропуски пока не будем
# studytime_g - на 100% обратно скоррелирован со studytime и его можно удалить, но пока не будем его удалять

# failures — количество внеучебных неудач (n, если 1<=n<3, иначе 0)

# In[39]:


pd.DataFrame(ds.failures.value_counts())


# содержит четыре уникальных значения, но не соответствуют значениям указанным в датасете (n, если 1<=n<3, иначе 4)
# видимо в описании закралась ошибка, оставляем так как есть (без значения 4)

# In[40]:


ds.loc[:, ['failures']].info()


# 22 строки в которых есть пропуски, можно заменить пропуски, но самое часто встречающееся значение 0
# удалять пропуски пока не будем

# famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)

# In[41]:


pd.DataFrame(ds.famrel.value_counts())


# одно значение отсутствует в описании датасет -1.0
# скорее всего это ошибка при внесении информации

# In[42]:


# заменим его на 1.0
ds.loc[ds['famrel'] == -1.0, 'famrel'] = 1.0


# In[43]:


# проверим
pd.DataFrame(ds.famrel.value_counts())


# In[44]:


ds.loc[:, ['famrel']].info()


# содержит пять уникальных значений как в описании датасет 
# 27 строк в которых сеть пропуски, заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем

# freetime - свободное время после школы (от 1 - очень мало до 5 - очень мого)

# In[45]:


pd.DataFrame(ds.freetime.value_counts())


# In[46]:


ds.loc[:, ['freetime']].info()


# содержит пять уникальных значения как в описании датасет
# 11 строк в которых есть пропуски, но заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем

# goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)

# In[47]:


pd.DataFrame(ds.goout.value_counts())


# In[48]:


ds.loc[:, ['goout']].info()


# содержит пять уникальных значения как в описании датасет 
# 8 строк в которых есть пропуски, но заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем

# health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)

# In[49]:


pd.DataFrame(ds.health.value_counts())


# In[50]:


ds.loc[:, ['health']].info()


# содержит пять уникальных значения как в описании датасета 
# 15 строк в которых есть пропуски,  но заполнить пропуски на данный момент не представляется возможным, так как вариантов для заполнения слишком много
# удалять пропуски пока не будем

# absences — количество пропущенных занятий

# In[51]:


ds.absences.hist()


# In[52]:


# построим box-plot чтобы проверить на наличие выбросов
sns.boxplot(data=ds.absences)


# In[53]:


# проведем анализ границ и расчитаем их значения, чтобы определить границу возможного выброса
median = ds.absences.median()
IQR = ds.absences.quantile(0.75) - ds.absences.quantile(0.25)
perc25 = ds.absences.quantile(0.25)
perc75 = ds.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25),
      '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


ds.absences.loc[ds.absences <= 385].hist(bins=25,
                                         range=(-1, 386),
                                         color='red',
                                         label='выбросы')

ds.absences.loc[ds.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=25,
                                                                              range=(-1,
                                                                                     386),
                                                                              color='green',
                                                                              label='IQR')

plt.legend()


# очень похоже на ассимитричное распределение
# проверим это, добавим к распределению небольшую погрешность смещения

# In[54]:


f = 0.001
absences_n = ds.absences.apply(lambda x: math.log(x+f))


# In[55]:


# построим box-plot чтобы проверить на наличие выбросов
sns.boxplot(data=absences_n)


# выбросов нет 
# удалять ничего не будем

# In[56]:


ds.absences.describe()


# распределение признака ассиметричное
# выбросов нет
# большинство пропусков находятся в пределах от 0 до 8

# score — баллы по госэкзамену по математике

# In[57]:


ds.score.hist()


# на первый взгляд cлевапотенциальный выброс
# 

# In[58]:


# построим box-plot чтобы проверить на наличие выбросов
sns.boxplot(data=ds.score)


# In[59]:


# сделаем анализ границ и расчитаем их точные значения на предмет возможных выбросов
median = ds.score.median()
IQR = ds.score.quantile(0.75) - ds.absences.quantile(0.25)
perc25 = ds.score.quantile(0.25)
perc75 = ds.score.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25),
      '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


ds.score.loc[ds.score <= 101].hist(bins=21,
                                   range=(-1, 101),
                                   color='red',
                                   label='выбросы')

ds.score.loc[ds.score.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=21,
                                                                        range=(-1,
                                                                               101),
                                                                        color='green',
                                                                        label='IQR')

plt.legend()


# In[60]:


ds.score.value_counts()


# много нулевых значений 37

# In[61]:


# посмотрим на распределение без нулевых значений
score_n = ds.score.apply(lambda x: x if x > 0 else np.nan)


# In[62]:


score_n.hist(bins=10)


# In[96]:


sns.distplot(score_n, bins=10)


# In[64]:


# сделаем анализ границ и расчитаем их точные значения
median = score_n.median()
IQR = score_n.quantile(0.75) - score_n.quantile(0.25)
perc25 = score_n.quantile(0.25)
perc75 = score_n.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25),
      '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))


score_n.loc[score_n <= 101].hist(bins=10,
                                 range=(19, 101),
                                 color='red',
                                 label='выбросы')
score_n.loc[score_n.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=10,
                                                                      range=(
                                                                          19, 101),
                                                                      color='green',
                                                                      label='IQR')

plt.legend()


# распределение без ноля демонстрирует, что дисперсия увеличена и распределение выглядит вытянутым в бока при такой ситуации оно может вбирать в себя выбросы которые будут возникать рядом
# скорее всего при внесении данных возникли ошибки - нулевые значения, и данные значения присутствует в достаточно большом количестве 37 
# так как у нас модель должна прогнозировать группу риска, то заменим нулевые значения на минимальное значение 10.0 (миниммальная оценка не попадающая в выбросы), таким образом мы расширяем потенциальную группу риска

# In[65]:


# заменим нулевые значения на минимальное значение 10.0
ds.loc[ds['score'] == 0.0, 'score'] = 10.0


# In[66]:


# проверим
ds.score.value_counts()


# In[67]:


ds.score.hist(bins=10)


# распределение значений нормальное с большой дисперсией
# выбросов нет 
# заменили значения 0.0 на минимально возможное 10.0, чтобы расширить потенциальную группу риска

# проанализируем группу категориальных признаков

# In[68]:


list_n = []
bin_columns = ['address', 'famsize', 'pstatus',
               'schoolsup', 'famsup', 'paid', 'activities',
               'nursery', 'higher', 'internet', 'romantic']


# In[69]:


for elem in bin_columns:
    a = ds[elem].unique()
    a = a[~pd.isnull(a)]
    if len(a) == 2:
        list_n.append([a[0], a[1]])
    else:
        print(
            f"со столбцом {elem} произошла ошибка. Уникальные значения: ds[elem].values")
list_n


# In[70]:


list_n[5] = ['yes', 'no']
list_n


# ошибок в написании возможных вариантов значений нет

# In[71]:


# приведем yes и no к единообразию
list_n[4] = ['yes', 'no']
list_n[5] = ['yes', 'no']
list_n[6] = ['yes', 'no']
list_n[10] = ['yes', 'no']


# In[72]:


# проверяем
list_n


# In[73]:


# заменяем
for i in range(len(bin_columns)):
    elem = bin_columns[i]
    ds.loc[ds[elem] == list_n[i][0], elem] = 1.0
    ds.loc[ds[elem] == list_n[i][1], elem] = 0.0


# In[74]:


# проверяем
list_n = []
for elem in bin_columns:
    a = ds[elem].unique()
    a = a[~pd.isnull(a)]
    if len(a) == 2:
        list_n.append([a[0], a[1]])
    else:
        print(
            f"со столбцом {elem} произошла ошибка. Уникальные значения: ds[elem].values")
list_n


# In[75]:


# ошибок нет
# продолжаем анализ значений
ds[bin_columns].describe()


# In[76]:


temp = ds[bin_columns].describe()


# In[77]:


# расчитаем кол-во пропущенных значений
395-temp.T['count']


# In[78]:


# для критериев с пропусками больше 10 выведем частоту встречаемости значений
ds.pivot_table(['address', 'famsize', 'pstatus', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic'], 'sex', aggfunc=['count'])


# критерии приведены к численным значениям и готовы к загрузкам в модель
# много пропусков в критерии pstatus 45
# заполнить пропуски для всех критериев на данный момент не представляется возможным, так как значения внутри распределены без очевидных перекосов

# Корреляционный анализ

# числовые переменные
# выясним какие колонки лучше всего коррелируют со score
# это поможет определить, какие параметры стоит оставить для модели, а какие — исключить

# In[79]:


ds_num = ds[['age', 'absences', 'score']]


# In[80]:


sns.pairplot(ds_num, kind='reg')


# In[81]:


# матрица корреляций:
ds_num.corr()


# оставляем эти два критерия потому что они не сколлерированны между собой
# между возрастом есть обратная корреляция, чем выше возраст тем ниже значение score

# категориальные переменные
# посмотрим различаются ли распределения в зависимости от значения этих переменных

# In[82]:


# для удобства составим списки этих значений
bin_columns.append('school')
rating_columns = ('medu', 'fedu', 'mjob', 'fjob',
                  'reason', 'guardian', 'traveltime', 'studytime',
                            'failures', 'famrel', 'freetime', 'goout', 'health')
all_columns = []
all_columns.extend(bin_columns)
all_columns.extend(rating_columns)


# анализ категориальных переменных

# In[83]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=column, y='score',
                data=ds.loc[ds.loc[:, column].isin(
                    ds.loc[:, column].value_counts().index[:])],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[84]:


# box-plot категорийных
for col in bin_columns:
    get_boxplot(col)


# кажется что плотности распределения различаются для следующих распределений:
# schoolsup
# nursery
# higher 
# нужно посмотреть, что покажет тест Стьюдента

# анализ оценочных категориальных переменных

# In[85]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x=column, y='score',
                data=ds.loc[ds.loc[:, column].isin(
                    ds.loc[:, column].value_counts().index[:])],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[86]:


# box plot для оценочных
for col in rating_columns:
    get_boxplot(col)


# кажется что плотности распределения существенно различаются для следующих распределений:
# medu
# fedu
# mjob
# fjob
# studytime
# failures
# goout
# health
# посмотрим что покажет тест Стьюдента

# тест Стьюдента по всем категориальным признакам
# проверим, есть ли статистическая разница в распределении оценок по всем категориальным признакам 
# проверим нулевую гипотезу о том,что распределения score по различным параметрам одинаковы

# In[87]:


def get_stat_dif(column):
    cols = ds.loc[:, column].value_counts().index[:]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        ttest = ttest_ind(ds.loc[ds.loc[:, column] == comb[0], 'score'].dropna(),
                          ds.loc[ds.loc[:, column] == comb[1], 'score'].dropna()).pvalue
#         print(f"для столбца {column} ttest:= {ttest}")
#         print(f"                     comb:= {combinations_all}")
        if ttest <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[88]:


for elem in all_columns:
    get_stat_dif(elem)


# достаточно отличаются 8 параметров: address, schoolsup, higher, romantic, medu, mjob, failures, goout оставим эти переменные в датасете
# всего получилось 10 переменных , которые возможно оказывают влияние на score: age, absences, address, schoolsup, higher, romantic, medu, mjob, failures, goout

# In[89]:


ds_model = ds.loc[:, ['age', 'absences', 'address', 'schoolsup',
                      'higher', 'romantic', 'medu', 'mjob', 'failures', 'goout', 'score']]


# In[90]:


# проверяем
ds_model.head(3)


# In[91]:


# проверим нет ли дублей (сильно скоррелированных столбцов)
ds_model.corr()


# In[92]:


# построим тепловую карту
plt.subplots(figsize=(10, 10))
sns.heatmap(ds_model.corr(), square=True, annot=True, linewidths=0.1)


# визуальный осмотр значений по этим столбцам показал, что значений очень близких к 1 или -1 нет, значит все хорошо

# Выводы
# в результате проведенного EDA данных датасета для модели, которая предсказывала бы результаты госэкзамена по математике для каждого ученика школы были получены следующие выводы:
# в данных достаточно много пустых значений, только 3 столбца из 29 заполнены полностью
# в некоторых процент пропусков доходит до 12%
# выбросы: в колонке age значение 22 удалено
# замены: в колонке score произведена замена значения 0.0 на минимально допустимое значение 10.0, чтобы расширить круг потенциальных учеников, которые могут попасть в группу риска
# гипотезы:
# чем больше возраст тем ниже может быть score
# чем больше неудач по другим предметам тем ниже может быть score
# чем больше ученик проводит времени с друзьями тем ниже может быть score
# чем лучше образование родителей тем выше может быть score
# отобраны 10 критериев, которые предлагается использовать для построения модели: age, absences, address, schoolsup, higher, romantic, medu, mjob, failures, goout

# In[93]:


print(full)


# In[ ]:





# In[ ]:





# In[ ]:




