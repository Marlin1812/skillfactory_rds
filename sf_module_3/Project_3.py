#!/usr/bin/env python
# coding: utf-8

# ![](https://www.pata.org/wp-content/uploads/2014/09/TripAdvisor_Logo-300x119.png)
# # TripAdvisor Rating
# ## Задача - предсказать рейтинг ресторана в TripAdvisor
# 

# # import

# In[69]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
from IPython.display import display
import math
from datetime import datetime
import ast
import numpy as np
import pandas as pd

# специальный удобный инструмент для разделения датасета
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[70]:


# Установим параметры для изображений:
from matplotlib import rcParams
sns.set_context(
    "notebook",
    font_scale=1.5,
    rc={
        "figure.figsize": (11, 8),
        "axes.titlesize": 18
    }
)

rcParams['figure.figsize'] = 11, 8


# In[71]:


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42

CURRENT_DATE = '07/27/2020'


# In[72]:


# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# # DATA

# In[74]:


#DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv('main_task.csv')
df_test = pd.read_csv('kaggle_task.csv')
sample_submission = pd.read_csv('sample_submission.csv')
pd.set_option('display.max_columns', 200)


# In[75]:


df_train.info()


# In[76]:


df_train.head(3)


# In[77]:


df_test.info()


# In[78]:


df_test.head(3)


# In[79]:


sample_submission.head(3)


# In[80]:


sample_submission.info()


# In[81]:


# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
# помечаем где у нас трейн
df_train['sample'] = 1
# помечаем где у нас тест
df_test['sample'] = 0
# в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями
df_test['Rating'] = 0
# объединяем
df = df_test.append(df_train, sort=False).reset_index(drop=True)


# In[82]:


df.info()


# In[83]:


display(df.describe(include='object'))
df.describe()


# Подробнее по признакам:
# * City: Город 
# * Cuisine Style: Кухня
# * Ranking: Ранг ресторана относительно других ресторанов в этом городе
# * Price Range: Цены в ресторане в 3 категориях
# * Number of Reviews: Количество отзывов
# * Reviews: 2 последних отзыва и даты этих отзывов
# * URL_TA: страница ресторана на 'www.tripadvisor.com' 
# * ID_TA: ID ресторана в TripAdvisor
# * Rating: Рейтинг ресторана

# In[84]:


df.sample(3)


# In[85]:


df.Reviews[1]


# Как видим, большинство признаков у нас требует очистки и предварительной обработки.

# # Cleaning and Prepping Data
# Обычно данные содержат в себе кучу мусора, который необходимо почистить, для того чтобы привести их в приемлемый формат. Чистка данных — это необходимый этап решения почти любой реальной задачи.   

# ## 1. Обработка NAN
# У наличия пропусков могут быть разные причины, но пропуски нужно либо заполнить, либо исключить из набора полностью. Но с пропусками нужно быть внимательным, даже отсутствие информации может быть важным признаком!
# По этому перед обработкой NAN лучше вынести информацию о наличии пропуска как отдельный признак

# In[86]:


# Для примера я возьму столбец Number of Reviews
df['Number_of_Reviews_isNAN'] = pd.isna(
    df['Number of Reviews']).astype('uint8')


# In[87]:


# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
df['Number of Reviews'].fillna(0, inplace=True)


# ## 2. Обработка признаков
# Для начала посмотрим какие признаки у нас могут быть категориальными.

# In[88]:


# Посмотрим на признаки и к-во уникальных значений
df.nunique(dropna=False)


# ## Restaurant_id

# In[89]:


df['Ranking'].hist(figsize=(20, 10), bins=100)
plt.tight_layout()
df['Restaurant_id'].apply(lambda x: x.split('_')[1]).astype(
    int).hist(figsize=(10, 5), bins=100)
plt.tight_layout()
# Restaurant_id Очень похож на Ranking, надо проверить корреляцию и при необходимости удалить


# ## Price Range

# In[90]:


df['Price Range'].value_counts()


# In[91]:


# в переменной много пропусков 17361 (34.7%)
# заполним значения в переменной по словарю
price_dict = {'$': 10, '$$ - $$$': 100, '$$$$': 1000}
df['Price_Range'] = df['Price Range'].map(price_dict)

# Пропуски заполняем медианным значением
df['Price_Range'] = df['Price_Range'].fillna(df['Price_Range'].median())
# df.info()


# In[92]:


# Обработка признака 'Reviews'
def make_list(list_string):
    list_string = list_string.replace('nan]', "'This is Nan']")
    list_string = list_string.replace('[nan', "['This is Nan'")
    result_list = ast.literal_eval(list_string)
    return result_list


df['Reviews'] = df['Reviews'].fillna("[[], []]")
df['Reviews'] = df['Reviews'].apply(make_list)


# ## Number of days between reviews"

# In[93]:


def delta_date(row):
    if len(row[1]) == 0 or len(row[1]) == 1:
        return 0

    elif len(row[1]) == 2:
        date1 = datetime.strptime(row[1][0], '%m/%d/%Y')
        date2 = datetime.strptime(row[1][1], '%m/%d/%Y')
        return abs(date1 - date2).days


df['Days between reviews'] = df['Reviews'].apply(delta_date)


# ## The number of days that have passed since the last recall

# In[94]:


current_date_dt = datetime.strptime(CURRENT_DATE, '%m/%d/%Y')


def since_last_days(row):
    if len(row[1]) == 0:
        date = datetime.strptime('01/01/2000', '%m/%d/%Y')

    elif len(row[1]) == 1:
        date = datetime.strptime(row[1][0], '%m/%d/%Y')

    else:
        date1 = datetime.strptime(row[1][0], '%m/%d/%Y')
        date2 = datetime.strptime(row[1][1], '%m/%d/%Y')
        date = max(date1, date2)

    return (current_date_dt - date).days


df['Days since last review'] = df['Reviews'].apply(since_last_days)


# ## Number of Reviews

# In[95]:


# в переменной 3200 (6.4%) пропущенных значений
# для удобства изменим название столбца
df.rename(columns={'Number of Reviews': 'Number_of_Reviews'}, inplace=True)
# Посмотрим на гистограмму числовых признаков 'Number_of_Reviews'
df['Number_of_Reviews'].hist(figsize=(5, 3), bins=30)
plt.tight_layout()


# In[96]:


# Заполним
df['Reviews count'] = df['Reviews'].apply(lambda x: len(x[0]))


# ## Cusines

# In[97]:


#Заполняем отсутствующие значения кухонь популярными кухнями в каждом городе в количестве равным медианному 
#количеству кухонь в городе
def get_most_common(row, count):
    new = []
    c = Counter(row)
    most = c.most_common(count)

    for item in most:
        new.append(item[0])
    new_str = ','.join(new)
    return new

df['Cuisine Style'] = df['Cuisine Style'].str[2:-2].str.split("', '")
cuisines_list = df['Cuisine Style']
city_and_cuisines = pd.concat([df['City'], cuisines_list], axis=1)
city_and_cuisines = city_and_cuisines.dropna()
cities_grouped = city_and_cuisines.groupby('City').agg({'Cuisine Style': sum}).reset_index()
cities_grouped.columns = ['City', 'Cuisine List']
cities_grouped_mean = city_and_cuisines.groupby('City')['Cuisine Style'].apply(lambda x: round(np.mean(x.str.len()))).reset_index()
cities_grouped_mean.columns = ['City2', 'Cuisine Count']
new = pd.concat([cities_grouped, cities_grouped_mean], axis=1, join='inner')
new = new.drop('City2', axis=1)
new['Common Cuisine'] = new.apply(lambda x: get_most_common(x['Cuisine List'], x['Cuisine Count']), axis=1)
new = new.set_index('City')
common_cuisine_dict = new['Common Cuisine'].to_dict()
df['Cuisine Style'] = df['Cuisine Style'].fillna(df['City'].map(common_cuisine_dict))


# ## Dummy variable for Cuisines

# In[98]:


df_new = df['Cuisine Style']
df_new_dummy = df_new.apply(lambda x: pd.Series(
    [1] * len(x), index=x)).fillna(0, downcast='infer')

df = pd.merge(df, df_new_dummy, left_index=True, right_index=True, how='left')


# ## Number of kitchens in the restaurant

# In[99]:


df['Cuisine count'] = df['Cuisine Style'].str.len()


# ## Capital sity

# In[100]:


# Добавим города столицы, значения 0 или 1

capitals = ['Paris', 'Stockholm', 'London', 'Berlin',
            'Bratislava', 'Vienna', 'Rome', 'Madrid',
            'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen',
            'Amsterdam', 'Lisbon', 'Prague', 'Oslo',
            'Helsinki', 'Ljubljana', 'Athens', 'Luxembourg', ]


def is_capital(city):
    if city in capitals:
        return 1
    else:
        return 0


df['is_capital'] = df['City'].apply(is_capital)


# ## City population

# In[101]:


cities = {
    'London': 8567000,
    'Paris': 9904000,
    'Madrid': 5567000,
    'Barcelona': 4920000,
    'Berlin': 3406000,
    'Milan': 2945000,
    'Rome': 3339000,
    'Prague': 1162000,
    'Lisbon': 2812000,
    'Vienna': 2400000,
    'Amsterdam': 1031000,
    'Brussels': 1743000,
    'Hamburg': 1757000,
    'Munich': 1275000,
    'Lyon': 1423000,
    'Stockholm': 1264000,
    'Budapest': 1679000,
    'Warsaw': 1707000,
    'Dublin': 1059000,
    'Copenhagen': 1085000,
    'Athens': 3242000,
    'Edinburgh': 504966,
    'Zurich': 1108000,
    'Oporto': 1337000,
    'Geneva': 1240000,
    'Krakow': 756000,
    'Oslo': 835000,
    'Helsinki': 1115000,
    'Bratislava': 423737,
    'Luxembourg': 107260,
    'Ljubljana': 314807,
}

df['Population'] = df['City'].map(cities)


# ## Relative rating

# In[102]:


rest_count = df.groupby('City')['Restaurant_id'].count().to_dict()
df['Total count of restaurants'] = df['City'].map(rest_count)
df['Relative ranking'] = df['Ranking'] / df['Total count of restaurants']

# df.drop(['Ranking', 'rest_total_count'], axis = 1, inplace=True)


# ## Number people per restaurant

# In[103]:


df['People per restaurant'] = df['Population'] /     df['Total count of restaurants']


# ## Country

# In[104]:


countries = {
    'London': 'GB',
    'Paris': 'FR',
    'Madrid': 'ES',
    'Barcelona': 'ES',
    'Berlin': 'DE',
    'Milan': 'IT',
    'Rome': 'IT',
    'Prague': 'CZ',
    'Lisbon': 'PT',
    'Vienna': 'AT',
    'Amsterdam': 'NL',
    'Brussels': 'BE',
    'Hamburg': 'DE',
    'Munich': 'DE',
    'Lyon': 'FR',
    'Stockholm': 'SE',
    'Budapest': 'HU',
    'Warsaw': 'PL',
    'Dublin': 'IE',
    'Copenhagen': 'DK',
    'Athens': 'GR',
    'Edinburgh': 'GB',
    'Zurich': 'CH',
    'Oporto': 'PT',
    'Geneva': 'CH',
    'Krakow': 'PL',
    'Oslo': 'NO',
    'Helsinki': 'FI',
    'Bratislava': 'SK',
    'Luxembourg': 'LU',
    'Ljubljana': 'SI',
}

df['Country'] = df['City'].map(countries)

countries_le = LabelEncoder()
countries_le.fit(df['Country'])
df['Country Code'] = countries_le.transform(df['Country'])


# In[105]:


df.head(3)


# In[106]:


df['Reviews on people'] = df['Reviews count'] / df['People per restaurant']


# ## City code

# In[107]:


cities_le = LabelEncoder()
cities_le.fit(df['City'])
df['City Code'] = cities_le.transform(df['City'])


# ## Restaurant chain

# In[108]:


restaurant_chain = set()
for chain in df['Restaurant_id']:
    restaurant_chain.update(chain)


def find_item1(cell):
    if item in cell:
        return 1
    return 0


for item in restaurant_chain:
    df['Restaurant chain'] = df['Restaurant_id'].apply(find_item1)


# ## Unique id

# In[109]:


df['ID_TA code'] = df['ID_TA'].apply(lambda x: int(x[1:]))


# In[110]:


df_train['Ranking'].describe()


# # EDA 
# [Exploratory Data Analysis](https://ru.wikipedia.org/wiki/Разведочный_анализ_данных) - Анализ данных
# На этом этапе мы строим графики, ищем закономерности, аномалии, выбросы или связи между признаками.
# В общем цель этого этапа понять, что эти данные могут нам дать и как признаки могут быть взаимосвязаны между собой.
# Понимание изначальных признаков позволит сгенерировать новые, более сильные и, тем самым, сделать нашу модель лучше.

# ### Посмотрим распределение признака

# In[111]:


plt.rcParams['figure.figsize'] = (10, 7)
df_train['Ranking'].hist(bins=100)


# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

# In[112]:


df_train['City'].value_counts(ascending=True).plot(kind='barh')


# А кто-то говорил, что французы любят поесть=) Посмотрим, как изменится распределение в большом городе:

# In[113]:


df_train['Ranking'][df_train['City'] == 'London'].hist(bins=100)


# In[114]:


# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()


# Получается, что Ranking имеет нормальное распределение, просто в больших городах больше ресторанов, из-за мы этого имеем смещение.

# ### Посмотрим распределение целевой переменной

# In[115]:


df_train['Rating'].value_counts(ascending=True).plot(kind='barh')


# ### Посмотрим распределение целевой переменной относительно признака

# In[116]:


df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)


# In[117]:


df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)


# Корреляция имеющихся признаков - практически единственным признаком, коррелирующим с Rating является Ranking. Он, в свою очередь, уже имеет слабую корреляцию практически со всеми признаками

# ### Корреляция признаков
# На этом графике уже сейчас вы сможете заметить, как признаки связаны между собой и с целевой переменной.

# In[118]:


plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(df.drop(['sample'], axis=1).corr(),)


# # Data Preprocessing
# 

# #### Запускаем и проверяем что получилось

# In[119]:


object_columns = [s for s in df.columns if df[s].dtypes == 'object']
df.drop(object_columns, axis=1, inplace=True)

df_preproc = df
df_preproc.sample(10)


# In[120]:


# df_preproc.info()
df_preproc.info(verbose=True)


# In[121]:


# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.Rating.values            # наш таргет
X = train_data.drop(['Rating'], axis=1)


# **Перед тем как отправлять наши данные на обучение, разделим данные на еще один тест и трейн, для валидации. 
# Это поможет нам проверить, как хорошо наша модель работает, до отправки submissiona на kaggle.**

# In[122]:


# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[123]:


# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape


# # Model 
# Сам ML

# In[124]:


# Импортируем необходимые библиотеки:
# инструмент для создания и обучения модели
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  # инструменты для оценки точности модели


# In[125]:


# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(
    n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)


# In[126]:


# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)


# In[127]:


# Округляем результаты с точностью 0.5
def rating_round(x, base=0.5):
    return base * round(x/base)


def predict(ds):
    return np.array([rating_round(x) for x in model.predict(ds)])


y_pred = predict(X_test)


# In[128]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# In[129]:


# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10, 10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')


# # Submission
# Если все устраевает - готовим Submission на кагл

# In[130]:


test_data.sample(10)


# In[131]:


test_data = test_data.drop(['Rating'], axis=1)


# In[ ]:


sample_submission


# In[ ]:


predict_submission = model.predict(test_data)


# In[ ]:


predict_submission


# In[ ]:


sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)


# # What's next?
# Или что делать, чтоб улучшить результат:
# * Обработать оставшиеся признаки в понятный для машины формат
# * Посмотреть, что еще можно извлечь из признаков
# * Сгенерировать новые признаки
# * Подгрузить дополнительные данные, например: по населению или благосостоянию городов
# * Подобрать состав признаков
# 
# В общем, процесс творческий и весьма увлекательный! Удачи в соревновании!
# 

# In[ ]:




