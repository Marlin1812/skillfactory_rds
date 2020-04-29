#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter
#print(os.listdir("../SF"))


# In[2]:


data = pd.read_csv(r'c:\Users\Lurie036\SF\data.csv')
data.head(5)


# In[5]:


len(data)


# In[6]:


data.mean()


# # Предобработка датасета

# In[277]:


answer_ls = [] # создадим список с ответами. сюда будем добавлять ответы по мере прохождения теста
# сюда можем вписать создание новых колонок в датасете


# # 1. У какого фильма из списка самый большой бюджет?
# Варианты ответов:
# 1. The Dark Knight Rises (tt1345836)
# 2. Spider-Man 3 (tt0413300)
# 3. Avengers: Age of Ultron (tt2395427)
# 4. The Warrior's Way	(tt1032751)
# 5. Pirates of the Caribbean: On Stranger Tides (tt1298650)

# In[280]:


# тут вводим ваш ответ и добавлем в его список ответов (сейчас для примера стоит "1")
answer_ls.append(4)


# In[16]:


data = pd.read_csv("data.csv")
data[data.budget == data.budget.max()].original_title


# # 2. Какой из фильмов самый длительный (в минутах)
# 1. The Lord of the Rings: The Return of the King	(tt0167260)
# 2. Gods and Generals	(tt0279111)
# 3. King Kong	(tt0360717)
# 4. Pearl Harbor	(tt0213149)
# 5. Alexander	(tt0346491)

# In[282]:


answer_ls.append(2)


# In[6]:


data = pd.read_csv("data.csv")
data[data.runtime == data.runtime.max()].original_title


# # 3. Какой из фильмов самый короткий (в минутах)
# Варианты ответов:
# 
# 1. Home on the Range	tt0299172
# 2. The Jungle Book 2	tt0283426
# 3. Winnie the Pooh	tt1449283
# 4. Corpse Bride	tt0121164
# 5. Hoodwinked!	tt0443536

# In[283]:


answer_ls.append(3)


# In[7]:


data = pd.read_csv("data.csv")
data[data.runtime == data.runtime.min()].original_title


# # 4. Средняя длительность фильма?
# 
# Варианты ответов:
# 1. 115
# 2. 110
# 3. 105
# 4. 120
# 5. 100
# 

# In[285]:


answer_ls.append(2)


# In[10]:


data = pd.read_csv("data.csv")
time_avg = data.runtime.mean()
round(time_avg)
print(round(time_avg))


# # 5. Средняя длительность фильма по медиане?
# Варианты ответов:
# 1. 106
# 2. 112
# 3. 101
# 4. 120
# 5. 115
# 
# 
# 

# In[286]:


answer_ls.append(1)


# In[9]:


data = pd.read_csv("data.csv")
time_avg = data.runtime.median()
round(time_avg)
print(round(time_avg))


# # 6. Какой самый прибыльный фильм?
# Варианты ответов:
# 1. The Avengers	tt0848228
# 2. Minions	tt2293640
# 3. Star Wars: The Force Awakens	tt2488496
# 4. Furious 7	tt2820852
# 5. Avatar	tt0499549

# In[287]:


answer_ls.append(5)


# In[12]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data.profit == data.profit.max()].original_title


# # 7. Какой фильм самый убыточный?
# Варианты ответов:
# 1. Supernova tt0134983
# 2. The Warrior's Way tt1032751
# 3. Flushed Away	tt0424095
# 4. The Adventures of Pluto Nash	tt0180052
# 5. The Lone Ranger	tt1210819

# In[288]:


answer_ls.append(2)


# In[13]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data.profit == data.profit.min()].original_title


# # 8. Сколько всего фильмов в прибыли?
# Варианты ответов:
# 1. 1478
# 2. 1520
# 3. 1241
# 4. 1135
# 5. 1398
# 

# In[289]:


answer_ls.append(1)


# In[14]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data.profit > 0].count()


# # 9. Самый прибыльный фильм в 2008 году?
# Варианты ответов:
# 1. Madagascar: Escape 2 Africa	tt0479952
# 2. Iron Man	tt0371746
# 3. Kung Fu Panda	tt0441773
# 4. The Dark Knight	tt0468569
# 5. Mamma Mia!	tt0795421

# In[290]:


answer_ls.append(4)


# In[15]:


data = pd.read_csv("data.csv")
data2 = data[data['release_year'] == 2008]
data2[data2.revenue == data2.revenue.max()].original_title


# # 10. Самый убыточный фильм за период с 2012 по 2014 (включительно)?
# Варианты ответов:
# 1. Winter's Tale	tt1837709
# 2. Stolen	tt1656186
# 3. Broken City	tt1235522
# 4. Upside Down	tt1374992
# 5. The Lone Ranger	tt1210819
# 

# In[291]:


answer_ls.append(5)


# In[16]:


data = pd.read_csv("data.csv")
data2 = data.query('release_year>2011 & release_year<2015')
data['profit'] = data2.revenue - data2.budget
data[data.profit == data.profit.min()].original_title	


# # 11. Какого жанра фильмов больше всего?
# Варианты ответов:
# 1. Action
# 2. Adventure
# 3. Drama
# 4. Comedy
# 5. Thriller

# In[292]:


answer_ls.append(3)


# In[17]:


data = pd.read_csv("data.csv")
Counter(data.genres.str.split('|').sum()).most_common(1)


# # 12. Какого жанра среди прибыльных фильмов больше всего?
# Варианты ответов:
# 1. Drama
# 2. Comedy
# 3. Action
# 4. Thriller
# 5. Adventure

# In[293]:


answer_ls.append(1)


# In[18]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
profitfilms = data[data.profit > 0]
Counter(profitfilms.genres.str.split('|').sum()).most_common(1)


# # 13. Кто из режиссеров снял больше всего фильмов?
# Варианты ответов:
# 1. Steven Spielberg
# 2. Ridley Scott 
# 3. Steven Soderbergh
# 4. Christopher Nolan
# 5. Clint Eastwood

# In[294]:


answer_ls.append(3)


# In[19]:


data = pd.read_csv("data.csv")
Counter(data.director.str.split('|').sum()).most_common(1)


# # 14. Кто из режиссеров снял больше всего Прибыльных фильмов?
# Варианты ответов:
# 1. Steven Soderbergh
# 2. Clint Eastwood
# 3. Steven Spielberg
# 4. Ridley Scott
# 5. Christopher Nolan

# In[295]:


answer_ls.append(4)


# In[20]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
profitfilms = data[data.profit > 0]
Counter(profitfilms.director.str.split('|').sum()).most_common(1)


# # 15. Кто из режиссеров принес больше всего прибыли?
# Варианты ответов:
# 1. Steven Spielberg
# 2. Christopher Nolan
# 3. David Yates
# 4. James Cameron
# 5. Peter Jackson
# 

# In[296]:



answer_ls.append(5)


# In[21]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
directors = set(data.director.str.split('|').sum())
name_dir = pd.Series({x:data[data.director.str.contains(x)].profit.sum() 
for x in directors}).sort_values(ascending = False)
name_dir.head(1)


# # 16. Какой актер принес больше всего прибыли?
# Варианты ответов:
# 1. Emma Watson
# 2. Johnny Depp
# 3. Michelle Rodriguez
# 4. Orlando Bloom
# 5. Rupert Grint

# In[297]:


answer_ls.append(1)


# In[22]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
actors = set(data.cast.str.split('|').sum())
name_actor = pd.Series({x:data[data.cast.str.contains(x)].profit.sum() 
for x in actors}).sort_values(ascending = False)
name_actor.head(1)


# # 17. Какой актер принес меньше всего прибыли в 2012 году?
# Варианты ответов:
# 1. Nicolas Cage
# 2. Danny Huston
# 3. Kirsten Dunst
# 4. Jim Sturgess
# 5. Sami Gayle

# In[298]:


answer_ls.append(3)


# In[23]:


data = pd.read_csv("data.csv")
data2 = data.query('release_year == 2012')
data['profit'] = data2.revenue - data2.budget
actors = set(data.cast.str.split('|').sum())
name_actor = pd.Series({x:data[data.cast.str.contains(x)].profit.sum() 
for x in actors}).sort_values(ascending = True)
name_actor.head(1)


# # 18. Какой актер снялся в большем количестве высокобюджетных фильмов? (в фильмах где бюджет выше среднего по данной выборке)
# Варианты ответов:
# 1. Tom Cruise
# 2. Mark Wahlberg 
# 3. Matt Damon
# 4. Angelina Jolie
# 5. Adam Sandler

# In[300]:


answer_ls.append(3)


# In[24]:


data = pd.read_csv("data.csv")
mean_budget = data.budget.mean()
upper_budget = data.query("budget > @mean_budget")
# Список актеров
actors_list = list([actor for actors in [s.split('|') for s in upper_budget.cast] for actor in actors])
# Кол-во актеров 
name_actor = Counter(actors_list)
name_actor.most_common(1)[0][0]


# # 19. В фильмах какого жанра больше всего снимался Nicolas Cage?  
# Варианты ответа:
# 1. Drama
# 2. Action
# 3. Thriller
# 4. Adventure
# 5. Crime

# In[301]:


answer_ls.append(2)


# In[43]:


data = pd.read_csv("data.csv")
import collections
data2=data[data.cast.str.match("Nicolas Cage", na=False)]
movies = data2.genres.str.split('|').sum()
counter=collections.Counter(movies)
print(counter.most_common(1))


# # 20. Какая студия сняла больше всего фильмов?
# Варианты ответа:
# 1. Universal Pictures (Universal)
# 2. Paramount Pictures
# 3. Columbia Pictures
# 4. Warner Bros
# 5. Twentieth Century Fox Film Corporation

# In[302]:


answer_ls.append(1)


# In[ ]:


data = pd.read_csv("data.csv")
data2=data[data.production_companies.str.contains("", na=False)]
b=data2.production_companies.str.split('|').sum()
counter=collections.Counter(b)
print(counter.most_common(1))


# # 21. Какая студия сняла больше всего фильмов в 2015 году?
# Варианты ответа:
# 1. Universal Pictures
# 2. Paramount Pictures
# 3. Columbia Pictures
# 4. Warner Bros
# 5. Twentieth Century Fox Film Corporation

# In[303]:


answer_ls.append(4)


# In[ ]:


data = pd.read_csv("data.csv")
data2 = data.query('release_year == 2015')
data3=data2[data2.production_companies.str.contains("", na=False)]
b=data3.production_companies.str.split('|').sum()
counter=collections.Counter(b)
print(counter.most_common(1))


# # 22. Какая студия заработала больше всего денег в жанре комедий за все время?
# Варианты ответа:
# 1. Warner Bros
# 2. Universal Pictures (Universal)
# 3. Columbia Pictures
# 4. Paramount Pictures
# 5. Walt Disney

# In[304]:


answer_ls.append(2)


# In[26]:


data = pd.read_csv("data.csv")
import operator
data['profit'] = data.revenue - data.budget
data2 = data[data.genres.str.match("Comedy", na=False)]
result = dict()
for i in range(len(data2)):
    comp = data2.iloc[i].production_companies.split('|')
    for company in comp:
        if company not in result:
            result[company] = 0
        else:
            result[company] += data2.iloc[i].profit
sorted_d = sorted(result.items(), key = operator.itemgetter(1))
print(sorted_d[-1])


# # 23. Какая студия заработала больше всего денег в 2012 году?
# Варианты ответа:
# 1. Universal Pictures (Universal)
# 2. Warner Bros
# 3. Columbia Pictures
# 4. Paramount Pictures
# 5. Lucasfilm

# In[306]:


answer_ls.append(3)


# In[19]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
companies = set(data.production_companies.str.split('|').sum())
date = data[(data.release_year == 2012)]
company = pd.Series({x: date[date.production_companies.str.contains(x)].profit.sum()
for x in companies}).sort_values(ascending = False)
print(company.head(1))


# # 24. Самый убыточный фильм от Paramount Pictures
# Варианты ответа:
# 
# 1. K-19: The Widowmaker tt0267626
# 2. Next tt0435705
# 3. Twisted tt0315297
# 4. The Love Guru tt0811138
# 5. The Fighter tt0964517

# In[309]:


answer_ls.append(1)


# In[28]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data['production_companies'].str.contains('Paramount Pictures')].groupby(['original_title'])[['profit']].sum().sort_values(['profit'],ascending=True).head(1)


# # 25. Какой Самый прибыльный год (заработали больше всего)?
# Варианты ответа:
# 1. 2014
# 2. 2008
# 3. 2012
# 4. 2002
# 5. 2015

# In[310]:


answer_ls.append(5)


# In[29]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data['production_companies'].str.contains('')].groupby(['release_year'])[['profit']].sum().sort_values(['profit'],ascending=False).head(1)


# # 26. Какой Самый прибыльный год для студии Warner Bros?
# Варианты ответа:
# 1. 2014
# 2. 2008
# 3. 2012
# 4. 2010
# 5. 2015

# In[311]:


answer_ls.append(1)


# In[30]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data[data['production_companies'].str.contains('Warner Bros')].groupby(['release_year'])[['profit']].sum().sort_values(['profit'],ascending=False).head(1)


# # 27. В каком месяце за все годы суммарно вышло больше всего фильмов?
# Варианты ответа:
# 1. Январь
# 2. Июнь
# 3. Декабрь
# 4. Сентябрь
# 5. Май

# In[312]:


answer_ls.append(4)


# In[32]:


data = pd.read_csv("data.csv")
import collections
data['release_date'] = pd.to_datetime(data['release_date'])
date_m = data['release_date'].dt.month
quantity = collections.Counter(date_m)
quantity.most_common(1)


# # 28. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)
# Варианты ответа:
# 1. 345
# 2. 450
# 3. 478
# 4. 523
# 5. 381

# In[313]:


answer_ls.append(2)


# In[33]:


data = pd.read_csv("data.csv")
import collections
data['release_date'] = pd.to_datetime(data['release_date'])
date_m = data['release_date'].dt.month.isin([6,7,8])
quantity = collections.Counter(date_m)
quantity


# # 29. Какой режисер выпускает (суммарно по годам) больше всего фильмов зимой?
# Варианты ответов:
# 1. Steven Soderbergh
# 2. Christopher Nolan
# 3. Clint Eastwood
# 4. Ridley Scott
# 5. Peter Jackson

# In[314]:


answer_ls.append(5)


# In[34]:


data = pd.read_csv("data.csv")
data['release_date'] = pd.to_datetime(data['release_date'])
date_m = data['release_date'].dt.month.isin([12,1,2])
data['winter_month'] = data['release_date'].dt.month.isin([12,1,2])
directors = set(data.director.str.split('|').sum())
b = pd.Series({x:data[data.director.str.contains(x)].winter_month.sum() for x in directors}).sort_values(ascending = False).head(1)
b


# # 30. Какой месяц чаще всего по годам самый прибыльный?
# Варианты ответа:
# 1. Январь
# 2. Июнь
# 3. Декабрь
# 4. Сентябрь
# 5. Май

# In[315]:


answer_ls.append(2)


# In[35]:


data = pd.read_csv("data.csv")
data['release_date'] = pd.to_datetime(data['release_date'])
data['profit'] = data.revenue - data.budget
data['month'] = data['release_date'].dt.month
piv = data.pivot_table(values='profit', index=['release_year'], columns=['month'], aggfunc = 'sum')
piv.idxmax(axis = 1).value_counts().head(1)


# # 31. Названия фильмов какой студии в среднем самые длинные по количеству символов?
# Варианты ответа:
# 1. Universal Pictures (Universal)
# 2. Warner Bros
# 3. Jim Henson Company, The
# 4. Paramount Pictures
# 5. Four By Two Productions

# In[316]:


answer_ls.append(5)


# In[36]:


data = pd.read_csv("data.csv")
companies = data.production_companies.str.replace("(", '').str.replace(")", '').str.replace('+', ' ')
production_companies = set(companies.str.split('|').sum())
data['name_lenght'] = data.original_title.str.len()
mean_title = pd.Series({x:data[companies.str.contains(x)].name_lenght.mean() for x in production_companies}).sort_values(ascending = False)
mean_title.head(1)

#data['companies'] = data.production_companies.str.split('|')
#data['simbol'] = data.original_title.str.len()
#data_new = data[['companies', 'simbol']]
#plain = data_new.explode('companies')
#plain.groupby(['companies'])['simbol'].mean().sort_values(ascending=False)


# # 32. Названия фильмов какой студии в среднем самые длинные по количеству слов?
# Варианты ответа:
# 1. Universal Pictures (Universal)
# 2. Warner Bros
# 3. Jim Henson Company, The
# 4. Paramount Pictures
# 5. Four By Two Productions

# In[317]:


answer_ls.append(5)


# In[37]:


data = pd.read_csv("data.csv")
companies = data.production_companies.str.replace("(", '').str.replace(")", '').str.replace('+', ' ')
production_companies = set(companies.str.split('|').sum())
data['name_lenght'] = data.original_title.str.count(' ')+1
mean_title = pd.Series({x:data[companies.str.contains(x)].name_lenght.mean() for x in production_companies}).sort_values(ascending = False)
mean_title.head(1)


# # 33. Сколько разных слов используется в названиях фильмов?(без учета регистра)
# Варианты ответа:
# 1. 6540
# 2. 1002
# 3. 2461
# 4. 28304
# 5. 3432

# In[318]:


answer_ls.append(3)


# In[38]:


data = pd.read_csv("data.csv")
name_movies = data.original_title.str.lower().str.split()
words = set(name_movies.sum())
sorted(set(name_movies.sum()))
len(words)


# # 34. Какие фильмы входят в 1 процент лучших по рейтингу?
# Варианты ответа:
# 1. Inside Out, Gone Girl, 12 Years a Slave
# 2. BloodRayne, The Adventures of Rocky & Bullwinkle
# 3. The Lord of the Rings: The Return of the King
# 4. 300, Lucky Number Slevin

# In[319]:


answer_ls.append(1)


# In[39]:


data = pd.read_csv("data.csv")
data.loc[data['vote_average']>data.quantile(0.99, numeric_only=True)['vote_average']]['original_title']


# # 35. Какие актеры чаще всего снимаются в одном фильме вместе
# Варианты ответа:
# 1. Johnny Depp & Helena Bonham Carter
# 2. Hugh Jackman & Ian McKellen
# 3. Vin Diesel & Paul Walker
# 4. Adam Sandler & Kevin James
# 5. Daniel Radcliffe & Rupert Grint

# In[320]:


answer_ls.append(5)


# In[40]:


data = pd.read_csv("data.csv")
from itertools import combinations #запускаем intertools combinations
combi = [] #создаем список
for actor in data.cast:
   for x in combinations(actor.split('|'), 2): # разбиваем актеров и создаем пары
        combi.append(', '.join(x)) # добавляем в список пары актеров
Counter(combi).most_common()[0:1] # считаем пары актеров чаще всего играющих вместе


# # 36. У какого из режиссеров выше вероятность выпустить фильм в прибыли? (5 баллов)101
# Варианты ответа:
# 1. Quentin Tarantino
# 2. Steven Soderbergh
# 3. Robert Rodriguez
# 4. Christopher Nolan
# 5. Clint Eastwood

# In[321]:


answer_ls.append(4)


# In[4]:


data = pd.read_csv("data.csv")
data['profit'] = data.revenue - data.budget
data['profitable'] = data.profit.apply(lambda x: 1 if x > 0 else 0)
data['director'] = data.director.str.split('|')
data_new = data[['director', 'profitable']].explode('director')
s = data_new.groupby('director')['profitable'].sum().sort_values(ascending=False)
c = data_new.groupby('director')['profitable'].count().sort_values(ascending=False)
merged = pd.merge(s,c, left_index=True, right_index=True).reset_index()
merged.columns = ['directors', 'movies_profit', 'movies_total']
merged['%'] = (merged.movies_profit / merged.movies_total) * 100
merged[merged['%'] == 100].head(8)


# In[24]:


print('full')


# # Submission

# In[322]:


len(answer_ls)


# In[323]:


pd.DataFrame({'Id':range(1,len(answer_ls)+1), 'Answer':answer_ls}, columns=['Id', 'Answer'])


# In[ ]:




