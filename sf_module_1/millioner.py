{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['.ipynb_checkpoints', 'Api vk.ipynb.txt', 'API_IT.NET.USER.ZS_DS2_en_csv_v2_889089.csv', 'Clear Date.ipynb', 'data', 'data.csv', 'data.json', 'data_sf.csv', 'Game.ipynb', 'Hello World.ipynb', 'log.csv', 'Matrix.ipynb', 'Merge table.ipynb', 'Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_889089.csv', 'Metadata_Indicator_API_IT.NET.USER.ZS_DS2_en_csv_v2_889089.csv', 'movies.csv', 'movies_example.txt', 'Pandas.ipynb', 'Pandas2.ipynb', 'Projeсt Millioner.ipynb', 'Python A3 (23.02.2020).ipynb', 'Python B1-B3 (08.03.2020).ipynb', 'Python B5-B6 (21.03.2020).ipynb', 'Python-B6 .ipynb', 'ratings.csv', 'ratings_example.txt', 'sample.csv', 'Star Trek.ipynb', 'users.csv', '[SF-DST] Movies IMBD v3.0.ipynb', '[SF-DST] Movies IMBD v3.0.py', 'Введение_в_программирование_циклы.ipynb', 'вводный урок (3).ipynb', 'вводный урок.txt', 'Знакомство с pandas (3).txt', 'Знакомство с pandas.txt', 'парсинг сайтов.ipynb', 'Строки, файлы и регулярные выражения.ipynb', 'Строки.ipynb', 'функции в python.ipynb.txt', 'что такое список.ipynb']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "from collections import Counter\n",
    "print(os.listdir(\"../SF\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>popularity</th>\n",
       "      <th>budget</th>\n",
       "      <th>revenue</th>\n",
       "      <th>original_title</th>\n",
       "      <th>cast</th>\n",
       "      <th>director</th>\n",
       "      <th>tagline</th>\n",
       "      <th>overview</th>\n",
       "      <th>runtime</th>\n",
       "      <th>genres</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>release_date</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>tt0369610</td>\n",
       "      <td>32.985763</td>\n",
       "      <td>150000000</td>\n",
       "      <td>1513528810</td>\n",
       "      <td>Jurassic World</td>\n",
       "      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>\n",
       "      <td>Colin Trevorrow</td>\n",
       "      <td>The park is open.</td>\n",
       "      <td>Twenty-two years after the events of Jurassic ...</td>\n",
       "      <td>124</td>\n",
       "      <td>Action|Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Universal Studios|Amblin Entertainment|Legenda...</td>\n",
       "      <td>6/9/2015</td>\n",
       "      <td>5562</td>\n",
       "      <td>6.5</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>tt1392190</td>\n",
       "      <td>28.419936</td>\n",
       "      <td>150000000</td>\n",
       "      <td>378436354</td>\n",
       "      <td>Mad Max: Fury Road</td>\n",
       "      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>\n",
       "      <td>George Miller</td>\n",
       "      <td>What a Lovely Day.</td>\n",
       "      <td>An apocalyptic story set in the furthest reach...</td>\n",
       "      <td>120</td>\n",
       "      <td>Action|Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>\n",
       "      <td>5/13/2015</td>\n",
       "      <td>6185</td>\n",
       "      <td>7.1</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>tt2908446</td>\n",
       "      <td>13.112507</td>\n",
       "      <td>110000000</td>\n",
       "      <td>295238201</td>\n",
       "      <td>Insurgent</td>\n",
       "      <td>Shailene Woodley|Theo James|Kate Winslet|Ansel...</td>\n",
       "      <td>Robert Schwentke</td>\n",
       "      <td>One Choice Can Destroy You</td>\n",
       "      <td>Beatrice Prior must confront her inner demons ...</td>\n",
       "      <td>119</td>\n",
       "      <td>Adventure|Science Fiction|Thriller</td>\n",
       "      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>\n",
       "      <td>3/18/2015</td>\n",
       "      <td>2480</td>\n",
       "      <td>6.3</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>tt2488496</td>\n",
       "      <td>11.173104</td>\n",
       "      <td>200000000</td>\n",
       "      <td>2068178225</td>\n",
       "      <td>Star Wars: The Force Awakens</td>\n",
       "      <td>Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...</td>\n",
       "      <td>J.J. Abrams</td>\n",
       "      <td>Every generation has a story.</td>\n",
       "      <td>Thirty years after defeating the Galactic Empi...</td>\n",
       "      <td>136</td>\n",
       "      <td>Action|Adventure|Science Fiction|Fantasy</td>\n",
       "      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>\n",
       "      <td>12/15/2015</td>\n",
       "      <td>5292</td>\n",
       "      <td>7.5</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>tt2820852</td>\n",
       "      <td>9.335014</td>\n",
       "      <td>190000000</td>\n",
       "      <td>1506249360</td>\n",
       "      <td>Furious 7</td>\n",
       "      <td>Vin Diesel|Paul Walker|Jason Statham|Michelle ...</td>\n",
       "      <td>James Wan</td>\n",
       "      <td>Vengeance Hits Home</td>\n",
       "      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>\n",
       "      <td>137</td>\n",
       "      <td>Action|Crime|Thriller</td>\n",
       "      <td>Universal Pictures|Original Film|Media Rights ...</td>\n",
       "      <td>4/1/2015</td>\n",
       "      <td>2947</td>\n",
       "      <td>7.3</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     imdb_id  popularity     budget     revenue                original_title  \\\n",
       "0  tt0369610   32.985763  150000000  1513528810                Jurassic World   \n",
       "1  tt1392190   28.419936  150000000   378436354            Mad Max: Fury Road   \n",
       "2  tt2908446   13.112507  110000000   295238201                     Insurgent   \n",
       "3  tt2488496   11.173104  200000000  2068178225  Star Wars: The Force Awakens   \n",
       "4  tt2820852    9.335014  190000000  1506249360                     Furious 7   \n",
       "\n",
       "                                                cast          director  \\\n",
       "0  Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...   Colin Trevorrow   \n",
       "1  Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...     George Miller   \n",
       "2  Shailene Woodley|Theo James|Kate Winslet|Ansel...  Robert Schwentke   \n",
       "3  Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...       J.J. Abrams   \n",
       "4  Vin Diesel|Paul Walker|Jason Statham|Michelle ...         James Wan   \n",
       "\n",
       "                         tagline  \\\n",
       "0              The park is open.   \n",
       "1             What a Lovely Day.   \n",
       "2     One Choice Can Destroy You   \n",
       "3  Every generation has a story.   \n",
       "4            Vengeance Hits Home   \n",
       "\n",
       "                                            overview  runtime  \\\n",
       "0  Twenty-two years after the events of Jurassic ...      124   \n",
       "1  An apocalyptic story set in the furthest reach...      120   \n",
       "2  Beatrice Prior must confront her inner demons ...      119   \n",
       "3  Thirty years after defeating the Galactic Empi...      136   \n",
       "4  Deckard Shaw seeks revenge against Dominic Tor...      137   \n",
       "\n",
       "                                      genres  \\\n",
       "0  Action|Adventure|Science Fiction|Thriller   \n",
       "1  Action|Adventure|Science Fiction|Thriller   \n",
       "2         Adventure|Science Fiction|Thriller   \n",
       "3   Action|Adventure|Science Fiction|Fantasy   \n",
       "4                      Action|Crime|Thriller   \n",
       "\n",
       "                                production_companies release_date  vote_count  \\\n",
       "0  Universal Studios|Amblin Entertainment|Legenda...     6/9/2015        5562   \n",
       "1  Village Roadshow Pictures|Kennedy Miller Produ...    5/13/2015        6185   \n",
       "2  Summit Entertainment|Mandeville Films|Red Wago...    3/18/2015        2480   \n",
       "3          Lucasfilm|Truenorth Productions|Bad Robot   12/15/2015        5292   \n",
       "4  Universal Pictures|Original Film|Media Rights ...     4/1/2015        2947   \n",
       "\n",
       "   vote_average  release_year  \n",
       "0           6.5          2015  \n",
       "1           7.1          2015  \n",
       "2           6.3          2015  \n",
       "3           7.5          2015  \n",
       "4           7.3          2015  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(r'c:\\Users\\Lurie036\\SF\\data.csv')\n",
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1890"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "popularity      1.560611e+00\n",
       "budget          5.450696e+07\n",
       "revenue         1.552890e+08\n",
       "runtime         1.096534e+02\n",
       "vote_count      7.856667e+02\n",
       "vote_average    6.140899e+00\n",
       "release_year    2.007862e+03\n",
       "dtype: float64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Предобработка датасета"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 277,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls = [] # создадим список с ответами. сюда будем добавлять ответы по мере прохождения теста\n",
    "# сюда можем вписать создание новых колонок в датасете"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. У какого фильма из списка самый большой бюджет?\n",
    "Варианты ответов:\n",
    "1. The Dark Knight Rises (tt1345836)\n",
    "2. Spider-Man 3 (tt0413300)\n",
    "3. Avengers: Age of Ultron (tt2395427)\n",
    "4. The Warrior's Way\t(tt1032751)\n",
    "5. Pirates of the Caribbean: On Stranger Tides (tt1298650)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "metadata": {},
   "outputs": [],
   "source": [
    "# тут вводим ваш ответ и добавлем в его список ответов (сейчас для примера стоит \"1\")\n",
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>popularity</th>\n",
       "      <th>budget</th>\n",
       "      <th>revenue</th>\n",
       "      <th>original_title</th>\n",
       "      <th>cast</th>\n",
       "      <th>director</th>\n",
       "      <th>tagline</th>\n",
       "      <th>overview</th>\n",
       "      <th>runtime</th>\n",
       "      <th>genres</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>release_date</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>491</td>\n",
       "      <td>tt1032751</td>\n",
       "      <td>0.25054</td>\n",
       "      <td>425000000</td>\n",
       "      <td>11087569</td>\n",
       "      <td>The Warrior's Way</td>\n",
       "      <td>Kate Bosworth|Jang Dong-gun|Geoffrey Rush|Dann...</td>\n",
       "      <td>Sngmoo Lee</td>\n",
       "      <td>Assassin. Hero. Legend.</td>\n",
       "      <td>An Asian assassin (Dong-gun Jang) is forced to...</td>\n",
       "      <td>100</td>\n",
       "      <td>Adventure|Fantasy|Action|Western|Thriller</td>\n",
       "      <td>Boram Entertainment Inc.</td>\n",
       "      <td>12/2/2010</td>\n",
       "      <td>74</td>\n",
       "      <td>6.4</td>\n",
       "      <td>2010</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       imdb_id  popularity     budget   revenue     original_title  \\\n",
       "491  tt1032751     0.25054  425000000  11087569  The Warrior's Way   \n",
       "\n",
       "                                                  cast    director  \\\n",
       "491  Kate Bosworth|Jang Dong-gun|Geoffrey Rush|Dann...  Sngmoo Lee   \n",
       "\n",
       "                     tagline  \\\n",
       "491  Assassin. Hero. Legend.   \n",
       "\n",
       "                                              overview  runtime  \\\n",
       "491  An Asian assassin (Dong-gun Jang) is forced to...      100   \n",
       "\n",
       "                                        genres      production_companies  \\\n",
       "491  Adventure|Fantasy|Action|Western|Thriller  Boram Entertainment Inc.   \n",
       "\n",
       "    release_date  vote_count  vote_average  release_year  \n",
       "491    12/2/2010          74           6.4          2010  "
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data[data.budget == data.budget.max()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Какой из фильмов самый длительный (в минутах)\n",
    "1. The Lord of the Rings: The Return of the King\t(tt0167260)\n",
    "2. Gods and Generals\t(tt0279111)\n",
    "3. King Kong\t(tt0360717)\n",
    "4. Pearl Harbor\t(tt0213149)\n",
    "5. Alexander\t(tt0346491)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 282,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>popularity</th>\n",
       "      <th>budget</th>\n",
       "      <th>revenue</th>\n",
       "      <th>original_title</th>\n",
       "      <th>cast</th>\n",
       "      <th>director</th>\n",
       "      <th>tagline</th>\n",
       "      <th>overview</th>\n",
       "      <th>runtime</th>\n",
       "      <th>genres</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>release_date</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>1158</td>\n",
       "      <td>tt0279111</td>\n",
       "      <td>0.469518</td>\n",
       "      <td>56000000</td>\n",
       "      <td>12923936</td>\n",
       "      <td>Gods and Generals</td>\n",
       "      <td>Stephen Lang|Jeff Daniels|Robert Duvall|Kevin ...</td>\n",
       "      <td>Ronald F. Maxwell</td>\n",
       "      <td>The nations heart was touched by...</td>\n",
       "      <td>The film centers mostly around the personal an...</td>\n",
       "      <td>214</td>\n",
       "      <td>Drama|History|War</td>\n",
       "      <td>Turner Pictures|Antietam Filmworks</td>\n",
       "      <td>2/21/2003</td>\n",
       "      <td>23</td>\n",
       "      <td>5.8</td>\n",
       "      <td>2003</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        imdb_id  popularity    budget   revenue     original_title  \\\n",
       "1158  tt0279111    0.469518  56000000  12923936  Gods and Generals   \n",
       "\n",
       "                                                   cast           director  \\\n",
       "1158  Stephen Lang|Jeff Daniels|Robert Duvall|Kevin ...  Ronald F. Maxwell   \n",
       "\n",
       "                                  tagline  \\\n",
       "1158  The nations heart was touched by...   \n",
       "\n",
       "                                               overview  runtime  \\\n",
       "1158  The film centers mostly around the personal an...      214   \n",
       "\n",
       "                 genres                production_companies release_date  \\\n",
       "1158  Drama|History|War  Turner Pictures|Antietam Filmworks    2/21/2003   \n",
       "\n",
       "      vote_count  vote_average  release_year  \n",
       "1158          23           5.8          2003  "
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data[data.runtime == data.runtime.max()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Какой из фильмов самый короткий (в минутах)\n",
    "Варианты ответов:\n",
    "\n",
    "1. Home on the Range\ttt0299172\n",
    "2. The Jungle Book 2\ttt0283426\n",
    "3. Winnie the Pooh\ttt1449283\n",
    "4. Corpse Bride\ttt0121164\n",
    "5. Hoodwinked!\ttt0443536"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 283,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>popularity</th>\n",
       "      <th>budget</th>\n",
       "      <th>revenue</th>\n",
       "      <th>original_title</th>\n",
       "      <th>cast</th>\n",
       "      <th>director</th>\n",
       "      <th>tagline</th>\n",
       "      <th>overview</th>\n",
       "      <th>runtime</th>\n",
       "      <th>genres</th>\n",
       "      <th>production_companies</th>\n",
       "      <th>release_date</th>\n",
       "      <th>vote_count</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>769</td>\n",
       "      <td>tt1449283</td>\n",
       "      <td>1.425344</td>\n",
       "      <td>30000000</td>\n",
       "      <td>14460000</td>\n",
       "      <td>Winnie the Pooh</td>\n",
       "      <td>Jim Cummings|Travis Oates|Jim Cummings|Bud Luc...</td>\n",
       "      <td>Stephen Anderson|Don Hall</td>\n",
       "      <td>Oh Pooh.</td>\n",
       "      <td>During an ordinary day in Hundred Acre Wood, W...</td>\n",
       "      <td>63</td>\n",
       "      <td>Animation|Family</td>\n",
       "      <td>Walt Disney Pictures|Walt Disney Animation Stu...</td>\n",
       "      <td>4/13/2011</td>\n",
       "      <td>174</td>\n",
       "      <td>6.8</td>\n",
       "      <td>2011</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       imdb_id  popularity    budget   revenue   original_title  \\\n",
       "769  tt1449283    1.425344  30000000  14460000  Winnie the Pooh   \n",
       "\n",
       "                                                  cast  \\\n",
       "769  Jim Cummings|Travis Oates|Jim Cummings|Bud Luc...   \n",
       "\n",
       "                      director   tagline  \\\n",
       "769  Stephen Anderson|Don Hall  Oh Pooh.   \n",
       "\n",
       "                                              overview  runtime  \\\n",
       "769  During an ordinary day in Hundred Acre Wood, W...       63   \n",
       "\n",
       "               genres                               production_companies  \\\n",
       "769  Animation|Family  Walt Disney Pictures|Walt Disney Animation Stu...   \n",
       "\n",
       "    release_date  vote_count  vote_average  release_year  \n",
       "769    4/13/2011         174           6.8          2011  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data[data.runtime == data.runtime.min()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Средняя длительность фильма?\n",
    "\n",
    "Варианты ответов:\n",
    "1. 115\n",
    "2. 110\n",
    "3. 105\n",
    "4. 120\n",
    "5. 100\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 285,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "110"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "time_avg = data.runtime.mean()\n",
    "round(time_avg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Средняя длительность фильма по медиане?\n",
    "Варианты ответов:\n",
    "1. 106\n",
    "2. 112\n",
    "3. 101\n",
    "4. 120\n",
    "5. 115\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 286,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "106"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "time_avg = data.runtime.median()\n",
    "round(time_avg)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Какой самый прибыльный фильм?\n",
    "Варианты ответов:\n",
    "1. The Avengers\ttt0848228\n",
    "2. Minions\ttt2293640\n",
    "3. Star Wars: The Force Awakens\ttt2488496\n",
    "4. Furious 7\ttt2820852\n",
    "5. Avatar\ttt0499549"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 287,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "239    Avatar\n",
       "Name: original_title, dtype: object"
      ]
     },
     "execution_count": 240,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data.profit == data.profit.max()].original_title"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Какой фильм самый убыточный?\n",
    "Варианты ответов:\n",
    "1. Supernova tt0134983\n",
    "2. The Warrior's Way tt1032751\n",
    "3. Flushed Away\ttt0424095\n",
    "4. The Adventures of Pluto Nash\ttt0180052\n",
    "5. The Lone Ranger\ttt1210819"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "491    The Warrior's Way\n",
       "Name: original_title, dtype: object"
      ]
     },
     "execution_count": 241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data.profit == data.profit.min()].original_title"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Сколько всего фильмов в прибыли?\n",
    "Варианты ответов:\n",
    "1. 1478\n",
    "2. 1520\n",
    "3. 1241\n",
    "4. 1135\n",
    "5. 1398\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "imdb_id                 1478\n",
       "popularity              1478\n",
       "budget                  1478\n",
       "revenue                 1478\n",
       "original_title          1478\n",
       "cast                    1478\n",
       "director                1478\n",
       "tagline                 1478\n",
       "overview                1478\n",
       "runtime                 1478\n",
       "genres                  1478\n",
       "production_companies    1478\n",
       "release_date            1478\n",
       "vote_count              1478\n",
       "vote_average            1478\n",
       "release_year            1478\n",
       "profit                  1478\n",
       "dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data.profit > 0].count()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9. Самый прибыльный фильм в 2008 году?\n",
    "Варианты ответов:\n",
    "1. Madagascar: Escape 2 Africa\ttt0479952\n",
    "2. Iron Man\ttt0371746\n",
    "3. Kung Fu Panda\ttt0441773\n",
    "4. The Dark Knight\ttt0468569\n",
    "5. Mamma Mia!\ttt0795421"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "600    The Dark Knight\n",
       "Name: original_title, dtype: object"
      ]
     },
     "execution_count": 252,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2 = data[data['release_year'] == 2008]\n",
    "data2[data2.revenue == data2.revenue.max()].original_title"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Самый убыточный фильм за период с 2012 по 2014 (включительно)?\n",
    "Варианты ответов:\n",
    "1. Winter's Tale\ttt1837709\n",
    "2. Stolen\ttt1656186\n",
    "3. Broken City\ttt1235522\n",
    "4. Upside Down\ttt1374992\n",
    "5. The Lone Ranger\ttt1210819\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1246    The Lone Ranger\n",
       "Name: original_title, dtype: object"
      ]
     },
     "execution_count": 253,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2 = data.query('release_year>2011 & release_year<2015')\n",
    "data['profit'] = data2.revenue - data2.budget\n",
    "data[data.profit == data.profit.min()].original_title\t"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Какого жанра фильмов больше всего?\n",
    "Варианты ответов:\n",
    "1. Action\n",
    "2. Adventure\n",
    "3. Drama\n",
    "4. Comedy\n",
    "5. Thriller"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Drama', 782)]"
      ]
     },
     "execution_count": 255,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "Counter(data.genres.str.split('|').sum()).most_common(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 12. Какого жанра среди прибыльных фильмов больше всего?\n",
    "Варианты ответов:\n",
    "1. Drama\n",
    "2. Comedy\n",
    "3. Action\n",
    "4. Thriller\n",
    "5. Adventure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Drama', 560)]"
      ]
     },
     "execution_count": 257,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "profitfilms = data[data.profit > 0]\n",
    "Counter(profitfilms.genres.str.split('|').sum()).most_common(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 13. Кто из режиссеров снял больше всего фильмов?\n",
    "Варианты ответов:\n",
    "1. Steven Spielberg\n",
    "2. Ridley Scott \n",
    "3. Steven Soderbergh\n",
    "4. Christopher Nolan\n",
    "5. Clint Eastwood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Steven Soderbergh', 13)]"
      ]
     },
     "execution_count": 301,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "Counter(data.director.str.split('|').sum()).most_common(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 14. Кто из режиссеров снял больше всего Прибыльных фильмов?\n",
    "Варианты ответов:\n",
    "1. Steven Soderbergh\n",
    "2. Clint Eastwood\n",
    "3. Steven Spielberg\n",
    "4. Ridley Scott\n",
    "5. Christopher Nolan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Ridley Scott', 12)]"
      ]
     },
     "execution_count": 259,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "profitfilms = data[data.profit > 0]\n",
    "Counter(profitfilms.director.str.split('|').sum()).most_common(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 15. Кто из режиссеров принес больше всего прибыли?\n",
    "Варианты ответов:\n",
    "1. Steven Spielberg\n",
    "2. Christopher Nolan\n",
    "3. David Yates\n",
    "4. James Cameron\n",
    "5. Peter Jackson\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Peter Jackson    5202593685\n",
       "dtype: int64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "directors = set(data.director.str.split('|').sum())\n",
    "name_dir = pd.Series({x:data[data.director.str.contains(x)].profit.sum() \n",
    "for x in directors}).sort_values(ascending = False)\n",
    "name_dir.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 16. Какой актер принес больше всего прибыли?\n",
    "Варианты ответов:\n",
    "1. Emma Watson\n",
    "2. Johnny Depp\n",
    "3. Michelle Rodriguez\n",
    "4. Orlando Bloom\n",
    "5. Rupert Grint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Emma Watson    6666245597\n",
       "dtype: int64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "actors = set(data.cast.str.split('|').sum())\n",
    "name_actor = pd.Series({x:data[data.cast.str.contains(x)].profit.sum() \n",
    "for x in actors}).sort_values(ascending = False)\n",
    "name_actor.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 17. Какой актер принес меньше всего прибыли в 2012 году?\n",
    "Варианты ответов:\n",
    "1. Nicolas Cage\n",
    "2. Danny Huston\n",
    "3. Kirsten Dunst\n",
    "4. Jim Sturgess\n",
    "5. Sami Gayle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Kirsten Dunst   -68109207.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 299,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2 = data.query('release_year == 2012')\n",
    "data['profit'] = data2.revenue - data2.budget\n",
    "actors = set(data.cast.str.split('|').sum())\n",
    "name_actor = pd.Series({x:data[data.cast.str.contains(x)].profit.sum() \n",
    "for x in actors}).sort_values(ascending = True)\n",
    "name_actor.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 18. Какой актер снялся в большем количестве высокобюджетных фильмов? (в фильмах где бюджет выше среднего по данной выборке)\n",
    "Варианты ответов:\n",
    "1. Tom Cruise\n",
    "2. Mark Wahlberg \n",
    "3. Matt Damon\n",
    "4. Angelina Jolie\n",
    "5. Adam Sandler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 283,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Matt Damon'"
      ]
     },
     "execution_count": 283,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "mean_budget = data.budget.mean()\n",
    "upper_budget = data.query(\"budget > @mean_budget\")\n",
    "# Список актеров\n",
    "actors_list = list([actor for actors in [s.split('|') for s in upper_budget.cast] for actor in actors])\n",
    "# Кол-во актеров \n",
    "name_actor = Counter(actors_list)\n",
    "name_actor.most_common(1)[0][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 19. В фильмах какого жанра больше всего снимался Nicolas Cage?  \n",
    "Варианты ответа:\n",
    "1. Drama\n",
    "2. Action\n",
    "3. Thriller\n",
    "4. Adventure\n",
    "5. Crime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Action', 16)]\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2=data[data.cast.str.match(\"Nicolas Cage\", na=False)]\n",
    "movies = data2.genres.str.split('|').sum()\n",
    "counter=collections.Counter(movies)\n",
    "print(counter.most_common(1))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 20. Какая студия сняла больше всего фильмов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Paramount Pictures\n",
    "3. Columbia Pictures\n",
    "4. Warner Bros\n",
    "5. Twentieth Century Fox Film Corporation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Universal Pictures', 173)]\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2=data[data.production_companies.str.contains(\"\", na=False)]\n",
    "b=data2.production_companies.str.split('|').sum()\n",
    "counter=collections.Counter(b)\n",
    "print(counter.most_common(1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 21. Какая студия сняла больше всего фильмов в 2015 году?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures\n",
    "2. Paramount Pictures\n",
    "3. Columbia Pictures\n",
    "4. Warner Bros\n",
    "5. Twentieth Century Fox Film Corporation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Warner Bros.', 12)]\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data2 = data.query('release_year == 2015')\n",
    "data3=data2[data2.production_companies.str.contains(\"\", na=False)]\n",
    "b=data3.production_companies.str.split('|').sum()\n",
    "counter=collections.Counter(b)\n",
    "print(counter.most_common(1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 22. Какая студия заработала больше всего денег в жанре комедий за все время?\n",
    "Варианты ответа:\n",
    "1. Warner Bros\n",
    "2. Universal Pictures (Universal)\n",
    "3. Columbia Pictures\n",
    "4. Paramount Pictures\n",
    "5. Walt Disney"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Universal Pictures', 4973287921)\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "import operator\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data2 = data[data.genres.str.match(\"Comedy\", na=False)]\n",
    "result = dict()\n",
    "for i in range(len(data2)):\n",
    "    comp = data2.iloc[i].production_companies.split('|')\n",
    "    for company in comp:\n",
    "        if company not in result:\n",
    "            result[company] = 0\n",
    "        else:\n",
    "            result[company] += data2.iloc[i].profit\n",
    "sorted_d = sorted(result.items(), key = operator.itemgetter(1))\n",
    "print(sorted_d[-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 23. Какая студия заработала больше всего денег в 2012 году?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Columbia Pictures\n",
    "4. Paramount Pictures\n",
    "5. Lucasfilm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Columbia Pictures    2501406608\n",
       "dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "companies = set(data.production_companies.str.split('|').sum())\n",
    "date = data[(data.release_year == 2012)]\n",
    "company = pd.Series({x: date[date.production_companies.str.contains(x)].profit.sum()\n",
    "for x in companies}).sort_values(ascending = False)\n",
    "company.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 24. Самый убыточный фильм от Paramount Pictures\n",
    "Варианты ответа:\n",
    "\n",
    "1. K-19: The Widowmaker tt0267626\n",
    "2. Next tt0435705\n",
    "3. Twisted tt0315297\n",
    "4. The Love Guru tt0811138\n",
    "5. The Fighter tt0964517"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>profit</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>original_title</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>K-19: The Widowmaker</td>\n",
       "      <td>-64831034</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        profit\n",
       "original_title                \n",
       "K-19: The Widowmaker -64831034"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data['production_companies'].str.contains('Paramount Pictures')].groupby(['original_title'])[['profit']].sum().sort_values(['profit'],ascending=True).head(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 25. Какой Самый прибыльный год (заработали больше всего)?\n",
    "Варианты ответа:\n",
    "1. 2014\n",
    "2. 2008\n",
    "3. 2012\n",
    "4. 2002\n",
    "5. 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>profit</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>release_year</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>2015</td>\n",
       "      <td>18668572378</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   profit\n",
       "release_year             \n",
       "2015          18668572378"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data['production_companies'].str.contains('')].groupby(['release_year'])[['profit']].sum().sort_values(['profit'],ascending=False).head(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 26. Какой Самый прибыльный год для студии Warner Bros?\n",
    "Варианты ответа:\n",
    "1. 2014\n",
    "2. 2008\n",
    "3. 2012\n",
    "4. 2010\n",
    "5. 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>profit</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>release_year</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>2014</td>\n",
       "      <td>2295464519</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  profit\n",
       "release_year            \n",
       "2014          2295464519"
      ]
     },
     "execution_count": 235,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data[data['production_companies'].str.contains('Warner Bros')].groupby(['release_year'])[['profit']].sum().sort_values(['profit'],ascending=False).head(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 27. В каком месяце за все годы суммарно вышло больше всего фильмов?\n",
    "Варианты ответа:\n",
    "1. Январь\n",
    "2. Июнь\n",
    "3. Декабрь\n",
    "4. Сентябрь\n",
    "5. Май"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(9, 227)]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['release_date'] = pd.to_datetime(data['release_date'])\n",
    "date_m = data['release_date'].dt.month\n",
    "quantity = collections.Counter(date_m)\n",
    "quantity.most_common(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 28. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)\n",
    "Варианты ответа:\n",
    "1. 345\n",
    "2. 450\n",
    "3. 478\n",
    "4. 523\n",
    "5. 381"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({True: 450, False: 1440})"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "import collections\n",
    "data['release_date'] = pd.to_datetime(data['release_date'])\n",
    "date_m = data['release_date'].dt.month.isin([6,7,8])\n",
    "quantity = collections.Counter(date_m)\n",
    "quantity"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 29. Какой режисер выпускает (суммарно по годам) больше всего фильмов зимой?\n",
    "Варианты ответов:\n",
    "1. Steven Soderbergh\n",
    "2. Christopher Nolan\n",
    "3. Clint Eastwood\n",
    "4. Ridley Scott\n",
    "5. Peter Jackson"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Peter Jackson    7\n",
       "dtype: int64"
      ]
     },
     "execution_count": 253,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['release_date'] = pd.to_datetime(data['release_date'])\n",
    "date_m = data['release_date'].dt.month.isin([12,1,2])\n",
    "data['winter_month'] = data['release_date'].dt.month.isin([12,1,2])\n",
    "directors = set(data.director.str.split('|').sum())\n",
    "b = pd.Series({x:data[data.director.str.contains(x)].winter_month.sum() for x in directors}).sort_values(ascending = False).head(1)\n",
    "b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 30. Какой месяц чаще всего по годам самый прибыльный?\n",
    "Варианты ответа:\n",
    "1. Январь\n",
    "2. Июнь\n",
    "3. Декабрь\n",
    "4. Сентябрь\n",
    "5. Май"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6    7\n",
       "dtype: int64"
      ]
     },
     "execution_count": 254,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['release_date'] = pd.to_datetime(data['release_date'])\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data['month'] = data['release_date'].dt.month\n",
    "piv = data.pivot_table(values='profit', index=['release_year'], columns=['month'], aggfunc = 'sum')\n",
    "piv.idxmax(axis = 1).value_counts().head(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 31. Названия фильмов какой студии в среднем самые длинные по количеству символов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Jim Henson Company, The\n",
    "4. Paramount Pictures\n",
    "5. Four By Two Productions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Four By Two Productions    83.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 255,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "companies = data.production_companies.str.replace(\"(\", '').str.replace(\")\", '').str.replace('+', ' ')\n",
    "production_companies = set(companies.str.split('|').sum())\n",
    "data['name_lenght'] = data.original_title.str.len()\n",
    "mean_title = pd.Series({x:data[companies.str.contains(x)].name_lenght.mean() for x in production_companies}).sort_values(ascending = False)\n",
    "mean_title.head(1)\n",
    "\n",
    "#data['companies'] = data.production_companies.str.split('|')\n",
    "#data['simbol'] = data.original_title.str.len()\n",
    "#data_new = data[['companies', 'simbol']]\n",
    "#plain = data_new.explode('companies')\n",
    "#plain.groupby(['companies'])['simbol'].mean().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 32. Названия фильмов какой студии в среднем самые длинные по количеству слов?\n",
    "Варианты ответа:\n",
    "1. Universal Pictures (Universal)\n",
    "2. Warner Bros\n",
    "3. Jim Henson Company, The\n",
    "4. Paramount Pictures\n",
    "5. Four By Two Productions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Four By Two Productions    12.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 256,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "companies = data.production_companies.str.replace(\"(\", '').str.replace(\")\", '').str.replace('+', ' ')\n",
    "production_companies = set(companies.str.split('|').sum())\n",
    "data['name_lenght'] = data.original_title.str.count(' ')+1\n",
    "mean_title = pd.Series({x:data[companies.str.contains(x)].name_lenght.mean() for x in production_companies}).sort_values(ascending = False)\n",
    "mean_title.head(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 33. Сколько разных слов используется в названиях фильмов?(без учета регистра)\n",
    "Варианты ответа:\n",
    "1. 6540\n",
    "2. 1002\n",
    "3. 2461\n",
    "4. 28304\n",
    "5. 3432"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2461"
      ]
     },
     "execution_count": 308,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "name_movies = data.original_title.str.lower().str.split()\n",
    "words = set(name_movies.sum())\n",
    "sorted(set(name_movies.sum()))\n",
    "len(words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 34. Какие фильмы входят в 1 процент лучших по рейтингу?\n",
    "Варианты ответа:\n",
    "1. Inside Out, Gone Girl, 12 Years a Slave\n",
    "2. BloodRayne, The Adventures of Rocky & Bullwinkle\n",
    "3. The Lord of the Rings: The Return of the King\n",
    "4. 300, Lucky Number Slevin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9                                          Inside Out\n",
       "34                                               Room\n",
       "118                                      Interstellar\n",
       "119                           Guardians of the Galaxy\n",
       "125                                The Imitation Game\n",
       "128                                         Gone Girl\n",
       "138                          The Grand Budapest Hotel\n",
       "370                                         Inception\n",
       "600                                   The Dark Knight\n",
       "873                                       The Pianist\n",
       "1082    The Lord of the Rings: The Return of the King\n",
       "1184                          The Wolf of Wall Street\n",
       "1192                                 12 Years a Slave\n",
       "1801                                          Memento\n",
       "Name: original_title, dtype: object"
      ]
     },
     "execution_count": 257,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data.loc[data['vote_average']>data.quantile(0.99, numeric_only=True)['vote_average']]['original_title']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 35. Какие актеры чаще всего снимаются в одном фильме вместе\n",
    "Варианты ответа:\n",
    "1. Johnny Depp & Helena Bonham Carter\n",
    "2. Hugh Jackman & Ian McKellen\n",
    "3. Vin Diesel & Paul Walker\n",
    "4. Adam Sandler & Kevin James\n",
    "5. Daniel Radcliffe & Rupert Grint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Daniel Radcliffe, Rupert Grint', 8)]"
      ]
     },
     "execution_count": 265,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "from itertools import combinations #запускаем intertools combinations\n",
    "combi = [] #создаем список\n",
    "for actor in data.cast:\n",
    "   for x in combinations(actor.split('|'), 2): # разбиваем актеров и создаем пары\n",
    "        combi.append(', '.join(x)) # добавляем в список пары актеров\n",
    "Counter(combi).most_common()[0:1] # считаем пары актеров чаще всего играющих вместе\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 36. У какого из режиссеров выше вероятность выпустить фильм в прибыли? (5 баллов)101\n",
    "Варианты ответа:\n",
    "1. Quentin Tarantino\n",
    "2. Steven Soderbergh\n",
    "3. Robert Rodriguez\n",
    "4. Christopher Nolan\n",
    "5. Clint Eastwood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer_ls.append(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>directors</th>\n",
       "      <th>movies_profit</th>\n",
       "      <th>movies_total</th>\n",
       "      <th>%</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Ridley Scott</td>\n",
       "      <td>12</td>\n",
       "      <td>12</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>Steven Spielberg</td>\n",
       "      <td>10</td>\n",
       "      <td>10</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Tim Burton</td>\n",
       "      <td>9</td>\n",
       "      <td>9</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>Antoine Fuqua</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>Peter Jackson</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>10</td>\n",
       "      <td>Brett Ratner</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>Christopher Nolan</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>Michael Bay</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            directors  movies_profit  movies_total      %\n",
       "0        Ridley Scott             12            12  100.0\n",
       "1    Steven Spielberg             10            10  100.0\n",
       "4          Tim Burton              9             9  100.0\n",
       "6       Antoine Fuqua              8             8  100.0\n",
       "7       Peter Jackson              8             8  100.0\n",
       "10       Brett Ratner              8             8  100.0\n",
       "11  Christopher Nolan              8             8  100.0\n",
       "12        Michael Bay              8             8  100.0"
      ]
     },
     "execution_count": 274,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv(\"data.csv\")\n",
    "data['profit'] = data.revenue - data.budget\n",
    "data['profitable'] = data.profit.apply(lambda x: 1 if x > 0 else 0)\n",
    "data['director'] = data.director.str.split('|')\n",
    "data_new = data[['director', 'profitable']].explode('director')\n",
    "s = data_new.groupby('director')['profitable'].sum().sort_values(ascending=False)\n",
    "c = data_new.groupby('director')['profitable'].count().sort_values(ascending=False)\n",
    "merged = pd.merge(s,c, left_index=True, right_index=True).reset_index()\n",
    "merged.columns = ['directors', 'movies_profit', 'movies_total']\n",
    "merged['%'] = (merged.movies_profit / merged.movies_total) * 100\n",
    "merged[merged['%'] == 100].head(8)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "36"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(answer_ls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>Answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>13</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>13</td>\n",
       "      <td>14</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>14</td>\n",
       "      <td>15</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>15</td>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>16</td>\n",
       "      <td>17</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>18</td>\n",
       "      <td>19</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>19</td>\n",
       "      <td>20</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>20</td>\n",
       "      <td>21</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>21</td>\n",
       "      <td>22</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>22</td>\n",
       "      <td>23</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>23</td>\n",
       "      <td>24</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>24</td>\n",
       "      <td>25</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25</td>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>26</td>\n",
       "      <td>27</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>27</td>\n",
       "      <td>28</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>28</td>\n",
       "      <td>29</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>29</td>\n",
       "      <td>30</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>30</td>\n",
       "      <td>31</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>31</td>\n",
       "      <td>32</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>32</td>\n",
       "      <td>33</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>33</td>\n",
       "      <td>34</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>34</td>\n",
       "      <td>35</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>35</td>\n",
       "      <td>36</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Id  Answer\n",
       "0    1       4\n",
       "1    2       2\n",
       "2    3       3\n",
       "3    4       2\n",
       "4    5       1\n",
       "5    6       5\n",
       "6    7       2\n",
       "7    8       1\n",
       "8    9       4\n",
       "9   10       5\n",
       "10  11       3\n",
       "11  12       1\n",
       "12  13       3\n",
       "13  14       4\n",
       "14  15       5\n",
       "15  16       1\n",
       "16  17       3\n",
       "17  18       3\n",
       "18  19       2\n",
       "19  20       1\n",
       "20  21       4\n",
       "21  22       2\n",
       "22  23       3\n",
       "23  24       1\n",
       "24  25       5\n",
       "25  26       1\n",
       "26  27       4\n",
       "27  28       2\n",
       "28  29       5\n",
       "29  30       2\n",
       "30  31       5\n",
       "31  32       5\n",
       "32  33       3\n",
       "33  34       1\n",
       "34  35       5\n",
       "35  36       4"
      ]
     },
     "execution_count": 323,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame({'Id':range(1,len(answer_ls)+1), 'Answer':answer_ls}, columns=['Id', 'Answer'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
