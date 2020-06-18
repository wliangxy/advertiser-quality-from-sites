import numpy as np
import json
import os
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import Comment
import lxml
import re
import requests
from langdetect import detect_langs

def get_yelp_data(path):
  with open(path, 'r') as file:
    raw_data = file.readlines()
  raw_data = map(lambda x: x.rstrip(), raw_data)
  json_data = '[' + ','.join(raw_data) + ']'
  df = pd.read_json(json_data)
  return df

path = os.path.join('..', '..', 'yelp_data', 'yelp_academic_dataset_business.json')
df = get_yelp_data(path)

#print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 209393 entries, 0 to 209392
Data columns (total 14 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   business_id   209393 non-null  object 
 1   name          209393 non-null  object 
 2   address       209393 non-null  object 
 3   city          209393 non-null  object 
 4   state         209393 non-null  object 
 5   postal_code   209393 non-null  object 
 6   latitude      209393 non-null  float64
 7   longitude     209393 non-null  float64
 8   stars         209393 non-null  float64
 9   review_count  209393 non-null  int64  
 10  is_open       209393 non-null  int64  
 11  attributes    180348 non-null  object 
 12  categories    208869 non-null  object 
 13  hours         164550 non-null  object 
dtypes: float64(3), int64(2), object(9)
memory usage: 22.4+ MB
'''

with open(os.path.join('..','..','yelp_data','saved_links.txt'), 'r') as file:
  s = file.read()
  s = re.sub('[\[\]\']', ' ', s)
  urls = s.split(',')

# some found urls (not many! a few of them) are digits! Recheck yelp.com html file to resolve the issues.
urls = [None if (url.strip() == 'None' or url.strip()[0].isdigit()) else url.strip() for url in urls]

#print(urls[:5])
#['http://www.therangeatlakenorman.com/', 'None', 'http://www.felinus.ca', 'None', 'https://www.usemyguyservices.com']

n = len(df)
#print('Number of processed entries = ', len(urls))
#Number of processed entries =  58977

urls = urls + list(np.nan for _ in range(n - len(urls)))
#print('Number of entries = ', n)
#Number of entries =  209393

df['url'] = urls
#df.replace('None', np.nan, inplace=True)
df = df[pd.notnull(df['url'])]

#print('Shape of dataframe = ', df.shape)
#Shape of dataframe =  (37291, 15)

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(df.head())
'''
              business_id                          name  \
0  f9NumwFMBDn751xgFiRbNA      The Range At Lake Norman   
2  XNoUzKckATkOD1hP6vghZg                       Felinus   
4  51M2Kk903DFYI6gnB5I6SQ       USE MY GUY SERVICES LLC   
5  cKyLV5oWZJ2NudWgqs8VZw   Oasis Auto Center - Gilbert   
7  ScYkbYNkDgCneBrD9vqhCQ  Junction Tire & Auto Service   

                     address       city state postal_code   latitude  \
0            10913 Bailey Rd  Cornelius    NC       28031  35.462724   
2      3554 Rue Notre-Dame O   Montreal    QC     H4C 1P4  45.479984   
4         4827 E Downing Cir       Mesa    AZ       85205  33.428065   
5  1720 W Elliot Rd, Ste 105    Gilbert    AZ       85233  33.350399   
7        6910 E Southern Ave       Mesa    AZ       85209  33.393885   

    longitude  stars  review_count  is_open  \
0  -80.852612    3.5            36        1   
2  -73.580070    5.0             5        1   
4 -111.726648    4.5            26        1   
5 -111.827142    4.5            38        1   
7 -111.682226    5.0            18        1   

                                          attributes  \
0  {'BusinessAcceptsCreditCards': 'True', 'BikePa...   
2                                               None   
4  {'BusinessAcceptsCreditCards': 'True', 'ByAppo...   
5             {'BusinessAcceptsCreditCards': 'True'}   
7  {'BusinessAcceptsCreditCards': 'True', 'ByAppo...   

                                          categories  \
0  Active Life, Gun/Rifle Ranges, Guns & Ammo, Sh...   
2                   Pets, Pet Services, Pet Groomers   
4  Home Services, Plumbing, Electricians, Handyma...   
5  Auto Repair, Automotive, Oil Change Stations, ...   
7  Auto Repair, Oil Change Stations, Automotive, ...   

                                               hours  \
0  {'Monday': '10:0-18:0', 'Tuesday': '11:0-20:0'...   
2                                               None   
4  {'Monday': '0:0-0:0', 'Tuesday': '9:0-16:0', '...   
5  {'Monday': '7:0-18:0', 'Tuesday': '7:0-18:0', ...   
7  {'Monday': '7:30-17:0', 'Tuesday': '7:30-17:0'...   

                                                 url  
0               http://www.therangeatlakenorman.com/  
2                              http://www.felinus.ca  
4                   https://www.usemyguyservices.com  
5                         http://oasisautocenter.net  
7  http://junctiontire.net/tires-auto-repair-mesa-az  
'''

def visible_tags(item):
  return not item.parent.name in {'meta', 'head', 'script', 'style', '[document]'} and not isinstance(item, Comment)

is_eng = []
max_text_size = 20 # maximum size for language detection

for url in df['url']:
  is_eng.append(False)
  try:
    page = requests.get(url, stream=True, allow_redirects=False, headers={'Connection': 'close'}, timeout=1.0)
    page.encoding = 'utf-8' 
  except:
    continue
  soup = BeautifulSoup(page.content, 'lxml')
  texts = soup.findAll(text=True)
  visible_texts = filter(visible_tags, texts)
  visible_texts = u' '.join(s.strip() for s in visible_texts)
  if visible_texts is None: continue
  try:
    langs = detect_langs(visible_texts[:max_text_size])
    for i in range(min(2, len(langs))):
      if langs[i].lang == 'en':
        is_eng[-1] = True
  except:
    pass

print(n, sum(is_eng))

