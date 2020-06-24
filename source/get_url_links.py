import os
import datetime
import numpy as np
import json
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import Comment
import lxml
import re
import requests
from langdetect import detect_langs

def get_urls(path):
  with open(path, 'r') as file:
    s = file.read()
  s = re.sub('[\[\]\']', ' ', s)
  urls = s.split(',')
  # some found urls (not many! a few of them) are digits! Recheck yelp.com html file to resolve the issues.
  urls = [None if (url.strip() == 'None' or url.strip()[0].isdigit()) else url.strip() for url in urls]
  return urls

def get_yelp_data(path):
  with open(path, 'r') as file:
    raw_data = file.readlines()
  raw_data = map(lambda x: x.rstrip(), raw_data)
  json_data = '[' + ','.join(raw_data) + ']'
  df = pd.read_json(json_data)
  return df

def save_content_of_websites(output_path=None):
  path = os.path.join('..', '..', 'yelp_data', 'yelp_academic_dataset_business.json')
  df = get_yelp_data(path)
  print(df.info())
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
  n = len(df)
  urls = get_urls(os.path.join('..','..','yelp_data','saved_links.txt'))
  print('size of url list = {}'.format(len(urls)))
  urls = urls + list(np.nan for _ in range(n - len(urls)))

  df['url'] = urls
  df = df[pd.notnull(df['url'])]
  webpage_content = []
  c, not_found = 0, 0
  for url in df['url']:
    webpage_content.append(None)
    if c > 0 and c % 10 == 0:
      print('{}th url is processed. {} are not found'.format(c, not_found))
    try:
      page = requests.get(url, stream=True, timeout=10.0)
      page.encoding = 'utf-8'
      webpage_content[-1] = page.content
    except:
      not_found += 1
    c += 1
  df['is_eng'] = is_eng
  df['webpage_text'] = webpage_content

  #dt = str(datetime.datetime.now())
  #dt = dt[:dt.find('.')].replace(' ', '_').replace(':','-')i
  #file_name = 'business_{}.csv'.format(dt)
  if output_path is None:
    file_name = 'business.csv'

    folder_path = os.path.join('..', '..', 'yelp_data', 'updated')
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
  else:
    file_path = os.path.join(output_path)

  df.to_csv(file_path, index=False, header=True)
  print('Check out{}'.format(str(file_path)))
0
  #pd.set_option('display.max_columns', None)

if __name__ == '__main__':
  save_content_of_websites()
