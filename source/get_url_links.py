import os
import re


def get_saved_urls():
  with open(os.path.join('..','..','yelp_data','saved_links.txt'), 'r') as file:
    s = file.read()
  s = re.sub('[\[\]\']', ' ', s)
  urls = s.split(',')
  # some found urls (not many! a few of them) are digits! Recheck yelp.com html file to resolve the issues.
  urls = [None if (url.strip() == 'None' or url.strip()[0].isdigit()) else url.strip() for url in urls]
  return urls

