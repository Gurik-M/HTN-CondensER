#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cohere
import pandas as pd
import requests
import datetime
from tqdm import tqdm
pd.set_option('display.max_colwidth', None)

def get_post_titles(**kwargs):
    """ Gets data from the pushshift api. Read more: https://github.com/pushshift/api """
    base_url = f"https://api.pushshift.io/reddit/search/submission/"
    payload = kwargs
    request = requests.get(base_url, params=payload)
    return [a['title'] for a in request.json()['data']]

api_key = 'api key'

co = cohere.Client(api_key)


symptoms_examples = [
("none", "Could you please tell me what it is!"),
("chest pain", "Chest pain!"),
("blurred vision, headaches", "Occasionally blurred vision with headaches"),
("food poisoning", "I think I may have some minor food poisoning of some sorts"),
("fingers shaking", "My fingers won't stop shaking!"),
("nausea and dry mouth", "Sudden onset of nausea and dry mouth"),
]


# In[2]:


class cohereExtractor():
    def __init__(self, examples, example_labels, labels, task_desciption, example_prompt):
        self.examples = examples
        self.example_labels = example_labels
        self.labels = labels
        self.task_desciption = task_desciption
        self.example_prompt = example_prompt

    def make_prompt(self, example):
        examples = self.examples + [example]
        labels = self.example_labels + [""]
        return (self.task_desciption +
                "\n---\n".join( [examples[i] + "\n" +
                                self.example_prompt + 
                                 labels[i] for i in range(len(examples))]))

    def extract(self, example):
      extraction = co.generate(
          model='large',
          prompt=self.make_prompt(example),
          max_tokens=10,
          temperature=0.1,
          stop_sequences=["\n"])
      return(extraction.generations[0].text[:-1])


cohereMovieExtractor = cohereExtractor([e[1] for e in symptoms_examples], 
                                       [e[0] for e in symptoms_examples], [],
                                       "", 
                                       "extract the symptoms from the post:")

print(cohereMovieExtractor.make_prompt('<input text here>'))


# In[44]:


num_posts = 100

symptoms_list = get_post_titles(size=num_posts, 
      after=str(int(datetime.datetime(2015,1,1,0,0).timestamp())), 
      before=str(int(datetime.datetime(2020,1,1,0,0).timestamp())), 
      subreddit="medical_advice", 
      sort_type="score", 
      sort="desc")

# Show the list
symptoms_list


# In[45]:


results = []
for text in tqdm(symptoms_list):
    try:
        extracted_text = cohereMovieExtractor.extract(text)
        results.append(extracted_text)
    except Exception as e:
        print('ERROR: ', e)


# In[46]:


pd.DataFrame(data={'text': symptoms_list, 'extracted_text': results})


# In[ ]:




