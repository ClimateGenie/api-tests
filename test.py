from uuid import uuid4
import requests as r
from tqdm import tqdm
import logging
from random import sample, choices
import json
from apscheduler.executors.pool import ProcessPoolExecutor
import sys
import numpy as np
import pandas as pd
import wandb
import dill
import nltk
import gensim
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import os
import time
from multiprocessing import Pool
 

logging.basicConfig(level=40, stream=sys.stdout)

class User():
  def __init__(self):
      pass
    
  def signup(self):
    self.uuid = r.get('http://159.196.178.94:8080/signup').json()
    
  def generate_timestamps(self):
      #Average person will spend 50 secs with standard dev of 20
      s = np.random.normal(50, 20, 18)
      #Negative becomes positive
      s = [np.abs(x) for x in s]
      
      count = 0
      self.timestamps = []
      for t in s:
        self.timestamps.append(count+t)
        count += t
        
        
    
    
def generate_paths(user_list):
    # Load in dataset, then distubute posts from r/news to timestamps according to the number of comments.
    # We assume that people are almost always just covering the top posts, with only a few people  reading the smaller posts
    # R news replicates multiple sources covering the same topic yeehaw
    api = wandb.Api()
    dataset_path = api.artifact('genie/recents:latest').download()+ '/dataset.pickle'
    with open(dataset_path,'rb') as f:
        dataset_dict = dill.loads(f.read())
    filter_path = api.artifact('genie/potential_filter:latest').download() + '/filter.pickle'
    with open(filter_path, 'rb') as f:
        filter_model = dill.loads(f.read())['model']

    df = dataset_dict['df_general']

    df['sentences'] = df.article.apply(lambda doc : [item for sublist in [ x.splitlines() for x in nltk.tokenize.sent_tokenize(doc)] for item in sublist])
    df['tokens'] = df.post_title.apply(lambda title: [x for x in gensim.utils.simple_preprocess(title) if x in filter_model.index])
    df['p'] = df.tokens.apply(lambda tokens: filter_model.loc[tokens].values)
    df['ps'] =  df.p.apply(lambda probs:  np.prod(probs)/(np.prod(probs)+np.prod(1-probs)))
    df['filter'] = df.ps.apply(lambda ps:  ps > 0.95)

    ls = [item for sublist in [[k]*v for k,v in df.comments.to_dict().items()] for item in sublist]

    # now we sample all the pages
    samples = df.loc[choices(ls,k=len(user_list)*18)]
    for i, user in tqdm(enumerate( user_list), total = len(user_list)):
        for j in range(18):
            user.timestamps[j] = (user.timestamps[j], samples.iloc[i*18+j])
    
    return user_list

def generate_requests(user_list):

    requests = []
    for user in tqdm(user_list, total=len(user_list)):
        for time, article in user.timestamps:
            if article['filter'] == True:
                requests.append({'offset':time, 'data':{ 'user_id':str(user.uuid), 'url':article.media_url, 'sentences': article.sentences}})
    return requests


def post(data,time):
    res = r.post('http://159.196.178.94:8080/',json=data)
    with open('responses', 'ab+') as f:
            dill.dump([time,res],f)
    logging.log(35,f'{time},{res.elapsed}')
      

def init_user(user):
    user.generate_timestamps()
    user.uuid = uuid4()
    return user

def generate_users(n_users=10):
    user_list = []
    for i in range(n_users):
        user = User()
        user_list.append(user)
    pool = Pool()
    user_list = list(tqdm(pool.imap(init_user, user_list), total= len(user_list)))
    user_list= generate_paths(user_list)
    return user_list

def main(user_list):
    requests = generate_requests(user_list)

    requests = [x for x in requests if x['offset'] < 15*60]


    try:
        os.remove('responses')
    except FileNotFoundError:
        pass

    print(f'Running Test for {len(user_list)} users')
    s = BackgroundScheduler(executors = { 'processpool': ProcessPoolExecutor(48) })
    now = datetime.now()
    for request in requests:
        s.add_job(post, trigger = 'date', run_date =  now + timedelta(seconds=request['offset']),kwargs= {'data':request['data'], 'time':request['offset']})

    s.start()
    for i in tqdm(range(15*60),total=15*60):
        time.sleep(1)

    data = []
    try: 
        with open('responses', 'rb') as fr:
            try:
                while True:
                    data.append(dill.load(fr))
            except EOFError:
                pass
    except FileNotFoundError:
        pass

    # Now we save the responses
    df = pd.DataFrame(columns=['user_id','request_time', 'request_data', 'elapsed_time'])
    for res in data:
        request_data = json.loads(res[1].request.body)
        df.loc[len(df)] = [request_data['user_id'], res[0], request_data, res[1].elapsed, res.code]

    df.to_pickle(f'{len(user_list)}.pickle')

if __name__ == '__main__':
    nltk.download('punkt',download_dir='./venv/nltk_data')
    if len(sys.argv) > 1:
        lengths = [int(x) for x in sys.argv[1:]]
        core_list = generate_users(max(lengths))
        for users in lengths:
            r.get('http://159.196.178.94:8080/reset')
            main(sample(core_list,users))

         
    else:
        print('No lengths specified')

