import requests as r
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


class User():
  def __init__(self):
      pass
    
  def signup(self):
    self.uuid = r.get('http://159.196.178.94:8080/signup').json()
    
  def generate_timestamps(self):
      #Average person will spend 50 secs with standard dev of 20
      s = np.random.normal(50, 20, 12)
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

    df = dataset_dict['df_general']

    df['sentences'] = df.article.apply(lambda doc : [item for sublist in [ x.splitlines() for x in nltk.tokenize.sent_tokenize(doc)] for item in sublist])
    grouped_dfs = [x for x in df.groupby('comments')]
    ls_dfs = []
    for i,x in grouped_dfs:
        ls_dfs.append([x]*i) 
    
    # Generate a dataframe where a page picking a random page is portportional to the number of comments
    normalized_df = pd.concat([item for sublist in ls_dfs for item in sublist],ignore_index=True) 

    # now we sample all the pages
    samples = normalized_df.sample(n=len(user_list)*12)
    for i, user in enumerate( user_list):
        for j in range(12):
            user.timestamps[j] = (user.timestamps[j], samples.iloc[i*12+j])
    
    return user_list

def generate_requests(user_list):

    ## First get the filter
    api = wandb.Api()
    filter_path = api.artifact('genie/potential_filter:latest').download() + '/filter.pickle'
    with open(filter_path, 'rb') as f:
        filter_model = dill.loads(f.read())['model']

    requests = [] 
    for user in user_list:
        for time, article in user.timestamps:
            tokens = gensim.utils.simple_preprocess(article.post_title)
            tokens = [x for x in tokens if x in filter_model.index]
            probs = filter_model.loc[tokens].values
            prob = np.prod(probs)/(np.prod(probs)+np.prod(1-probs))
            if prob > 0.95:
                requests.append({'offset':time, 'data':{ 'user_id':user.uuid, 'url':article.media_url, 'sentences': article.sentences}})
    return requests


def post(data,time):
    res = r.post('http://159.196.178.94:8080/',json=data)
    with open('responses', 'ab+') as f:
            dill.dump([time,res],f)
    print(time,res.elapsed)
      

def main(n_users=10):
    user_list = []
    for i in range(n_users):
        user = User()
        user.signup()
        user.generate_timestamps()
        user_list.append(user)
    user_list= generate_paths(user_list)
    requests = generate_requests(user_list)

    requests = [x for x in requests if x['offset'] < 10*60]

    try:
        os.remove('responses')
    except FileNotFoundError:
        pass
    s = BackgroundScheduler()
    s.add_executor('processpool')
    now = datetime.now()
    for request in requests:
        s.add_job(post, trigger = 'date', run_date =  now + timedelta(seconds=request['offset']),kwargs= {'data':request['data'], 'time':request['offset']})

    s.start()
    time.sleep(10*60)

    data = []
    with open('responses', 'rb') as fr:
        try:
            while True:
                data.append(dill.load(fr))
        except EOFError:
            pass

    # Now we save the responses
    df = pd.DataFrame(columns=['user_id','request_time', 'request_data', 'elapsed_time'])
    for res in data:
        request_data = json.loads(res[1].request.body)
        df.loc[len(df)] = [request_data['user_id'], res[0], request_data, res[1].elapsed]

    df.to_pickle(f'{n_users}.pickle')

if __name__ == '__main__':
    nltk.download('punkt',download_dir='./venv/nltk_data')
    if len(sys.argv) > 1:
        lengths = sys.argv[1:]
        for users in lengths:
            r.get('http://159.196.178.94:8080/reset')
            main(int(users))
         
    else:
        print('No lengths specified')

