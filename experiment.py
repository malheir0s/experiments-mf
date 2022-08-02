import torch
import torch.nn as nn
import torch.nn.functional as F

from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset as surprise_Dataset
from surprise import accuracy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import sys
import json


df = pd.read_csv('../input/movielens/ml-1m/ratings.dat', header=None, names=['user', 'item', 'rating', 'timestamp'], sep='::')

from datetime import datetime
dates = df['timestamp'].apply(datetime.fromtimestamp)
#df['ts'] = dates.dt.year.astype(int)*100 + dates.dt.month.astype(int)
df['ts'] = dates.dt.year.astype(int)
df = df.sort_values(by=['timestamp']).reset_index(drop=True)

def encode_columns(col):
    keys = col.unique()
    key_to_id = {key:idx for idx, key in enumerate(keys)}
    return key_to_id

u_map = encode_columns(df['user'])
i_map = encode_columns(df['item'])

df['user_id'] = df['user'].map(u_map)
df['item_id'] = df['item'].map(i_map)




# # dataset definition
class MFDataset(Dataset):
    # load the dataset
    def __init__(self, interactions):
        # ['user_id', 'ts', 'item_id', 'popularity', 'propensity', 'rating']
        self.user_id = interactions['user_id'].values
        self.item_id = interactions['item_id'].values        
        self.rating = interactions['rating'].values

    # number of rows in the dataset
    def __len__(self):
        return len(self.user_id)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.user_id[idx], self.item_id[idx], self.rating[idx]]

class MF(nn.Module):

    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()

        self.n_users = num_users
        self.n_items = num_items
        self.emb_size = emb_size

        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, user, item):
        u_emb = self.user_emb(user)
        i_emb = self.item_emb(item)
        return torch.mul(u_emb, i_emb).sum(-1).float()

    def calculate_loss(self, pred, target, weight):
        loss = (pred - target)**2        
        return torch.mul(torch.reciprocal(weight), loss).mean()

    def predict(self, user, item):
        pred = self.forward(user, item)
        return pred

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1))
        return scores

    def full_predict(self, user):
        #test_item_emb = self.item_emb.weight.view(self.n_items, 1, self.emb_size)
        scores = torch.matmul(self.user_emb(user), self.item_emb.weight.transpose(0,1))
        return scores    


def test_model(model):
    model.eval()
    user_ids = torch.LongTensor(test['user_id'].values)
    item_ids = torch.LongTensor(test['item_id'].values)
    ratings = torch.FloatTensor(test['rating'].values)
    y_hat = model(user_ids, item_ids)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())


def weighted_mse_loss(input, target, weight):
    return torch.sum(torch.reciprocal(weight)*((input - target)**2))


def train_epocs(model, interactions, epochs=10, lr=0.001, wd=0, weights=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        
    model.train()
    
    if weights == None:
        weights = torch.ones(len(interactions))

    for _ in range(epochs):
        user_ids = torch.LongTensor(interactions['user_id'].values)
        item_ids = torch.LongTensor(interactions['item_id'].values)
        ratings = torch.FloatTensor(interactions['rating'].values)
         
        # clear the gradients
        optimizer.zero_grad()  
        # compute the model output / forward pass
        y_hat = model(user_ids, item_ids)       
        # compute the loss
        #loss = F.mse_loss(y_hat, ratings)
        loss = model.calculate_loss(y_hat, ratings, weights)
        # backpropagate the error through the model
        loss.backward()
        # update model weights
        optimizer.step()

       # print(loss.item())
    
    #test(model)

def train_data_loader(model, train_loader, test_set, epochs, lr, wd, weights=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
 
    model.train()
    prev_accuracy = float('-inf')
    last_improvement = 0
    require_improvement= 10
    
    knn = KNNBasic()
    reader = Reader(rating_scale=(1, 5))
    train_knn = surprise_Dataset.load_from_df(test_set[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
    knn.fit(train_knn)
        
    for _ in range(epochs):
        loss = 0
        for user_ids, item_ids, ratings, in train_loader:
            #print("user_ids", user_ids)
            #print("item_ids", item_ids)
            #print("ratings", ratings)

            train_batch = pd.DataFrame({'user_id': user_ids.numpy(), 'item_id': item_ids.numpy(), 'rating': ratings.numpy()})

            train_batch_items = {}
            for user_id, group in train_batch.groupby(['user_id']):
              train_batch_items[user_id] = np.asarray(group['item_id'])
            

            torch_rmse = evaluate_ratings(model, train_batch_items, test_set)
            print("torch_rmse", torch_rmse)
            
            surprise_train_batch = surprise_Dataset.load_from_df(train_batch[['user_id', 'item_id', 'rating']], reader).build_full_trainset().build_testset()
            knn_rmse = accuracy.rmse(knn.test(surprise_train_batch), verbose=True)

            
            
            print("knn_rmse", knn_rmse)
               
            weights = torch.ones(len(item_ids))
            # clear the gradients
            optimizer.zero_grad()  
            # compute the model output / forward pass
            y_hat = model(user_ids, item_ids)
            # compute the loss
            #loss = F.mse_loss(y_hat, ratings)
            loss = model.calculate_loss(y_hat, ratings, weights)
            # backpropagate the error through the model
            loss.backward()
            # update model weights
            optimizer.step()

        delta = loss - prev_accuracy
        if(delta > 1e-5):
            prev_accuracy = loss
            last_improvement = 0
        else:
            last_improvement += 1
        if last_improvement >= require_improvement:
            break

        ## code here, compute error & early stop learning

       # print(loss.item())
    
    #test(model)

#https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b/Base/Evaluation/metrics.py#L247

def rr(is_relevant):
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0

def dcg(relevance_scores):
    return np.sum(np.divide(np.power(2, relevance_scores) - 1, np.log2(np.arange(relevance_scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)

def ndcg(ranked_relevance, pos_items, at=None):
    
    relevance = np.ones_like(pos_items[:at])   

    rank_dcg = dcg(ranked_relevance[:at])

    if rank_dcg == 0.0:
        return 0.0

    ideal_dcg = dcg(relevance)
    if ideal_dcg == 0.0:
        return 0.0
        
    return rank_dcg / ideal_dcg    


def compute_metrics(model, user_id, train_items, test_items, topK):
    preds = model.full_predict(torch.LongTensor([user_id])).detach().numpy().squeeze()
    preds[train_items] = -sys.maxsize
    
    ranked_list = np.argsort(-preds)[:topK]
    # index_partition = np.argpartition(-preds, topK-1)[:topK]
    # index_sorted = np.argsort(-preds[index_partition])
    # ranked_list = index_partition[index_sorted]

    rank_scores = np.asarray([item in test_items for item in ranked_list])
    
    #with open ('/kaggle/working/fb_preds.txt', 'w') as f:
        #f.write("\n".join(str(item) for item in rank_scores))

    _test_items = np.array(list(test_items))
    _ndcg = ndcg(rank_scores.astype(int), _test_items, at=topK)

    _rr = rr(rank_scores)
    return _ndcg, _rr

def evaluate(model, user_test_relevance, user_train_items, topK, n_items):
    total_ndcg = 0.0
    total_rr = 0.0
    n_user = 0

    for user_id, pos_items in user_test_relevance.items():    
        if user_id not in user_train_items: 
            continue

        n_user += 1

        train_items = user_train_items.get(user_id)
        
        _pos_items_array = np.asarray(list(pos_items))
        
        if (len(_pos_items_array) > 0):
            probs = np.ones(n_items)     
            probs[train_items] = 0.0 #probabilidade dos items de treino sao 0, pra nao usa-los
            probs[_pos_items_array] = 0.0 #probabilidade dos items de teste???
            probs = probs / (n_items - (len(train_items) + len(_pos_items_array)))

            ranking_items = np.random.choice(n_items, size=100, replace=False, p=probs)
            ranking_items = np.concatenate((ranking_items, _pos_items_array), axis=-1) 

            #_ndcg, _rr = compute_metrics(model, user_id, ranking_items, pos_items, topK)
        
        
            _ndcg, _rr = compute_metrics(model, user_id, train_items, ranking_items, topK)

            total_ndcg += _ndcg
            total_rr += _rr

    total_ndcg /= n_user
    total_rr  /= n_user
    return total_ndcg, total_rr


def evaluate_ratings(model, user_train_items, test):
    
    mse = 0.0
    n_users = 0
    for user_id, group in test.groupby(['user_id']):
        if user_id not in user_train_items: 
            continue

        n_users += 1
        n_items = len(group)

        user_ids = torch.LongTensor(np.repeat(user_id, n_items))
        item_ids = torch.LongTensor(group['item_id'].values)

        
        ratings = torch.FloatTensor(group['rating'].values)

        weights = torch.LongTensor(np.repeat(1, n_items))

        y_hat = model(user_ids, item_ids)       
        # compute the loss
        #loss = F.mse_loss(y_hat, ratings)
        mse += model.calculate_loss(y_hat, ratings, weights)
    return mse/n_users


def main(train, validation, test, ndgc_list, mrr_listn):

  pop = train.groupby(['item_id']).agg(popularity=('user_id','nunique')).reset_index()
  
  ## filtering items with more than 10 iteractions
  items_filtered_by_pop = set(pop.query("`popularity` > 10")['item_id'].values)
  train = train[train['item_id'].isin(items_filtered_by_pop)]
  
  pop = train.groupby(['item_id']).agg(popularity=('user_id','nunique')).reset_index()
  
  denominator = train['user_id'].nunique()
  pop['popularity'] = pop['popularity']/denominator

  counts = train.groupby(['ts', 'item_id']).agg(numerator=('user_id','nunique')).reset_index()
  sum_counts = train.groupby(['ts']).agg(denominator=('user_id', 'nunique')).reset_index()
  merged = counts.merge(sum_counts, on='ts')
  merged['propensity'] = merged['numerator']/merged['denominator']

  merged = merged.merge(pop, on='item_id')
  merged = merged.merge(train[['user_id', 'item_id', 'ts', 'rating']], on=['item_id','ts'])

  assert len(train) == len(merged)


  train = merged[['user_id', 'ts', 'item_id', 'popularity', 'propensity', 'rating']]

  assert train.duplicated(subset=['user_id', 'item_id']).sum() == 0

  user_avg_rating = {}
  user_train_items = {}
  for user_id, group in train.groupby(['user_id']):
      user_avg_rating[user_id] = np.mean(group['rating'].values)
      user_train_items[user_id] = np.asarray(group['item_id'])
      if len(user_train_items[user_id]) < 10:
            del user_train_items[user_id]

  idx = (test['item_id'].isin(train['item_id'])) & (test['user_id'].isin(train['user_id']))
  test = test[idx].reset_index(drop=True)
  #(~test['user_id'].isin(train['user_id'])).sum()

  user_valid_relevance = {}
  user_valid_pos_items = {}
  for user_id, group in validation.groupby(['user_id']):
      if user_id in user_avg_rating:
          user_valid_pos_items[user_id] = set(group['item_id'].values)

          idx = group['rating'] >= user_avg_rating[user_id]
      
          user_valid_relevance[user_id] = set(group.loc[idx, 'item_id'])
  
  user_test_relevance = {}
  user_test_pos_items = {}
  for user_id, group in test.groupby(['user_id']):
      if user_id in user_avg_rating:
          user_test_pos_items[user_id] = set(group['item_id'].values)

          idx = group['rating'] >= user_avg_rating[user_id]
      
          user_test_relevance[user_id] = set(group.loc[idx, 'item_id'])

  num_users, num_items = len(u_map), len(i_map)

  n_epochs = 100
  emb_size = [25]
  lr = [1e-6]
  wd = [1e-6]

  train_ds = MFDataset(train)
  train_dl = DataLoader(train_ds, batch_size=int(len(train_ds)/n_epochs), shuffle=True)
  
  #best_mrr = 0
  #best_ndgc = 0
  best_mse = float('inf')
  model1 = None
  for _lr in lr:
      for _wd in wd:
        for _emb_size in emb_size:

          model1 = MF(num_users, num_items, emb_size=_emb_size)
          train_data_loader(model1, train_dl, test, n_epochs, _lr, _wd)


          
          #print(_lr,_wd, ndcg1, mrr1,)

          mse1 = evaluate_ratings(model1, user_train_items, validation)
          if mse1 < best_mse:
            best_mse = mse1
            #best_mrr = mrr1
            #best_ndgc = ndcg1
          #print(f'{mse1:.3f},{ndcg1:.3f} {mrr1:.3f}')


  ndcg1, mrr1 = evaluate(model1, user_test_relevance, user_train_items, 5, num_items)
  mrr_list.append(mrr1)
  ndgc_list.append(ndcg1)


## irá salvar todos os melhores mrr e ndgc de cada treino.
mrr_list = []
ndgc_list = []




import time
start_time = time.perf_counter()
ts_cut = int(len(df) * 0.8)
train = df[:ts_cut].copy().reset_index(drop=True)
test = df[ts_cut:].copy().reset_index(drop=True)
cutoff = int(0.5*len(test))
validation = test[:cutoff].copy().reset_index(drop=True)
test = test[cutoff:].copy().reset_index(drop=True)
    
main(train, validation, test, ndgc_list, mrr_list)

print("elapsed time: %s ms ---" % ((time.perf_counter() - start_time)*1000.0))


with open ('/kaggle/working/fb-ndgc.txt', 'w') as f:
    f.write("\n".join(str(item) for item in ndgc_list))

with open ('/kaggle/working/fb-mrr.txt', 'w') as f:
    f.write("\n".join(str(item) for item in mrr_list))