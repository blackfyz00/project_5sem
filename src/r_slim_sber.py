from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd

class Slim:
    def __init__(self, alpha=0.1, l1_ratio=0.5, tol=0.01):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol

    def fit(self, train):
        self.train = train
        self.X = pd.pivot_table(train, values='rating', index='user_id', columns='item_id').fillna(0).to_numpy()
        num_items = self.X.shape[1]
        self.W = np.zeros((self.X.shape[1], self.X.shape[1]))
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, tol=self.tol)
        for j in tqdm(range(num_items), desc='progress'):
            y = self.X[:, j].copy()
            self.X[:, j] = 0
            model.fit(self.X, y)

            model_coef = model.coef_
            model_coef[model_coef < 0] = 0
            self.W[:, j] = model_coef

            self.X[:, j] = y

    def predict(self, user_id, topk=10, filter_seen=True):
        rank_matrix = self.X @ self.W

        if filter_seen:
            rank_matrix = np.multiply(rank_matrix, np.invert(self.X.astype(bool)))

        labels_topk = np.argsort(rank_matrix, axis=1)[:, -topk:][:,::-1]
        test_history = self.train[self.train.user_id.isin(user_id)].sort_values('user_id')
        last_item = test_history.groupby('user_id')['item_id'].apply(lambda x: x.iloc[-1]).reset_index(drop=True)
        pred = pd.DataFrame({'user_id': np.repeat(np.sort(user_id), 10), 'item_id': list(labels_topk[last_item].reshape(-1,1))})
        pred = pred.explode('item_id')
        pred['rating'] = [x % topk for x in range(len(pred))]

        return pred