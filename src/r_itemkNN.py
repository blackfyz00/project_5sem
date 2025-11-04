import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class ItemKNN:
    def __init__(self, k_neighbours=50, normalize=True, filter_seen=True):
        self.k_neighbours = k_neighbours
        self.normalize = normalize
        self.filter_seen = filter_seen
        self.user_enc = None
        self.item_enc = None
        self.user_mapping = None
        self.item_mapping = None
        self.similarity_matrix = None
        self.train_matrix = None
        self.n_users = None
        self.n_items = None

    def fit(self, train_log: pd.DataFrame):
        self.user_enc = OrdinalEncoder(dtype=int)
        self.item_enc = OrdinalEncoder(dtype=int)

        # В fit():
        train_log = train_log[train_log['rating'] >= 3].copy()
        train_log['rating'] = 1.0  # implicit

        user_ids = self.user_enc.fit_transform(train_log[['user_id']]).flatten()
        item_ids = self.item_enc.fit_transform(train_log[['item_id']]).flatten()

        self.n_users = user_ids.max() + 1
        self.n_items = item_ids.max() + 1

        self.raw_train_matrix = np.zeros((self.n_users, self.n_items))
        self.raw_train_matrix[user_ids, item_ids] = train_log['rating'].values

        # Используем копию для схожести
        similarity_matrix = self.raw_train_matrix.copy()

        if self.normalize:
            norms = np.linalg.norm(similarity_matrix, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            similarity_matrix = similarity_matrix / norms

        self.similarity_matrix = similarity_matrix.T @ similarity_matrix
        np.fill_diagonal(self.similarity_matrix, 0)  # <<< КРИТИЧНО

        return self

    def _predict_full(self):
        k = min(self.k_neighbours, self.n_items - 1)
        if k == 0:
            return np.zeros((self.n_users, self.n_items))

        # Топ-k соседей, исключая самого себя
        neighbour_ids = np.argsort(-self.similarity_matrix, axis=1)[:, 1:k+1]

        predicted = np.zeros((self.n_users, self.n_items))

        for item in range(self.n_items):
            neighbours = neighbour_ids[item]
            sims = self.similarity_matrix[item, neighbours]

            # Только положительная схожесть
            mask = sims > 0
            if not np.any(mask):
                continue
            sims = sims[mask]
            neighbours = neighbours[mask]

            ratings = self.raw_train_matrix[:, neighbours]
            weighted_sum = ratings @ sims
            predicted[:, item] = weighted_sum / sims.sum()

        if self.filter_seen:
            seen_mask = self.raw_train_matrix > 0
            predicted[seen_mask] = -np.inf

        return predicted

    def predict(self, users=None, k=10):
        full_pred = self._predict_full()

        if users is None:
            # Все пользователи из train
            user_indices = np.arange(self.n_users)
            user_ids_original = self.user_enc.inverse_transform(user_indices.reshape(-1, 1)).flatten()
        else:
            users = np.array(users).reshape(-1, 1)
            user_indices = self.user_enc.transform(users).astype(int).flatten()
            user_ids_original = users.flatten()

        pred_subset = full_pred[user_indices]

        # Топ-k
        topk_indices = np.argpartition(pred_subset, -k, axis=1)[:, -k:]
        topk_scores = np.take_along_axis(pred_subset, topk_indices, axis=1)
        sort_idx = np.argsort(-topk_scores, axis=1)
        topk_indices = np.take_along_axis(topk_indices, sort_idx, axis=1)
        topk_scores = np.take_along_axis(topk_scores, sort_idx, axis=1)

        # Обратно в оригинальные item_id
        item_ids_original = self.item_enc.inverse_transform(topk_indices.reshape(-1, 1)).flatten()

        return pd.DataFrame({
            'user_id': np.repeat(user_ids_original, k),
            'item_id': item_ids_original.flatten(),
            'rating': topk_scores.flatten()
        })