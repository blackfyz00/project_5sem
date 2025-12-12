from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OrdinalEncoder
from replay.metrics import HitRate, NDCG, Coverage, OfflineMetrics
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import os

class Slim:
    def __init__(self, alpha=0.1, l1_ratio=0.5, tol=0.01):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.user_enc = OrdinalEncoder(dtype=int)

    def fit(self, train, item_enc=None):
        self.train = train = train.copy()

        # user_enc — всегда внутренний (если не делаете ансамбль по user_id)
        self.user_enc = OrdinalEncoder(dtype=int)
        train['user_idx'] = self.user_enc.fit_transform(train[['user_id']]).flatten()

        # item_enc — может быть внешним
        if item_enc is not None:
            self.item_enc = item_enc
            train['item_idx'] = self.item_enc.transform(train[['item_id']]).flatten()
        else:
            raise Exception("Нужно передать item_enc")

        self.idx_to_item_id = {
            idx: item_id for item_id, idx in zip(train['item_id'], train['item_idx'])
        }

        n_users = train['user_idx'].max() + 1
        n_items = train['item_idx'].max() + 1
        self.n_users = n_users
        self.n_items = n_items

        self.X = np.zeros((n_users, n_items), dtype=np.float32)
        self.X[train['user_idx'], train['item_idx']] = 1.0

        num_items = n_items
        self.W = np.zeros((num_items, num_items), dtype=np.float32)

        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, tol=self.tol, max_iter=1000)

        for j in tqdm(range(num_items), desc='Fitting SLIM'):
            y = self.X[:, j].copy()
            self.X[:, j] = 0.0  # маскируем целевой айтем

            # Обучаем модель: предсказать y через остальные айтемы
            model.fit(self.X, y)

            # Обнуляем отрицательные веса
            coef = model.coef_.copy()
            coef[coef < 0] = 0.0
            self.W[:, j] = coef

            # Возвращаем y обратно
            self.X[:, j] = y

    def predict(self, users=None, k=10, filter_seen=True):
        rank_matrix = self.X @ self.W  # shape (n_users, n_items)

        if filter_seen:
            rank_matrix[self.X > 0] = -np.inf

        if users is None:
            user_indices = np.arange(self.X.shape[0])
            user_ids = self.user_enc.inverse_transform(user_indices.reshape(-1, 1)).flatten()
        else:
            users = np.array(users)
            user_indices = self.user_enc.transform(users.reshape(-1, 1)).flatten().astype(int)
            user_ids = users

        pred_subset = rank_matrix[user_indices]  # (len(users), n_items)

        topk_indices = np.argpartition(pred_subset, -k, axis=1)[:, -k:]
        topk_scores = np.take_along_axis(pred_subset, topk_indices, axis=1)
        sort_idx = np.argsort(-topk_scores, axis=1)
        topk_indices = np.take_along_axis(topk_indices, sort_idx, axis=1)
        topk_scores = np.take_along_axis(topk_scores, sort_idx, axis=1)

        # Декодируем item индексы в item_id
        item_ids_flat = np.array([self.idx_to_item_id[i] for i in topk_indices.flatten()])

        return pd.DataFrame({
            'user_id': np.repeat(user_ids, k),
            'item_id': item_ids_flat,
            'rating': topk_scores.flatten()
        })
    
    def save(self, path: str):
            """Сохраняет SLIM модель."""
            os.makedirs(path, exist_ok=True)

            # Сохраняем матрицу весов
            np.savez_compressed(
                os.path.join(path, "slim_weights.npz"),
                W=self.W
            )

            # Сохраняем энкодеры и маппинги
            joblib.dump(self.item_enc, os.path.join(path, "item_enc.joblib"))
            joblib.dump(self.idx_to_item_id, os.path.join(path, "idx_to_item_id.joblib"))

            # Сохраняем параметры
            params = {
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "tol": self.tol,
                "n_items": self.n_items,
            }
            joblib.dump(params, os.path.join(path, "params.joblib"))

    @classmethod
    def load(cls, path: str):
        """Загружает SLIM модель."""
        params = joblib.load(os.path.join(path, "params.joblib"))
        model = cls(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            tol=params["tol"]
        )

        # Загружаем веса
        weights = np.load(os.path.join(path, "slim_weights.npz"))
        model.W = weights["W"]
        model.n_items = params["n_items"]

        # Загружаем item-энкодер и маппинг
        model.item_enc = joblib.load(os.path.join(path, "item_enc.joblib"))
        model.idx_to_item_id = joblib.load(os.path.join(path, "idx_to_item_id.joblib"))

        # Восстанавливаем обратный маппинг item_id -> idx (если понадобится)
        model.item_id_to_idx = {v: k for k, v in model.idx_to_item_id.items()}

        return model