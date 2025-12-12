import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib

# --- Добавляем корень проекта в sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Теперь можно импортировать
from src.r_itemkNN import ItemKNN
from src.r_slim_modified import Slim

# --- Настройки путей ---
MODEL_PATH_KNN = os.path.join(PROJECT_ROOT, "itemknn_model")
MODEL_PATH_SLIM = os.path.join(PROJECT_ROOT, "slim_model")
EDA_DATA_PATH = os.path.join(PROJECT_ROOT, "edadata.csv")
DATA_PATH = os.path.join(PROJECT_ROOT, "newdata.csv")

@st.cache_resource
def load_models():
    knn_model = ItemKNN.load(MODEL_PATH_KNN)
    slim_model = Slim.load(MODEL_PATH_SLIM)
    return knn_model, slim_model

knn_model, slim_model = load_models()

# --- Загрузка данных и модели (кэшировано) ---
@st.cache_resource
def load_resources():
    
    # Загружаем метаданные
    edadata = pd.read_csv(EDA_DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    
    # Создаём отображение: (artist, track) → item_id
    track_choices = edadata[['item_id', 'artist_name', 'track_name']].drop_duplicates().copy()
    track_choices['display'] = track_choices['artist_name'] + " – " + track_choices['track_name']
    track_choices = track_choices.set_index('item_id')
    
    return edadata, data, track_choices

# --- Streamlit UI ---
st.set_page_config(page_title="🎯 Рекомендации по трекам", page_icon="🎯")
st.title("🎯 Получить рекомендации на основе любимых треков")

edadata, data, track_choices = load_resources()

# Выбор треков
selected_display = st.multiselect(
    "Выберите ваши любимые треки",
    options=track_choices['display'].tolist(),
    max_selections=10
)

if st.button("Получить рекомендации"):
    if not selected_display:
        st.warning("Пожалуйста, выберите хотя бы один трек.")
    else:
        # Получаем item_id выбранных треков
        selected_items = track_choices[track_choices['display'].isin(selected_display)].index.tolist()
        
        # Проверяем, что все выбранные item_id есть в моделях
        # Используем KNN как основу (но убедитесь, что SLIM обучен на тех же item_id!)
        known_items = set(knn_model.item_enc.categories_[0])
        unknown = [i for i in selected_items if i not in known_items]
        if unknown:
            st.warning(f"Следующие треки не были в обучающих данных: {unknown}. Они будут пропущены.")
            selected_items = [i for i in selected_items if i in known_items]
        
        if not selected_items:
            st.error("Ни один из выбранных треков не найден в модели.")
        else:
            # Преобразуем в индексы (через KNN, но должен совпадать с SLIM!)
            selected_indices = knn_model.item_enc.transform(np.array(selected_items).reshape(-1, 1)).flatten()

            # === Ансамбль: KNN + SLIM ===
            # Убедитесь, что размерности совпадают
            assert knn_model.n_items == slim_model.n_items, "Модели обучены на разном количестве айтемов!"

            # Рекомендации от SLIM: q @ W
            q = np.zeros(slim_model.n_items)
            q[selected_indices] = 1.0
            slim_scores = q @ slim_model.W  # shape: (n_items,)

            # Рекомендации от KNN: сумма столбцов similarity_matrix
            knn_scores = np.zeros(knn_model.n_items)
            for idx in selected_indices:
                knn_scores += knn_model.similarity_matrix[:, idx]

            # Ансамбль (можно настроить веса)
            ensemble_scores = 0.5 * knn_scores + 0.5 * slim_scores

            # Фильтруем уже выбранные треки
            ensemble_scores[selected_indices] = -np.inf

            # Топ-10
            top_k = 10
            top_indices = np.argpartition(ensemble_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-ensemble_scores[top_indices])]

            # Декодируем в оригинальные item_id (через KNN — они должны совпадать)
            top_item_ids = knn_model.item_enc.inverse_transform(top_indices.reshape(-1, 1)).flatten()

            # Получаем метаданные
            recommendations = data[data['item_id'].isin(top_item_ids)].copy()
            recommendations['sort_key'] = recommendations['item_id'].map({id_: i for i, id_ in enumerate(top_item_ids)})
            recommendations = recommendations.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

            st.subheader("Ваши рекомендации (ансамбль ItemKNN + SLIM):")
            st.dataframe(recommendations, use_container_width=True)

st.sidebar.caption("© 2025 Music Recommender")