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
    edadata = pd.read_csv(EDA_DATA_PATH)
    data = pd.read_csv(DATA_PATH)

    # Белый список item_id — только те, что остались после вашей фильтрации
    allowed_item_ids = set(edadata['item_id'].unique())

    # Для UI: создаём уникальный список треков из белого списка
    # (по-прежнему убираем дубли по artist+track)
    ui_tracks = edadata[['item_id', 'artist_name', 'track_name']].drop_duplicates(
        subset=['artist_name', 'track_name'], keep='first'
    ).copy()
    ui_tracks['display'] = ui_tracks['artist_name'] + " – " + ui_tracks['track_name']
    ui_tracks = ui_tracks.set_index('item_id')

    return allowed_item_ids, ui_tracks, data

# --- Streamlit UI ---
st.set_page_config(page_title="🎯 Рекомендации по трекам", page_icon="🎯")
st.title("🎯 Получить рекомендации на основе любимых треков")

allowed_item_ids, track_choices, data = load_resources()

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

            # После получения top_item_ids (из модели)
            # Фильтруем: оставляем ТОЛЬКО те, что есть в белом списке (edadata)
            allowed_top_item_ids = [item_id for item_id in top_item_ids if item_id in allowed_item_ids]

            if not allowed_top_item_ids:
                st.warning("Ни одна рекомендация не прошла фильтрацию.")
            else:
                # Теперь берём метаданные ИЗ DATA, но только для разрешённых item_id
                recommendations = data[data['item_id'].isin(allowed_top_item_ids)].copy()

                # Восстанавливаем порядок (top-k порядок из модели)
                item_id_to_rank = {item_id: i for i, item_id in enumerate(top_item_ids)}
                recommendations['rank'] = recommendations['item_id'].map(item_id_to_rank)
                recommendations = recommendations.sort_values('rank').drop('rank', axis=1).reset_index(drop=True)

                st.subheader("Ваши рекомендации:")
                st.dataframe(recommendations, use_container_width=True)

st.sidebar.caption("© 2025 Music Recommender")