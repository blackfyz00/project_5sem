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

    # Белый список item_id — только те, что остались после фильтрации в edadata
    allowed_item_ids = set(edadata['item_id'].unique())

    track_catalog = data.drop_duplicates(subset=['item_id'], keep='first').set_index('item_id')

    track_catalog['display'] = track_catalog['artist_name'] + " – " + track_catalog['track_name']

    # UI-список — только display и item_id (для выбора)
    ui_tracks = track_catalog[['display']].copy()

    return allowed_item_ids, ui_tracks, track_catalog, edadata

# --- Streamlit UI ---
st.set_page_config(page_title="🎯 Рекомендации по трекам", page_icon="🎯")
st.title("🎯 Получить рекомендации на основе любимых треков")

allowed_item_ids, track_choices, track_catalog, eda_data = load_resources()

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
            selected_items = [i for i in selected_items if i in known_items]
        
        if not selected_items:
            # Выводим 10 самых популярных треков из eda_data в диапазоне строк 140–150
            subset = eda_data.iloc[140:151]  # 140 включительно, 151 — не включительно → 140–150
            top_popular = subset.nlargest(10, 'popularity')
            
            # Получаем item_id этих треков
            fallback_item_ids = top_popular['item_id'].tolist()
            
            # Фильтруем по наличию в track_catalog (на всякий случай)
            fallback_item_ids = [item_id for item_id in fallback_item_ids if item_id in track_catalog.index]
            
            if fallback_item_ids:
                recommendations = track_catalog.loc[fallback_item_ids].reset_index()
                st.subheader("Рекомендуемые треки:")
                st.dataframe(recommendations, use_container_width=True)
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

            all_sorted_indices = np.argsort(-ensemble_scores)
  
            all_item_ids = knn_model.item_enc.inverse_transform(all_sorted_indices.reshape(-1, 1)).flatten()

            allowed_recommendations = [item_id for item_id in all_item_ids if item_id in allowed_item_ids]

            # Берём ТОП-10 из разрешённых
            top_10_allowed = allowed_recommendations[:10]

            # Теперь берём метаданные ИЗ DATA, но только для этих 10
            recommendations = track_catalog.loc[top_10_allowed].reset_index()

            st.subheader("Ваши рекомендации:")
            st.dataframe(recommendations, use_container_width=True)

st.sidebar.caption("© 2025 Music Recommender")