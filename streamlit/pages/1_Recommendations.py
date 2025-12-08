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

# --- Настройки путей ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "itemknn_model")
EDA_DATA_PATH = os.path.join(PROJECT_ROOT, "edadata.csv")
DATA_PATH = os.path.join(PROJECT_ROOT, "newdata.csv")

# --- Загрузка данных и модели (кэшировано) ---
@st.cache_resource
def load_resources():
    # Загружаем модель
    model = ItemKNN.load(MODEL_PATH)
    
    # Загружаем метаданные
    edadata = pd.read_csv(EDA_DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    
    # Создаём отображение: (artist, track) → item_id
    track_choices = edadata[['item_id', 'artist_name', 'track_name']].drop_duplicates().copy()
    track_choices['display'] = track_choices['artist_name'] + " – " + track_choices['track_name']
    track_choices = track_choices.set_index('item_id')
    
    return model, edadata, data, track_choices

# --- Streamlit UI ---
st.set_page_config(page_title="🎯 Рекомендации по трекам", page_icon="🎯")
st.title("🎯 Получить рекомендации на основе любимых треков")

model, edadata, data, track_choices = load_resources()

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
        
        # Проверяем, что все выбранные item_id есть в модели
        known_items = set(model.item_enc.categories_[0])
        unknown = [i for i in selected_items if i not in known_items]
        if unknown:
            st.warning(f"Следующие треки не были в обучающих данных: {unknown}. Они будут пропущены.")
            selected_items = [i for i in selected_items if i in known_items]
        
        if not selected_items:
            st.error("Ни один из выбранных треков не найден в модели.")
        else:
            # Преобразуем в индексы
            selected_indices = model.item_enc.transform(np.array(selected_items).reshape(-1, 1)).flatten()
            
            # Вычисляем рекомендации: score[j] = sum_i sim[j, i] for i in selected
            sim_matrix = model.similarity_matrix  # (n_items, n_items)
            scores = np.zeros(model.n_items)
            for idx in selected_indices:
                scores += sim_matrix[:, idx]
            
            # Исключаем уже выбранные треки (если нужно)
            if model.filter_seen:
                mask_seen = np.zeros(model.n_items, dtype=bool)
                mask_seen[selected_indices] = True
                scores[mask_seen] = -np.inf
            
            # Топ-10
            top_k = 10
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            
            # Обратно в оригинальные item_id
            top_item_ids = model.item_enc.inverse_transform(top_indices.reshape(-1, 1)).flatten()
            
            # Достаём полные строки из edadata
            recommendations = data[data['item_id'].isin(top_item_ids)].copy()
            
            # Сохраняем порядок
            recommendations['sort_key'] = recommendations['item_id'].map({id_: i for i, id_ in enumerate(top_item_ids)})
            recommendations = recommendations.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
            
            st.subheader("Ваши рекомендации:")
            st.dataframe(recommendations, use_container_width=True)