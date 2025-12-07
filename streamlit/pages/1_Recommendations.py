import streamlit as st
import sys
import os

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

import joblib
from src.r_itemkNN import ItemKNN

st.set_page_config(page_title="🎯 Рекомендации", page_icon="🎯")

st.title("🎯 Получить рекомендации")

user_input = st.text_input("Введите ID пользователя", placeholder="Например: user_123")

if st.button("Получить топ-10"):
    if user_input.strip():
        try:
            # Здесь загружаем модель и делаем предсказание

            @st.cache_resource
            def load_model():
                return joblib.load("itemknn_model.joblib")

            model = load_model()
            recs = model.predict(users=[user_input], k=10)
            st.subheader(f"Рекомендации для: `{user_input}`")
            st.dataframe(
                recs[['item_id', 'rating']].rename(
                    columns={'item_id': 'Трек (ID)', 'rating': 'Оценка'}
                ).reset_index(drop=True),
                use_container_width=True
            )
        except FileNotFoundError:
            st.error("⚠️ Модель не найдена. Обучите её и сохраните как `itemknn_model.joblib`.")
        except ValueError as e:
            if "seen during" in str(e):
                st.warning("❌ Пользователь не найден в обучающих данных.")
            else:
                st.error(f"Ошибка: {e}")
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
    else:
        st.warning("Пожалуйста, введите ID пользователя.")