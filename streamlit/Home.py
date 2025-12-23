import streamlit as st
from strings import string
import base64
import os
import pathlib
import re

script_dir = pathlib.Path(__file__).parent.resolve()
screenshots_dir = script_dir / "screenshots" 

st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded"  # или "auto"
)

# Увеличиваем базовый размер шрифта
st.markdown("""
    <style>
        /* Основной текст */
        html, body, [class*="css"] {
            font-size: 18px !important;
        }
        
        /* Заголовки */
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.75rem !important; }
        h4 { font-size: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎧 Рекомендательная система музыки: DYVMEK")
st.markdown("""
Добро пожаловать в персонализированную систему рекомендаций музыки.
""")

def sort_key_numeric(filename):
    # Извлекаем числа из имени файла с помощью регулярного выражения
    # Пример: для "10.png" вернет ['10', 'png']
    parts = re.findall(r'(\d+)|(\D+)', filename)
    # Преобразуем части в int, если это число, иначе оставляем строку
    return [int(p[0]) if p[0] else p[1] for p in parts]

# Используем вашу функцию в качестве ключа для сортировки
sorted_files = sorted(os.listdir(screenshots_dir), key=sort_key_numeric)
col1, col2, col3 = st.columns([1, 40, 1])
for filename in sorted_files:
    if filename.endswith(".png"):
        with col2:
            full_path = screenshots_dir / filename
            st.image(full_path)
            st.markdown("---")

st.info("💡 Чтобы получить рекомендации, перейдите на страницу **«Рекомендации»** в боковом меню.")
st.sidebar.caption("© 2025 Music Recommender")