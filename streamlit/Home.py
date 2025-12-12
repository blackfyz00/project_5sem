import streamlit as st
from strings import string
import base64
import os

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

# # Подключаем Montserrat и применяем его ко всему приложению
# st.markdown("""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

#         /* Применяем шрифт ко всем элементам */
#         html, body, [class*="css"] {
#             font-family: 'Montserrat', sans-serif !important;
#         }

#         /* Опционально: увеличить размер шрифта */
#         p, div, span, li {
#             font-size: 18px !important;
#         }
#     </style>
# """, unsafe_allow_html=True)


st.title("🎧 Рекомендательная система музыки: DYVMEK")
st.markdown("""
Добро пожаловать в персонализированную систему рекомендаций музыки...
""")

# file_path = "pdf.pdf"
# def show_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf">'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Пример использования:
# show_pdf("pdf.pdf")  # предполагается, что файл лежит в той же папке, что и скрипт

st.markdown("---")
# st.markdown(string)

for filename in sorted(os.listdir("screenshots")):
    if filename.endswith(".png"):
        st.image(f"screenshots/{filename}")
        st.markdown("---")

text = string
st.markdown(
    f"""
    <div style="text-align: justify;">
    {text}
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.info("💡 Чтобы получить рекомендации, перейдите на страницу **«Рекомендации»** в боковом меню.")
st.sidebar.caption("© 2025 Music Recommender")