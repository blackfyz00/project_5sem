import streamlit as st

st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded"  # или "auto"
)

st.title("🎧 Music Recommender System")
st.markdown("""
Добро пожаловать в персонализированную систему рекомендаций музыки...
""")

st.markdown("---")
st.subheader("🔍 Как это работает?")
st.markdown("1. Анализ поведения...")

st.subheader("✨ Возможности")
st.markdown("- Получение персонализированных рекомендаций...")

st.subheader("📊 Качество модели")
st.markdown("- **HitRate@10**: 48%...")

st.markdown("---")
st.info("💡 Чтобы получить рекомендации, перейдите на страницу **«Рекомендации»** в боковом меню.")
st.sidebar.caption("© 2025 Music Recommender")