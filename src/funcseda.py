from key_mapping import key_mapping
import pandas as pd

class music_eda:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data.copy()

    def encode_key_column(self, column_name='key'):
        """Заменяет буквенные обозначения тональностей в столбце на числовые (0–11)."""
        df_encoded = self.data.copy()
        df_encoded[column_name] = df_encoded[column_name].map(key_mapping)
        
        # Опционально: проверка на неизвестные значения
        if df_encoded[column_name].isna().any():
            unknown_keys = df[df_encoded[column_name].isna()][column_name].unique()
            raise ValueError(f"Обнаружены неизвестные значения тональности: {unknown_keys}. "
                            f"Поддерживаемые: {list(key_mapping.keys())}")
        self.data = df_encoded
        return self.data
    
    def encode_genre_column(self, column_name='genre'):
        """
        Заменяет названия музыкальных жанров в столбце на числовые метки (0, 1, 2, ...).
        Поддерживаемые жанры:
            ['Movie', 'R&B', 'A Capella', 'Alternative', 'Country', 'Dance',
            'Electronic', 'Anime', 'Folk', 'Blues', 'Opera', 'Hip-Hop',
            "Children's Music", 'Rap', 'Indie',
            'Classical', 'Pop', 'Reggae', 'Reggaeton', 'Jazz', 'Rock', 'Ska',
            'Comedy', 'Soul', 'Soundtrack', 'World']
        """
        # Нормализуем разные варианты написания "Children's Music"
        df_normalized = self.data.copy()
        
        # Уникальные жанры (после нормализации)
        unique_genres = sorted(df_normalized[column_name].dropna().unique())
        
        # Создаём mapping жанр → индекс
        genre_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
        
        # Применяем mapping
        df_normalized[column_name] = df_normalized[column_name].map(genre_mapping)
        
        # Проверка на неизвестные/пропущенные значения
        if df_normalized[column_name].isna().any():
            unknown = df_normalized.loc[df_normalized[column_name].isna(), column_name].unique()
            raise ValueError(f"Обнаружены неизвестные или пропущенные жанры: {unknown}")
        
        self.data = df_normalized
        return self.data
    