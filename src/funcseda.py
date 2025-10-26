from src.key_mapping import key_mapping
import pandas as pd

class music_eda:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data.copy()

    def encode_key_column(self, column_name1='key', column_name2='mode'):
        """
        Заменяет буквенные обозначения тональностей в столбце 'key' на числовые (0–11)
        и кодирует лад в столбце 'mode': 'Major' → 1, 'Minor' → 0.
        """
        df_encoded = self.data.copy()
        
        # === Кодирование тональности (key) ===
        df_encoded[column_name1] = df_encoded[column_name1].map(key_mapping)
        if df_encoded[column_name1].isna().any():
            mask = df_encoded[column_name1].isna()
            unknown_keys = self.data.loc[mask, column_name1].dropna().unique()
            raise ValueError(
                f"Обнаружены неизвестные значения тональности: {unknown_keys}. "
                f"Поддерживаемые: {list(key_mapping.keys())}"
            )
        
        # === Кодирование лада (mode) ===
        # Приводим к нижнему регистру для устойчивости
        mode_lower = df_encoded[column_name2].str.lower()
        
        # Создаём mapping
        mode_mapping = {'major': 1, 'minor': 0}
        df_encoded[column_name2] = mode_lower.map(mode_mapping)
        
        # Проверка на неизвестные значения в mode
        if df_encoded[column_name2].isna().any():
            mask = df_encoded[column_name2].isna()
            unknown_modes = self.data.loc[mask, column_name2].dropna().unique()
            raise ValueError(
                f"Обнаружены неизвестные значения лада: {unknown_modes}. "
                f"Поддерживаемые: ['Major', 'Minor']."
            )
        
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
    
    def encode_time_signature(self, column_name='time_signature'):
        """
        Преобразует значения вида '4/4', '3/4'
        Результат — целые числа.
        """
        df_encoded = self.data.copy()
        
        # Убедимся, что значения — строки (на случай, если там уже числа или NaN)
        series = df_encoded[column_name].astype(str)
        
        # Разделяем по '/' и берём первую часть
        numerators = series.str.split('/').str[0]
        
        # Преобразуем в числовой тип
        df_encoded[column_name] = pd.to_numeric(numerators, errors='coerce')
        
        # Проверка на неудавшиеся преобразования (например, если формат не 'X/Y')
        if df_encoded[column_name].isna().any():
            mask = df_encoded[column_name].isna()
            bad_values = self.data.loc[mask, column_name].unique()
            raise ValueError(
                f"Невозможно извлечь числитель из значений time_signature: {bad_values}. "
                f"Ожидаются строки вида '4/4', '3/4' и т.п."
            )
        
        self.data = df_encoded
        return self.data
    
    def encode_track_id(self, column_name='track_id'):
        """
        Заменяет track_id на числовой индекс строки (0, 1, 2, ...).
        Полезно для упрощения и экономии памяти.
        """
        df_encoded = self.data.copy()
        df_encoded[column_name] = df_encoded.index  # или range(len(df_encoded))
        self.data = df_encoded
        return self.data
    
    def encode_tempo(self, column_name='tempo'):
        """
        Округляет значения темпа (tempo) до ближайшего целого числа
        и преобразует столбец к типу int.
        """
        df_encoded = self.data.copy()
        
        # Округляем до ближайшего целого и конвертируем в int
        df_encoded[column_name] = df_encoded[column_name].round().astype('Int64')  # Int64 поддерживает NaN
        
        self.data = df_encoded
        return self.data