from src.this_mapping import key_mapping, supported_genres
import pandas as pd
import numpy as np

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
        Поддерживаемые жанры фиксированы и заданы вручную для обеспечения воспроизводимости.
        """
        # Создаём прямой и обратный маппинги
        genre_to_idx = {genre: idx for idx, genre in enumerate(supported_genres)}
        idx_to_genre = {idx: genre for genre, idx in genre_to_idx.items()}

        # Сохраняем маппинги как атрибуты экземпляра (для будущего декодирования)
        self.genre_to_idx = genre_to_idx
        self.idx_to_genre = idx_to_genre

        df_encoded = self.data.copy()

        # Заменяем жанры на индексы
        df_encoded[column_name] = df_encoded[column_name].map(genre_to_idx)

        # Проверка на неизвестные или пропущенные значения
        if df_encoded[column_name].isna().any():
            unknown_values = self.data.loc[df_encoded[column_name].isna(), column_name].unique()
            raise ValueError(f"Обнаружены неизвестные или не поддерживаемые жанры: {unknown_values}")

        self.data = df_encoded
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


    def encode_track_id(self, input_column='track_id', output_column='item_id'):
        """
        Заменяет значения в колонке `input_column` (по умолчанию 'track_id')
        на последовательные целочисленные ID (0, 1, 2, ...) и переименовывает колонку
        в `output_column` (по умолчанию 'item_id').

        Полезно для упрощения, экономии памяти и совместимости с рекомендательными системами.
        """
        df_encoded = self.data.copy()
        
        if input_column not in df_encoded.columns:
            raise ValueError(f"Колонка '{input_column}' отсутствует в данных.")
        
        # Генерируем последовательные ID, независимо от текущего индекса
        df_encoded[output_column] = np.arange(len(df_encoded))
        
        if input_column != output_column:
            df_encoded = df_encoded.drop(columns=[input_column])
        
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
    
    def remove_genres(self, genres=['Anime', 'World', 'Comedy', 'Dance', 'Soundtrack', 'Reggaeton', 
                                    "Children's Music"]):
        # Удаляем строки, где genre_name находится в списке genres
        mask = ~self.data[f'genre'].isin(genres)
        self.data = self.data[mask].copy()
        return self.data
    
    def filter_by_popularity(self, min_popularity=61):  
        mask = self.data['popularity'] > min_popularity
        self.data = self.data[mask].copy()
        return self.data
        
    def do_encoding(self):
        """Выполняет все методы"""
        self.encode_key_column()
        self.encode_tempo()
        self.encode_time_signature()
        self.remove_genres()
        self.filter_by_popularity()
        self.encode_genre_column()
        return self.data

    
    