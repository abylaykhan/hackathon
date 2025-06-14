import pandas as pd
import numpy as np

class TableDataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data['event_time'] = pd.to_datetime(self.data['event_time'], utc=True)
        self.data['application_date'] = pd.to_datetime(self.data['application_date'], utc=True)
        self.data['client_event_time'] = pd.to_datetime(self.data['client_event_time'], utc=True)
        drop_cols = ['device_brand', 'device_manufacturer', 'device_model', 'global_user_properties', 'partner_id', 'user_creation_time']
        self.data = self.data.drop(drop_cols, axis=1)

    def create_features(self):
        arr_features = []
        for app_id in self.data['applicationid'].unique():
            data = self.data[self.data['applicationid'] == app_id].copy()
            features = self.create_application_features(data)
            arr_features.append(features)
        new_data = pd.DataFrame(arr_features)
        return new_data
    
    def create_application_features(self, data: pd.DataFrame) -> pd.Series:
        assert data['applicationid'].nunique() == 1

        df = data.copy()
        df = df.sort_values('event_time')

        app_date = df['application_date'].iloc[0]

        # Временные разности
        time_diffs = df['event_time'].diff().dt.total_seconds().dropna()
        time_span = (df['event_time'].max() - df['event_time'].min()).total_seconds()

        # Активность вокруг даты заявки
        events_before_app = (df['event_time'] < app_date).sum()
        events_after_app = (df['event_time'] >= app_date).sum()
        
        # Интенсивность событий
        avg_events_per_minute = len(df) / (time_span / 60) if time_span > 0 else 0
        std_time_diff = time_diffs.std()
        median_time_diff = time_diffs.median()
        p95_time_diff = np.percentile(time_diffs, 95) if len(time_diffs) > 0 else 0

        # Первое и последнее событие — от заявки
        time_to_first_event = (df['event_time'].min() - app_date).total_seconds()
        time_to_last_event = (df['event_time'].max() - app_date).total_seconds()

        # Типы событий
        event_type_counts = df['event_type'].value_counts().to_dict()
        unique_event_types = df['event_type'].nunique()
        top_event_type = df['event_type'].mode().iloc[0] if not df['event_type'].mode().empty else None

        # Устройства
        unique_devices = df['device_id'].nunique()
        device_switched = int(unique_devices > 1)
        switched_device_type = int(df['device_type'].nunique() > 1)
        most_common_device_type = df['device_type'].mode().iloc[0] if not df['device_type'].mode().empty else None

        # Геолокация
        if df['location_lat'].notna().sum() > 1 and df['location_lng'].notna().sum() > 1:
            df['lat_diff'] = df['location_lat'].diff().abs()
            df['lng_diff'] = df['location_lng'].diff().abs()
            df['geo_jump'] = ((df['lat_diff'] > 0.1) | (df['lng_diff'] > 0.1)).astype(int)
            geo_jumps = df['geo_jump'].sum()
            avg_lat_change = df['lat_diff'].mean()
            avg_lng_change = df['lng_diff'].mean()
        else:
            geo_jumps = 0
            avg_lat_change = 0
            avg_lng_change = 0

        # IP / регион
        unique_ips = df['ip_address'].nunique()
        unique_languages = df['language'].nunique()
        unique_platforms = df['platform'].nunique()
        unique_os = df['os_name'].nunique()
        unique_versions = df['version_name'].nunique()
        unique_countries = df['country'].nunique()
        unique_regions = df['region'].nunique()
        unique_cities = df['city'].nunique()

        # Сессии
        unique_sessions = df['session_id'].nunique()
        session_counts = df.groupby('session_id')['event_id'].count()
        avg_session_len = session_counts.mean()
        max_session_len = session_counts.max()
        session_length_std = session_counts.std()

        # Сессионные метрики
        first_session_len = session_counts.iloc[0] if len(session_counts) > 0 else 0
        last_session_len = session_counts.iloc[-1] if len(session_counts) > 0 else 0
        session_len_diff = last_session_len - first_session_len

        # Временные зоны и часы активности
        df['hour'] = df['event_time'].dt.hour
        events_night = df[df['hour'].between(0, 6)].shape[0]
        events_day = df[df['hour'].between(7, 18)].shape[0]
        events_evening = df[df['hour'].between(19, 23)].shape[0]
        most_active_hour = df['hour'].mode().iloc[0] if not df['hour'].mode().empty else -1

        # Последовательность и последние действия
        last_event_type = df['event_type'].iloc[-1]
        second_last_event_type = df['event_type'].iloc[-2] if len(df) > 1 else None
        event_type_entropy = df['event_type'].value_counts(normalize=True).apply(lambda p: -p * np.log2(p)).sum()

        # Выходные
        df['weekday'] = df['event_time'].dt.weekday
        weekend_events = df[df['weekday'] >= 5].shape[0]
        weekday_events = df[df['weekday'] < 5].shape[0]

        features = {
            # Временные характеристики
            'time_span_sec': time_span,
            'events_before_app': events_before_app,
            'events_after_app': events_after_app,
            'avg_time_diff_sec': time_diffs.mean(),
            'std_time_diff_sec': std_time_diff,
            'median_time_diff_sec': median_time_diff,
            'max_time_diff_sec': time_diffs.max(),
            'min_time_diff_sec': time_diffs.min(),
            'p95_time_diff_sec': p95_time_diff,
            'avg_events_per_minute': avg_events_per_minute,
            'time_to_first_event': time_to_first_event,
            'time_to_last_event': time_to_last_event,

            # События
            'unique_event_types': unique_event_types,
            'top_event_type': top_event_type,
            'event_type_entropy': event_type_entropy,
            'last_event_type': last_event_type,
            'second_last_event_type': second_last_event_type,
            **{f'event_type_count_{k}': v for k, v in event_type_counts.items()},

            # Устройства
            'unique_devices': unique_devices,
            'device_switched': device_switched,
            'switched_device_type': switched_device_type,
            'most_common_device_type': most_common_device_type,

            # Гео
            'geo_jumps': geo_jumps,
            'avg_lat_change': avg_lat_change,
            'avg_lng_change': avg_lng_change,

            # IP / язык / страна
            'unique_ips': unique_ips,
            'unique_languages': unique_languages,
            'unique_platforms': unique_platforms,
            'unique_os': unique_os,
            'unique_versions': unique_versions,
            'unique_countries': unique_countries,
            'unique_regions': unique_regions,
            'unique_cities': unique_cities,

            # Сессии
            'unique_sessions': unique_sessions,
            'avg_session_len': avg_session_len,
            'max_session_len': max_session_len,
            'session_len_std': session_length_std,
            'first_session_len': first_session_len,
            'last_session_len': last_session_len,
            'session_len_diff': session_len_diff,

            # Время суток
            'events_night': events_night,
            'events_day': events_day,
            'events_evening': events_evening,
            'most_active_hour': most_active_hour,

            # Дни недели
            'weekday_events': weekday_events,
            'weekend_events': weekend_events,

            'applicationid': df['applicationid'].iloc[0]
        }

        return pd.Series(features)
