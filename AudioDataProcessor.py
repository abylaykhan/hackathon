from pyannote.audio import Pipeline
import torch
from tqdm.notebook import tqdm

import os
import pandas as pd
import numpy as np

from collections import defaultdict
import librosa


from pydub import AudioSegment


class AudioDataProcessor:
    def __init__(self, path_to_audiofiles):
        self.path_to_audiofiles = path_to_audiofiles
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token='')
        pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline
    

    def trim_audio(self, file_path, output_path, start_sec=20, end_sec=80):
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000  # длительность аудио в секундах

        start_ms = start_sec * 1000
        end_ms = end_sec * 1000

        # Если файл короче конца обрезки — обрежем до конца доступного
        if end_ms > len(audio):
            end_ms = len(audio)

        if start_ms >= end_ms:
            start_ms = 0

        trimmed = audio[start_ms:end_ms]
        trimmed.export(output_path, format="wav")  # можно заменить формат, если нужно

    def create_trimmed_audio(self):
        input_dir = self.path_to_audiofiles
        output_dir = "trimmed_audios_new"
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith(".wav"):  # или .wav и т.д.
                in_path = os.path.join(input_dir, filename)
                out_path = os.path.join(output_dir, filename)
                self.trim_audio(in_path, out_path)
    
    def create_diarization_features(self):
        audio_dir = "trimmed_audios_new"
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

        results = []

        for filename in tqdm(audio_files):
            try:
                filepath = os.path.join(audio_dir, filename)
                diarization = self.pipeline(filepath)

                segments = []
                speaker_durations = defaultdict(float)
                speaker_turns = defaultdict(int)
                audio_end = 0.0

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start, end = turn.start, turn.end
                    duration = end - start
                    segments.append((start, end, duration, speaker))

                    speaker_durations[speaker] += duration
                    speaker_turns[speaker] += 1
                    audio_end = max(audio_end, end)

                # Если сегментов нет — пропустить
                if not segments:
                    continue

                # Сортировка по времени начала для анализа пауз
                segments_sorted = sorted(segments, key=lambda x: x[0])

                # Расчёт пауз
                pauses = [
                    segments_sorted[i + 1][0] - segments_sorted[i][1]
                    for i in range(len(segments_sorted) - 1)
                    if segments_sorted[i + 1][0] > segments_sorted[i][1]
                ]

                # Скалярные признаки
                total_segments = len(segments)
                durations = [s[2] for s in segments]
                total_speech = sum(durations)

                row = {
                    "filename": filename,
                    "duration": round(audio_end, 2),
                    "speakers_count": len(speaker_durations),
                    "total_segments": total_segments,
                    "avg_segment_duration": round(np.mean(durations), 2),
                    "std_segment_duration": round(np.std(durations), 2),
                    "max_speaker_duration": round(max(speaker_durations.values()), 2),
                    "min_speaker_duration": round(min(speaker_durations.values()), 2),
                    "dominant_speaker_ratio": round(max(speaker_durations.values()) / total_speech, 2),
                    "speaking_turns_diff": max(speaker_turns.values()) - min(speaker_turns.values()),
                    "speech_density": round(total_speech / audio_end, 2),
                    "avg_pause_between_segments": round(np.mean(pauses), 2) if pauses else 0.0,
                }

            except Exception as e:
                row = {
                    "filename": filename,
                    "duration": 0,
                    "speakers_count": 0,
                    "total_segments": 0,
                    "avg_segment_duration": 0,
                    "std_segment_duration": 0,
                    "max_speaker_duration": 0,
                    "min_speaker_duration": 0,
                    "dominant_speaker_ratio": 0,
                    "speaking_turns_diff": 0,
                    "speech_density": 0,
                    "avg_pause_between_segments": 0}
                print(e)
            results.append(row)

        # В DataFrame
        df_scalar = pd.DataFrame(results)
        return df_scalar
    
    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)

        features = []

        # === MFCC ===
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        # === Chroma (может не работать с низким sr) ===
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
        except Exception as e:
            print(f"[CHROMA] Пропущено ({file_path}): {e}")
            features.extend([0.0] * 12)

        # === Spectral contrast с безопасным n_bands ===
        try:
            fmin = 200.0
            max_n_bands = int(np.floor(np.log2((sr / 2) / fmin)))
            n_bands = min(6, max_n_bands)

            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=n_bands)
            contrast_mean = np.mean(contrast, axis=1)
            features.extend(contrast_mean)
        except Exception as e:
            print(f"[CONTRAST] Пропущено ({file_path}): {e}")
            features.extend([0.0] * 7)

        # === Zero Crossing Rate ===
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        return np.barray(features)

    def create_audio_features(self):
        folder_path = self.path_to_audiofiles
        data = []
        second_data = []
        for filename in tqdm(os.listdir(folder_path)):
            try:
                if filename.endswith(".wav"):
                    filepath = os.path.join(folder_path, filename)
                    features = self.extract_features(filepath)
                    data.append([filename] + list(features))
            except Exception as e:
                second_data.append(filename)
                print(e)

        # Преобразуем в DataFrame
        df = pd.DataFrame(data)
        df.columns = ['applicationid'] + [f'audio_feature_{i}' for i in range(len(df.columns[1:]))]
        return df

