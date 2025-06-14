import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from  lightgbm import LGBMClassifier
from collections import Counter
from sklearn.metrics import precision_recall_curve, auc
from lightgbm import LGBMClassifier
import yaml

from TableDataProcessor import TableDataProcessor
def pr_auc(y, pred):
    precision, recall, _ = precision_recall_curve(y, pred)
    return auc(recall, precision)

def main():
    df1 = pd.read_parquet('./data/test_amplitude_chunk_00.parquet')
    df1 = TableDataProcessor(df1).create_features()

    df2 = pd.read_parquet('./data/test_amplitude_chunk_00.parquet')
    df2 = TableDataProcessor(df2).create_features()

    df_test = pd.concat([df1, df2])

    application_data = pd.read_parquet('./data/test_app_data.parquet')

    df_test.columns = df_test.columns.str.lower()
    application_data.columns = application_data.columns.str.lower()

    df_test = pd.merge(df_test, application_data, how='outer', on=['applicationid'])

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    df_test = df_test.drop(config['drop_cols'], axis=1)