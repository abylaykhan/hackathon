import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from  lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import precision_recall_curve, auc
from lightgbm import LGBMClassifier
import yaml

pd.set_option('display.max_columns', None)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


df_test = pd.read_parquet('./amplituda_features.parquet')

application_data = pd.read_parquet('./valid_app_data.parquet')

df_test.columns = df_test.columns.str.lower()
application_data.columns = application_data.columns.str.lower()

df_test = pd.merge(df_test, application_data, how='outer', on=['applicationid'])

target_data = pd.read_parquet('./valid_target_df.parquet')
target_data.columns = target_data.columns.str.lower()

        
df_test = df_test.rename(columns=lambda x: x.strip())
df_test.columns = df_test.columns.str.upper()
df_test['TOTALAMOUNT'] = df_test['TOTALAMOUNT'].str.replace(' ', '').astype('float64')
df_test['SUM_CREDIT_KZT'] = df_test['SUM_CREDIT_KZT'].str.replace(' ', '').astype('float64')
df_test['DM5DPD1GCVPSUM'] = (df_test['DM5DPD1GCVPSUM'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test['DM5EXPSUM'] = (df_test['DM5EXPSUM'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test['DM5INCSUM'] = (df_test['DM5INCSUM'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test['DM6SCOREN6PD'] = (df_test['DM6SCOREN6PD'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test['DM6SCOREN6'] = (df_test['DM6SCOREN6'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test['FINALKDN'] = (df_test['FINALKDN'].str.replace(' ', '').str.replace(',', '.').str.replace('-', '0')).astype('float64')
df_test.columns = df_test.columns.str.lower()

id_cols = ['applicationid', 'create_date', 'create_datetime', 'data_issue',\
            'vintage', 'product_group', 'regregion', 'company_name', 'spf', \
                'mng_name_login_init', 'mng_name_init', 'regtown', 'birthcountry', 'regcounty', 'target']

id_cols += ['last_event_type', 'second_last_event_type', 'most_common_device_type']

with open("feature_mapping.yaml", "r") as file:
    feature_mapping = yaml.safe_load(file)

for key in feature_mapping:
    df_test[key] = df_test[key].map(feature_mapping[key])

audio_pd = pd.read_parquet('validation_audio_pd.parquet')
audio_pd.columns = audio_pd.columns.str.lower()
df_test = pd.merge(df_test, audio_pd, how='left', on=['applicationid'])
df_test = df_test.drop_duplicates(subset=['applicationid'], keep='first')

import pickle
scaler = pickle.load(open('scaler.pkl', 'rb'))
X = pd.DataFrame(scaler.transform(df_test[config['scaler_cols']]), columns=config['scaler_cols'])
model = pickle.load(open('model.pkl', 'rb'))

df_test['FINAL_PD'] = model.predict_proba(X[config['final_cols']])[:,1]
df_test['PRED'] = model.predict(X[config['final_cols']])
df_test[['applicationid', 'FINAL_PD', 'PRED']].to_csv('results.csv', index=False)